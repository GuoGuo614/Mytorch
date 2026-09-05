"""Benchmark V4 eager/Triton fused forwards and eager backwards on CUDA."""

import argparse
import json
import math
import statistics
import time

import numpy as np

import kernelleaf as kl


OPERATORS = ("linear", "softmax", "layernorm", "rmsnorm")


def _percentile(values, percentile):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _sync(device):
    device.xp.cuda.get_current_stream().synchronize()


def _make_case(name, args, implementation, requires_grad):
    rng = np.random.default_rng(args.seed)
    device = kl.cuda(0)
    shape = (args.rows, args.columns)

    def tensor(values, grad=requires_grad):
        return kl.Tensor(values.astype(args.dtype), device=device, requires_grad=grad)

    x = tensor(rng.normal(size=shape))
    if name == "linear":
        weight = tensor(rng.normal(size=(args.columns, args.out_features)))
        bias = tensor(rng.normal(size=(1, args.out_features)))
        op = kl.ops.Linear(implementation)
        inputs = (x, weight, bias)
    elif name == "softmax":
        op = kl.ops.Softmax(-1, implementation)
        inputs = (x,)
    elif name == "layernorm":
        weight = tensor(rng.normal(size=(args.columns,)))
        bias = tensor(rng.normal(size=(args.columns,)))
        op = kl.ops.LayerNorm(args.epsilon, implementation)
        inputs = (x, weight, bias)
    else:
        weight = tensor(rng.normal(size=(args.columns,)))
        op = kl.ops.RMSNorm(args.epsilon, implementation)
        inputs = (x, weight)
    return op, inputs


def _time(call, device, warmup, repeats):
    for _ in range(warmup):
        call()
    _sync(device)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        _sync(device)
        durations.append((time.perf_counter() - started) * 1000)
    return statistics.median(durations), _percentile(durations, 0.95)


def benchmark_one(name, implementation, args):
    device = kl.cuda(0)
    forward_op, forward_inputs = _make_case(name, args, implementation, False)

    def forward():
        return forward_op(*forward_inputs)

    forward_median, forward_p95 = _time(forward, device, args.warmup, args.repeats)
    selected = forward_op.selected_implementation

    def forward_backward():
        operation, inputs = _make_case(name, args, implementation, True)
        operation(*inputs).sum().backward()

    training_median, training_p95 = _time(
        forward_backward, device, args.warmup, args.repeats
    )

    pool = device.xp.get_default_memory_pool()
    pool.free_all_blocks()
    baseline = pool.total_bytes()
    forward_backward()
    _sync(device)
    peak_memory = max(0, pool.total_bytes() - baseline)
    return {
        "operator": name,
        "shape": [args.rows, args.columns],
        "out_features": args.out_features if name == "linear" else None,
        "dtype": args.dtype,
        "device": str(device),
        "requested_implementation": implementation,
        "selected_forward_implementation": selected,
        "backward_implementation": "eager",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "forward_median_ms": forward_median,
        "forward_p95_ms": forward_p95,
        "forward_backward_median_ms": training_median,
        "forward_backward_p95_ms": training_p95,
        "allocator_peak_bytes": peak_memory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("all",) + OPERATORS, default="all")
    parser.add_argument(
        "--implementation", choices=("all", "eager", "triton", "auto"), default="all"
    )
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not kl.is_cuda_available():
        parser.error("a working CUDA/CuPy device is required")
    operators = OPERATORS if args.operator == "all" else (args.operator,)
    implementations = (
        ("eager", "triton", "auto")
        if args.implementation == "all" else (args.implementation,)
    )
    for name in operators:
        for implementation in implementations:
            print(json.dumps(benchmark_one(name, implementation, args)))


if __name__ == "__main__":
    main()
