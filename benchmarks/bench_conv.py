"""Benchmark Conv2d naive, im2col, and auto forward implementations."""

import argparse
import json
import math
import statistics
import time
import tracemalloc

import numpy as np

import mytorch as mt
import mytorch.nn as nn


def _synchronize(device):
    if device.kind == "cuda":
        device.xp.cuda.get_current_stream().synchronize()


def _percentile(values, percentile):
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _measure_peak_memory(layer, inputs, device):
    if device.kind == "cuda":
        pool = device.xp.get_default_memory_pool()
        pool.free_all_blocks()
        baseline = pool.total_bytes()
        output = layer(inputs)
        _synchronize(device)
        peak = max(0, pool.total_bytes() - baseline)
    else:
        tracemalloc.start()
        output = layer(inputs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return output.shape, peak


def benchmark_one(args, implementation, device):
    np.random.seed(args.seed)
    batch, in_channels, height, width = args.shape
    inputs = mt.Tensor(
        np.random.randn(batch, in_channels, height, width).astype(args.dtype),
        device=device,
        requires_grad=False,
    )
    layer = nn.Conv2d(
        in_channels,
        args.out_channels,
        args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        bias=args.bias,
        device=device,
        dtype=args.dtype,
        implementation=implementation,
        max_im2col_bytes=args.max_im2col_bytes,
    )
    for _ in range(args.warmup):
        layer(inputs)
    _synchronize(device)

    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        output = layer(inputs)
        _synchronize(device)
        durations.append((time.perf_counter() - started) * 1000)
    output_shape, peak_memory = _measure_peak_memory(layer, inputs, device)
    return {
        "input_shape": list(args.shape),
        "weight_shape": list(layer.weight.shape),
        "output_shape": list(output_shape),
        "requested_implementation": implementation,
        "selected_implementation": layer.last_implementation,
        "device": str(device),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "median_ms": statistics.median(durations),
        "p95_ms": _percentile(durations, 0.95),
        # CPU uses tracemalloc; CUDA uses the CuPy pool's retained high-water delta.
        "peak_memory_bytes": peak_memory,
        "im2col_chunk_rows": layer.last_chunk_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=4, type=int, default=(2, 3, 16, 20),
                        metavar=("N", "C", "H", "W"))
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--implementation", choices=("all", "naive", "im2col", "auto"),
                        default="all")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-im2col-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.shape = tuple(args.shape)
    device = mt.cuda(0) if args.device == "cuda" else mt.cpu()
    implementations = (
        ("naive", "im2col", "auto")
        if args.implementation == "all" else (args.implementation,)
    )
    for implementation in implementations:
        print(json.dumps(benchmark_one(args, implementation, device)))


if __name__ == "__main__":
    main()
