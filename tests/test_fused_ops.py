import importlib.util
import subprocess
import sys

import numpy as np
import pytest

import kernelleaf as kl


TRITON_CUDA = kl.is_cuda_available() and importlib.util.find_spec("triton") is not None


def _tensor(value, device, requires_grad=True):
    return kl.Tensor(value, device=device, requires_grad=requires_grad)


def _output_width(operation, shape):
    return 29 if operation == "linear" else shape[-1]


def _run(operation, implementation, shape, dtype):
    rng = np.random.default_rng(120 + sum(shape))
    device = kl.cuda(0)
    x = _tensor(rng.normal(size=shape).astype(dtype), device)
    output_shape = shape[:-1] + (_output_width(operation, shape),)
    upstream_values = rng.normal(size=output_shape).astype(dtype)

    if operation == "linear":
        output_width = _output_width(operation, shape)
        weight = _tensor(rng.normal(size=(shape[-1], output_width)).astype(dtype), device)
        bias = _tensor(rng.normal(size=(1, output_width)).astype(dtype), device)
        op = kl.ops.Linear(implementation)
        inputs = (x, weight, bias)
    elif operation == "softmax":
        op = kl.ops.Softmax(-1, implementation)
        inputs = (x,)
    elif operation == "layernorm":
        weight = _tensor(rng.normal(size=(shape[-1],)).astype(dtype), device)
        bias = _tensor(rng.normal(size=(shape[-1],)).astype(dtype), device)
        op = kl.ops.LayerNorm(1e-5, implementation)
        inputs = (x, weight, bias)
    else:
        weight = _tensor(rng.normal(size=(shape[-1],)).astype(dtype), device)
        op = kl.ops.RMSNorm(1e-5, implementation)
        inputs = (x, weight)

    output = op(*inputs)
    upstream = _tensor(upstream_values, device, requires_grad=False)
    (output * upstream).sum().backward()
    return output.numpy(), [value.grad.numpy() for value in inputs], op.selected_implementation


def _assert_results_close(eager, triton, dtype):
    rtol, atol = ((3e-2, 3e-2) if dtype == np.float16 else (3e-4, 3e-5))
    np.testing.assert_allclose(eager[0], triton[0], rtol=rtol, atol=atol)
    for eager_grad, triton_grad in zip(eager[1], triton[1]):
        np.testing.assert_allclose(eager_grad, triton_grad, rtol=rtol, atol=atol)
    assert triton[2] == "triton"


def test_cpu_import_and_auto_dispatch_do_not_load_optional_runtimes():
    script = """
import sys
import numpy as np
import kernelleaf as kl
x = kl.Tensor(np.ones((2, 3), dtype=np.float32))
assert kl.ops.Softmax(implementation='auto')(x).shape == (2, 3)
assert 'cupy' not in sys.modules
assert 'triton' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_cpu_auto_uses_eager_and_forced_triton_is_clear():
    x = kl.Tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
    operation = kl.ops.Softmax(implementation="auto")
    operation(x)
    assert operation.selected_implementation == "eager"
    with pytest.raises(RuntimeError, match="requires CUDA arrays"):
        kl.ops.softmax(x, implementation="triton")


def test_public_modules_expose_dispatch_and_norm_shapes_on_cpu():
    x = kl.Tensor(np.arange(24, dtype=np.float32).reshape(3, 8))
    modules = (
        kl.nn.Linear(8, 5, implementation="auto"),
        kl.nn.Softmax(implementation="auto"),
        kl.nn.LayerNorm(8, implementation="auto"),
        kl.nn.RMSNorm(8, implementation="auto"),
    )
    assert modules[0](x).shape == (3, 5)
    for module in modules[1:]:
        assert module(x).shape == x.shape
    assert all(module.last_implementation == "eager" for module in modules)


@pytest.mark.skipif(not TRITON_CUDA, reason="working CUDA/CuPy/Triton unavailable")
@pytest.mark.parametrize("operation", ["linear", "softmax", "layernorm", "rmsnorm"])
@pytest.mark.parametrize("shape", [(2, 7), (17, 37), (8, 1003)], ids=["small", "irregular", "large"])
def test_float32_triton_forward_and_backward_match_eager(operation, shape):
    _assert_results_close(
        _run(operation, "eager", shape, np.float32),
        _run(operation, "triton", shape, np.float32),
        np.float32,
    )


@pytest.mark.skipif(not TRITON_CUDA, reason="working CUDA/CuPy/Triton unavailable")
@pytest.mark.parametrize("operation", ["linear", "softmax", "layernorm", "rmsnorm"])
def test_float16_triton_forward_and_backward_match_eager(operation):
    _assert_results_close(
        _run(operation, "eager", (11, 67), np.float16),
        _run(operation, "triton", (11, 67), np.float16),
        np.float16,
    )


@pytest.mark.skipif(not TRITON_CUDA, reason="working CUDA/CuPy/Triton unavailable")
def test_triton_execution_has_no_torch_runtime_dependency():
    script = """
import sys
import numpy as np
import kernelleaf as kl
x = kl.Tensor(np.ones((2, 17), dtype=np.float32), device=kl.cuda(0))
kl.ops.softmax(x, implementation='triton').numpy()
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.skipif(not TRITON_CUDA, reason="working CUDA/CuPy/Triton unavailable")
def test_triton_contract_errors_and_auto_fallback():
    device = kl.cuda(0)
    x64 = kl.Tensor(np.ones((2, 3), dtype=np.float64), device=device)
    operation = kl.ops.Softmax(implementation="auto")
    operation(x64)
    assert operation.selected_implementation == "eager"
    with pytest.raises(RuntimeError, match="supports float16 and float32"):
        kl.ops.softmax(x64, implementation="triton")

    x = kl.Tensor(np.ones((2, 3), dtype=np.float32), device=device)
    operation = kl.ops.Softmax(axis=0, implementation="auto")
    operation(x)
    assert operation.selected_implementation == "eager"
    with pytest.raises(RuntimeError, match="only the last axis"):
        kl.ops.softmax(x, axis=0, implementation="triton")

    noncontiguous = kl.ops.transpose(x)
    with pytest.raises(RuntimeError, match="requires contiguous arrays"):
        kl.ops.softmax(noncontiguous, implementation="triton")

    oversized = kl.Tensor(np.ones((1, 65537), dtype=np.float32), device=device)
    with pytest.raises(RuntimeError, match="fused block limit"):
        kl.ops.softmax(oversized, implementation="triton")

    three_dimensional = kl.Tensor(
        np.ones((2, 3, 4), dtype=np.float32), device=device
    )
    operation = kl.ops.Linear("auto")
    weight = kl.Tensor(np.ones((4, 5), dtype=np.float32), device=device)
    operation(three_dimensional, weight)
    assert operation.selected_implementation == "eager"
    with pytest.raises(RuntimeError, match="requires ndim"):
        kl.ops.linear(three_dimensional, weight, implementation="triton")
