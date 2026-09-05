import os
import subprocess
import sys

import numpy as np
import pytest

import kernelleaf as kl


def test_import_does_not_eagerly_import_cupy():
    code = "import sys, kernelleaf; print(int('cupy' in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.stdout.strip() == "0"


def test_cpu_backend_factories_and_helpers():
    values = kl.asarray([1, 2, 3], device=kl.cpu(), dtype="float64")
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float64
    assert kl.get_array_module(values) is np
    np.testing.assert_array_equal(kl.asnumpy(values), [1, 2, 3])
    assert kl.zeros(2, 3).shape == (2, 3)
    assert kl.ones(2).dtype == np.float32


def test_unavailable_cuda_has_actionable_error():
    if kl.is_cuda_available():
        pytest.skip("CUDA is available")
    with pytest.raises(RuntimeError, match="CuPy|CUDA"):
        kl.cuda(0)


@pytest.mark.skipif(not kl.is_cuda_available(), reason="working CUDA/CuPy unavailable")
def test_cuda_transfer_arithmetic_and_device_mismatch():
    gpu = kl.cuda(0)
    x_cpu = kl.Tensor(np.array([1, 2, 3], dtype=np.float32))
    x_gpu = x_cpu.to(gpu)
    assert x_gpu.device == gpu
    assert kl.get_array_module(x_gpu).__name__ == "cupy"

    y_gpu = ((x_gpu + 2) * x_gpu).sum()
    y_gpu.backward()
    np.testing.assert_allclose(y_gpu.numpy(), 26.0)
    np.testing.assert_allclose(x_gpu.grad.numpy(), [4, 6, 8])
    np.testing.assert_allclose(x_gpu.cpu().numpy(), x_cpu.numpy())

    with pytest.raises(ValueError, match="device mismatch.*cpu.*cuda:0"):
        _ = x_cpu + x_gpu
