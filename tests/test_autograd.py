import numpy as np

import kernelleaf as kl


def test_tensor_preserves_dtype_and_cpu_device():
    x = kl.Tensor(np.array([1, 2, 3], dtype=np.float64))
    assert x.dtype == np.dtype("float64")
    assert x.device == kl.cpu()
    np.testing.assert_array_equal(x.numpy(), [1, 2, 3])


def test_shared_graph_topological_backward():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = kl.Tensor(values)
    loss = (x * x + x).sum()
    loss.backward()
    np.testing.assert_allclose(x.grad.numpy(), 2 * values + 1)


def test_tensor_to_same_device_is_noop():
    x = kl.Tensor([1.0, 2.0])
    assert x.to(kl.cpu()) is x
    assert x.cpu() is x
