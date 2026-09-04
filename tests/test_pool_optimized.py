import numpy as np
import pytest

import mytorch as mt


DEVICES = [mt.cpu()]
if mt.is_cuda_available():
    DEVICES.append(mt.cuda(0))


def _pool_reference(values, upstream, kernel, stride):
    batch, channels, height, width = values.shape
    out_h = (height - kernel) // stride + 1
    out_w = (width - kernel) // stride + 1
    output = np.empty((batch, channels, out_h, out_w), dtype=values.dtype)
    gradient = np.zeros_like(values)
    for b in range(batch):
        for c in range(channels):
            for out_y in range(out_h):
                for out_x in range(out_w):
                    y = out_y * stride
                    x = out_x * stride
                    window = values[b, c, y:y + kernel, x:x + kernel]
                    flat_index = int(np.argmax(window))
                    output[b, c, out_y, out_x] = window.reshape(-1)[flat_index]
                    gradient[
                        b, c, y + flat_index // kernel, x + flat_index % kernel
                    ] += upstream[b, c, out_y, out_x]
    return output, gradient


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize(("kernel", "stride"), [(2, 2), (3, 1)])
def test_vectorized_maxpool_forward_backward_matches_reference(
    device, kernel, stride
):
    rng = np.random.default_rng(91 + kernel + stride)
    values = rng.normal(size=(2, 3, 7, 8)).astype(np.float32)
    out_h = (values.shape[2] - kernel) // stride + 1
    out_w = (values.shape[3] - kernel) // stride + 1
    upstream = rng.normal(size=(2, 3, out_h, out_w)).astype(np.float32)
    expected_output, expected_gradient = _pool_reference(
        values, upstream, kernel, stride
    )

    x = mt.Tensor(values, device=device, requires_grad=True)
    output = mt.ops.max_pool2d(x, kernel_size=kernel, stride=stride)
    (output * mt.Tensor(upstream, device=device, requires_grad=False)).sum().backward()

    np.testing.assert_allclose(output.numpy(), expected_output)
    np.testing.assert_allclose(x.grad.numpy(), expected_gradient, rtol=1e-6, atol=1e-6)


def test_maxpool_validates_shape_kernel_and_stride():
    with pytest.raises(ValueError, match="NCHW"):
        mt.ops.max_pool2d(mt.Tensor(np.ones((2, 3, 4))), 2, 2)
    with pytest.raises(ValueError, match="positive"):
        mt.ops.max_pool2d(mt.Tensor(np.ones((1, 1, 4, 4))), 0, 1)
    with pytest.raises(ValueError, match="fit"):
        mt.ops.max_pool2d(mt.Tensor(np.ones((1, 1, 2, 2))), 3, 1)


def test_maxpool_ties_route_gradient_to_first_maximum():
    x = mt.Tensor(np.ones((1, 1, 2, 2), dtype=np.float32), requires_grad=True)
    mt.ops.max_pool2d(x, kernel_size=2, stride=2).sum().backward()
    np.testing.assert_array_equal(
        x.grad.numpy(), np.array([[[[1, 0], [0, 0]]]], dtype=np.float32)
    )


@pytest.mark.skipif(not mt.is_cuda_available(), reason="working CUDA/CuPy unavailable")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_cuda_col2im_raw_kernel_supports_float_dtypes(dtype):
    rng = np.random.default_rng(22)
    values = rng.normal(size=(1, 2, 5, 6)).astype(dtype)
    weights = rng.normal(size=(3, 2, 3, 2)).astype(dtype)
    upstream = rng.normal(size=(1, 3, 3, 5)).astype(dtype)

    def run(implementation):
        x = mt.Tensor(values, device=mt.cuda(0), requires_grad=True)
        weight = mt.Tensor(weights, device=mt.cuda(0), requires_grad=True)
        output = mt.ops.Conv2d(implementation=implementation)(x, weight)
        (output * mt.Tensor(
            upstream, device=mt.cuda(0), requires_grad=False
        )).sum().backward()
        return output.numpy(), x.grad.numpy(), weight.grad.numpy()

    naive = run("naive")
    optimized = run("im2col")
    tolerance = 2e-2 if dtype == np.float16 else 2e-5
    for expected, actual in zip(naive, optimized):
        np.testing.assert_allclose(
            actual, expected, rtol=tolerance, atol=tolerance
        )
