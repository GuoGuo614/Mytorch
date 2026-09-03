import sys

import numpy as np

sys.path.append("./python")

import needle as ndl
import needle.nn as nn


DEVICE = ndl.cpu_numpy()


def tensor(data, requires_grad=True):
    return ndl.Tensor(np.asarray(data, dtype=np.float32), device=DEVICE,
                      requires_grad=requires_grad)


def test_pad_forward_backward():
    x = tensor([[1, 2], [3, 4]])
    y = ndl.ops.pad(x, ((1, 0), (0, 1)))
    np.testing.assert_array_equal(
        y.numpy(), np.array([[0, 0, 0], [1, 2, 0], [3, 4, 0]], dtype=np.float32)
    )
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.ones((2, 2), dtype=np.float32))


def test_max_pool2d_forward_backward():
    x = tensor([[[[1, 2, 3, 4], [5, 6, 7, 8],
                  [9, 10, 11, 12], [13, 14, 15, 16]]]])
    y = nn.MaxPool2d(2)(x)
    np.testing.assert_array_equal(y.numpy(), [[[[6, 8], [14, 16]]]])
    y.sum().backward()
    expected = np.zeros((1, 1, 4, 4), dtype=np.float32)
    expected[0, 0, 1::2, 1::2] = 1
    np.testing.assert_array_equal(x.grad.numpy(), expected)


def test_avg_pool2d_overlapping_backward():
    x = tensor(np.arange(9).reshape(1, 1, 3, 3))
    y = nn.AvgPool2d(2, stride=1)(x)
    np.testing.assert_allclose(y.numpy(), [[[[2, 3], [5, 6]]]])
    y.sum().backward()
    expected = np.array(
        [[[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(x.grad.numpy(), expected)


def test_logsoftmax_forward_backward():
    values = np.array([[1, 2, 3], [-1, 0, 1]], dtype=np.float32)
    x = tensor(values)
    y = nn.LogSoftmax(axis=1)(x)
    expected = values - np.log(np.exp(values).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5, atol=1e-5)
    y.sum().backward()
    softmax = np.exp(values) / np.exp(values).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(x.grad.numpy(), 1 - 3 * softmax, rtol=1e-5, atol=1e-5)


def test_conv2d_alias_and_explicit_padding():
    layer = nn.Conv2d(1, 2, 3, padding=0, bias=False, device=DEVICE)
    x = tensor(np.ones((1, 1, 5, 5), dtype=np.float32))
    assert layer(x).shape == (1, 2, 3, 3)
