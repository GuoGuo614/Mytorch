import numpy as np

import mytorch as mt
import mytorch.nn as nn


def test_linear_forward_backward():
    layer = nn.Linear(2, 2, device=mt.cpu(), dtype="float32")
    layer.weight.data = mt.Tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype="float32", requires_grad=False
    )
    layer.bias.data = mt.Tensor(
        [[0.5, -0.5]], dtype="float32", requires_grad=False
    )
    x = mt.Tensor([[1.0, 2.0]])
    y = layer(x)
    np.testing.assert_allclose(y.numpy(), [[7.5, 9.5]])
    y.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [[3.0, 7.0]])
    assert len(layer.parameters()) == 2
    assert all(parameter.grad is not None for parameter in layer.parameters())


def test_conv2d_small_forward_backward():
    x = mt.Tensor(np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3))
    weight = mt.Tensor(np.ones((1, 1, 2, 2), dtype=np.float32))
    y = mt.ops.conv2d(x, weight)
    np.testing.assert_allclose(y.numpy(), [[[[8, 12], [20, 24]]]])
    y.sum().backward()
    np.testing.assert_allclose(
        x.grad.numpy(), [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]
    )
    np.testing.assert_allclose(weight.grad.numpy(), [[[[8, 12], [20, 24]]]])
