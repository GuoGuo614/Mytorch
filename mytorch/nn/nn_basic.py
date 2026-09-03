"""The module.
"""
from typing import Any, Optional
from mytorch.autograd import Tensor
from mytorch import ops
import mytorch.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Optional[Any] = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        kwargs = {'device': device, 'dtype': dtype}
        self.has_bias = bias

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, **kwargs))
        self.bias = (
            Parameter(
                init.kaiming_uniform(out_features, 1, **kwargs).reshape(
                    (1, out_features)
                )
            )
            if bias
            else None
        )

    def forward(self, X: Tensor) -> Tensor:
        result = ops.matmul(X, self.weight)

        if self.has_bias:
            bias = ops.broadcast_to(self.bias, result.shape)
            result += bias

        return result


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        batch_size = X.shape[0]
        total_features = 1
        for i in range(1, len(X.shape)):
            total_features *= X.shape[i]

        return ops.reshape(X, (batch_size, total_features))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        lse = ops.logsumexp(logits, axes=(1,))

        batch_size, num_classes = logits.shape
        y_one_hot = init.one_hot(num_classes, y)

        selected_logits = ops.summation(logits * y_one_hot, axes=(1,))
        loss_per_sample = lse - selected_logits
        return ops.summation(loss_per_sample) / batch_size


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Optional[Any] = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        kwargs = {'device': device, 'dtype': dtype}

        # Learnable parameters
        self.weight = Parameter(init.ones(dim, 1, **kwargs))
        self.bias = Parameter(init.zeros(dim, 1, **kwargs))

        # Running statistics (not parameters, so not using Parameter)
        self.running_mean = init.zeros(dim, 1, **kwargs)
        self.running_var = init.ones(dim, 1, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, features = x.shape

        if self.training:
            # Compute batch statistics
            mean = ops.summation(x, axes=(0,)) / batch_size
            mean_broadcasted = ops.broadcast_to(ops.reshape(mean, (1, features)), x.shape)

            x_centered = x - mean_broadcasted
            variance = ops.summation(x_centered * x_centered, axes=(0,)) / batch_size

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance.data
        else:
            # Use running statistics during evaluation
            mean = self.running_mean
            variance = self.running_var
            mean_broadcasted = ops.broadcast_to(ops.reshape(mean, (1, features)), x.shape)
            x_centered = x - mean_broadcasted

        # Normalize
        std = ops.power_scalar(variance + self.eps, 0.5)
        std_broadcasted = ops.broadcast_to(ops.reshape(std, (1, features)), x.shape)
        x_normalized = ops.divide(x_centered, std_broadcasted)

        # Scale and shift
        weight_broadcasted = ops.broadcast_to(ops.reshape(self.weight, (1, features)), x.shape)
        bias_broadcasted = ops.broadcast_to(ops.reshape(self.bias, (1, features)), x.shape)

        return weight_broadcasted * x_normalized + bias_broadcasted



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Optional[Any] = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        kwargs = {'device': device, 'dtype': dtype}

        self.weight = Parameter(init.ones(dim, 1, **kwargs))
        self.bias = Parameter(init.zeros(dim, 1, **kwargs))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, features = x.shape

        mean = ops.summation(x, axes=(1,)) / features
        mean_broadcasted = ops.broadcast_to(ops.reshape(mean, (batch_size, 1)), x.shape)

        x_centered = x - mean_broadcasted  # x - mean
        variance = ops.summation(x_centered * x_centered, axes=(1,)) / features

        std = ops.power_scalar(variance + self.eps, 0.5)
        std_broadcasted = ops.broadcast_to(ops.reshape(std, (batch_size, 1)), x.shape)

        x_normalized = ops.divide(x_centered, std_broadcasted)

        weight_broadcasted = ops.broadcast_to(self.weight, x.shape)
        bias_broadcasted = ops.broadcast_to(self.bias, x.shape)

        return weight_broadcasted * x_normalized + bias_broadcasted


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=(1 - self.p), device=x.device, dtype=x.dtype)
            return x * mask / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x

class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True, device: Optional[Any] = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kwargs = {'device': device, 'dtype': dtype}
        self.has_bias = bias

        # Weight shape: (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        # 直接创建正确形状的权重
        bound = np.sqrt(6.0 / fan_in)
        self.weight = Parameter(init.rand(out_channels, in_channels, kernel_size, kernel_size,
                                         low=-bound, high=bound, **kwargs))

        if bias:
            bound_bias = 1.0 / np.sqrt(fan_in)
            self.bias = Parameter(init.rand(1, out_channels, 1, 1,
                                           low=-bound_bias, high=bound_bias, **kwargs))

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, in_channels, height, width)
        if self.padding > 0:
            x = ops.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        result = ops.conv2d(x, self.weight, stride=self.stride)

        if self.has_bias:
            bias = ops.broadcast_to(self.bias, result.shape)
            result += bias

        return result


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: Optional[int] = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, channels, height, width)
        return ops.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)


class AvgPool2d(Module):
    def __init__(self, kernel_size: int, stride: Optional[int] = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, channels, height, width)
        return ops.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)
