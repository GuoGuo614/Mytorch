"""The module.
"""
import math
from typing import Any, Optional
from kernelleaf.autograd import Tensor
from kernelleaf import ops
import kernelleaf.init as init
from kernelleaf.backend import Device, cpu, cuda, to_device


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

    def _named_tensors(self):
        seen = set()

        def walk(value, prefix):
            if isinstance(value, Tensor):
                if id(value) not in seen:
                    seen.add(id(value))
                    yield prefix, value
                return
            if isinstance(value, Module):
                for name, child in value.__dict__.items():
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    yield from walk(child, child_prefix)
                return
            if isinstance(value, dict):
                for name, child in value.items():
                    child_prefix = f"{prefix}.{name}" if prefix else str(name)
                    yield from walk(child, child_prefix)
                return
            if isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    child_prefix = f"{prefix}.{index}" if prefix else str(index)
                    yield from walk(child, child_prefix)

        yield from walk(self, "")

    def named_parameters(self):
        return [(name, tensor) for name, tensor in self._named_tensors()
                if isinstance(tensor, Parameter)]

    def parameters(self) -> list[Tensor]:
        """Return parameters once, preserving module traversal order."""
        return [parameter for _, parameter in self.named_parameters()]

    def named_buffers(self):
        return [(name, tensor) for name, tensor in self._named_tensors()
                if not isinstance(tensor, Parameter)]

    def to(self, device: Device):
        """Move parameters, persistent buffers, and existing gradients in place."""
        if not isinstance(device, Device):
            raise TypeError(f"device must be a Device, got {type(device).__name__}")
        for _, tensor in self._named_tensors():
            tensor.cached_data = to_device(
                tensor.realize_cached_data(), device, dtype=tensor.dtype
            )
            if getattr(tensor, "grad", None) is not None:
                tensor.grad = tensor.grad.to(device)
        return self

    def cpu(self):
        return self.to(cpu())

    def cuda(self, index=0):
        return self.to(cuda(index))

    def state_dict(self):
        """Return detached device-preserving copies of parameters and buffers."""
        state = {}
        for name, tensor in self._named_tensors():
            data = tensor.realize_cached_data().copy()
            state[name] = Tensor(
                data, device=tensor.device, dtype=tensor.dtype, requires_grad=False
            )
        return state

    def load_state_dict(self, state_dict, strict=True):
        current = dict(self._named_tensors())
        missing = sorted(set(current) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(current))
        if strict and (missing or unexpected):
            raise KeyError(
                f"state_dict mismatch: missing={missing}, unexpected={unexpected}"
            )
        for name, destination in current.items():
            if name not in state_dict:
                continue
            source = state_dict[name]
            source_data = (
                source.realize_cached_data() if isinstance(source, Tensor) else source
            )
            if tuple(source_data.shape) != tuple(destination.shape):
                raise ValueError(
                    f"shape mismatch for {name}: expected {destination.shape}, "
                    f"got {source_data.shape}"
                )
            destination.cached_data = to_device(
                source_data, destination.device, dtype=destination.dtype
            )
        return {"missing_keys": missing, "unexpected_keys": unexpected}

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
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: Optional[Any] = None, dtype: str = "float32",
                 implementation: str = "auto") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        kwargs = {'device': device, 'dtype': dtype}
        self.has_bias = bias
        self.implementation = implementation
        self.last_implementation = None

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
        operation = ops.Linear(self.implementation)
        result = (
            operation(X, self.weight, self.bias)
            if self.has_bias else operation(X, self.weight)
        )
        self.last_implementation = operation.selected_implementation
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


class Softmax(Module):
    def __init__(self, axis=-1, implementation="auto"):
        super().__init__()
        self.axis = axis
        self.implementation = implementation
        self.last_implementation = None

    def forward(self, x: Tensor) -> Tensor:
        operation = ops.Softmax(self.axis, self.implementation)
        result = operation(x)
        self.last_implementation = operation.selected_implementation
        return result

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
        if logits.device != y.device:
            raise ValueError(
                f"device mismatch: logits are on {logits.device}, labels are on {y.device}"
            )
        lse = ops.logsumexp(logits, axes=(1,))

        batch_size, num_classes = logits.shape
        y_one_hot = init.one_hot(
            num_classes, y, device=logits.device, dtype=logits.dtype
        )

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
        self.weight = Parameter(init.ones(dim, **kwargs))
        self.bias = Parameter(init.zeros(dim, **kwargs))

        # Running statistics (not parameters, so not using Parameter)
        self.running_mean = init.zeros(dim, **kwargs)
        self.running_var = init.ones(dim, **kwargs)

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


class BatchNorm2d(Module):
    """Channel-wise batch normalization for NCHW feature maps."""

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, device: Optional[Any] = None,
                 dtype: str = "float32") -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must be in [0, 1]")
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        kwargs = {"device": device, "dtype": dtype}
        self.weight = Parameter(init.ones(num_features, **kwargs))
        self.bias = Parameter(init.zeros(num_features, **kwargs))
        self.running_mean = init.zeros(num_features, **kwargs)
        self.running_var = init.ones(num_features, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 4 or x.shape[1] != self.num_features:
            raise ValueError(
                "BatchNorm2d expects NCHW input with "
                f"{self.num_features} channels, got {x.shape}"
            )
        batch, channels, height, width = x.shape
        broadcast_shape = (1, channels, 1, 1)
        if self.training:
            count = batch * height * width
            mean = ops.summation(x, axes=(0, 2, 3)) / count
            centered = x - ops.broadcast_to(
                ops.reshape(mean, broadcast_shape), x.shape
            )
            variance = ops.summation(
                centered * centered, axes=(0, 2, 3)
            ) / count
            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean.data
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * variance.data
            )
        else:
            mean = self.running_mean
            variance = self.running_var
            centered = x - ops.broadcast_to(
                ops.reshape(mean, broadcast_shape), x.shape
            )
        inverse_std = ops.power_scalar(variance + self.eps, -0.5)
        normalized = centered * ops.broadcast_to(
            ops.reshape(inverse_std, broadcast_shape), x.shape
        )
        weight = ops.broadcast_to(
            ops.reshape(self.weight, broadcast_shape), x.shape
        )
        bias = ops.broadcast_to(
            ops.reshape(self.bias, broadcast_shape), x.shape
        )
        return normalized * weight + bias



class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5,
                 device: Optional[Any] = None, dtype: str = "float32",
                 implementation: str = "auto") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.implementation = implementation
        self.last_implementation = None
        kwargs = {'device': device, 'dtype': dtype}

        self.weight = Parameter(init.ones(dim, **kwargs))
        self.bias = Parameter(init.zeros(dim, **kwargs))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"LayerNorm expected last dimension {self.dim}, got {x.shape[-1]}"
            )
        operation = ops.LayerNorm(self.eps, self.implementation)
        result = operation(x, self.weight, self.bias)
        self.last_implementation = operation.selected_implementation
        return result


class LayerNorm1d(LayerNorm):
    """Backward-compatible name for last-dimension LayerNorm."""


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5,
                 device: Optional[Any] = None, dtype: str = "float32",
                 implementation: str = "auto") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.implementation = implementation
        self.last_implementation = None
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"RMSNorm expected last dimension {self.dim}, got {x.shape[-1]}"
            )
        operation = ops.RMSNorm(self.eps, self.implementation)
        result = operation(x, self.weight)
        self.last_implementation = operation.selected_implementation
        return result


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
    """NCHW convolution with OIHW weights."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, bias: bool = True,
                 device: Optional[Any] = None, dtype: str = "float32",
                 implementation: str = "auto",
                 max_im2col_bytes: int = ops.DEFAULT_IM2COL_MAX_BYTES) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.implementation = implementation
        self.max_im2col_bytes = max_im2col_bytes
        self.last_implementation = None
        self.last_backward_implementation = None
        self.last_chunk_rows = None

        kwargs = {'device': device, 'dtype': dtype}
        self.has_bias = bias

        # Weight shape: (out_channels, in_channels, kernel_size, kernel_size)
        if isinstance(kernel_size, int):
            kernel_h, kernel_w = kernel_size, kernel_size
        elif isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
            kernel_h, kernel_w = int(kernel_size[0]), int(kernel_size[1])
        else:
            raise TypeError("kernel_size must be an int or a pair of ints")
        if kernel_h <= 0 or kernel_w <= 0:
            raise ValueError("kernel_size values must be positive")
        fan_in = in_channels * kernel_h * kernel_w

        # 直接创建正确形状的权重
        bound = math.sqrt(6.0 / fan_in)
        self.weight = Parameter(init.rand(out_channels, in_channels, kernel_h, kernel_w,
                                         low=-bound, high=bound, **kwargs))

        if bias:
            bound_bias = 1.0 / math.sqrt(fan_in)
            self.bias = Parameter(init.rand(1, out_channels, 1, 1,
                                           low=-bound_bias, high=bound_bias, **kwargs))

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, in_channels, height, width)
        if isinstance(self.padding, int):
            padding = (self.padding, self.padding)
        else:
            padding = tuple(self.padding)
        if len(padding) != 2 or min(padding) < 0:
            raise ValueError("padding must be a non-negative int or pair")
        if padding != (0, 0):
            x = ops.pad(
                x,
                ((0, 0), (0, 0), (padding[0], padding[0]),
                 (padding[1], padding[1])),
            )

        conv_op = ops.Conv2d(
            stride=self.stride,
            implementation=self.implementation,
            max_im2col_bytes=self.max_im2col_bytes,
        )
        result = conv_op(x, self.weight)
        self.last_implementation = conv_op.selected_implementation
        self.last_backward_implementation = (
            "im2col" if conv_op.selected_implementation == "triton"
            else conv_op.selected_implementation
        )
        self.last_chunk_rows = conv_op.chunk_rows

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


class AdaptiveAvgPool2d(Module):
    """Adaptive average pooling; V7 currently supports global output only."""

    def __init__(self, output_size=1) -> None:
        super().__init__()
        if output_size not in {1, (1, 1)}:
            raise NotImplementedError(
                "AdaptiveAvgPool2d currently supports output_size=1 only"
            )
        self.output_size = (1, 1)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"AdaptiveAvgPool2d expects NCHW input, got {x.shape}")
        batch, channels, height, width = x.shape
        pooled = ops.summation(x, axes=(2, 3)) / (height * width)
        return ops.reshape(pooled, (batch, channels, 1, 1))


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)
