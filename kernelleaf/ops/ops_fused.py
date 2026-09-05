"""TensorOp adapters for eager/Triton fused operations.

Only forward execution is fused in V4.  Backward deliberately uses transparent
TensorOp formulas (and GEMM for Linear), matching the MyTorch-1 design without
its Torch/DLPack bridge.
"""

import importlib

from ..autograd import Tensor, TensorOp
from ..backend import get_array_module
from ..kernels.runtime import resolve_implementation
from .ops_mathematic import broadcast_to, matmul, reshape, summation, transpose


def _normalize_axis(axis, dimensions):
    if dimensions == 0:
        raise ValueError("a reduction axis requires at least one dimension")
    normalized = axis % dimensions
    if not -dimensions <= axis < dimensions:
        raise ValueError(f"axis {axis} is out of bounds for {dimensions} dimensions")
    return normalized


def _sum_keepdim(value, axis):
    shape = list(value.shape)
    shape[axis] = 1
    return reshape(summation(value, axes=(axis,)), tuple(shape))


def _resolve_reduction(implementation, arrays, operation, axis):
    if axis != arrays[0].ndim - 1:
        if implementation == "triton":
            raise RuntimeError(
                f"cannot use implementation='triton' for {operation}: "
                "only the last axis is supported"
            )
        return "eager"
    return resolve_implementation(
        implementation,
        arrays,
        operation=operation,
        last_dim_reduction=True,
    )


class Linear(TensorOp):
    """Linear forward with optional Triton-fused bias; backward uses GEMMs."""

    def __init__(self, implementation="auto"):
        self.implementation = implementation
        self.selected_implementation = None

    def compute(self, x, weight, *bias_values):
        if x.ndim < 2 or weight.ndim != 2:
            raise ValueError("Linear expects input ndim >= 2 and a 2D weight")
        if x.shape[-1] != weight.shape[0]:
            raise ValueError(
                f"Linear shape mismatch: input {x.shape}, weight {weight.shape}"
            )
        bias = bias_values[0] if bias_values else None
        valid_bias_shapes = {(weight.shape[1],), (1, weight.shape[1])}
        if bias is not None and bias.shape not in valid_bias_shapes:
            raise ValueError(
                f"Linear bias must have shape ({weight.shape[1]},) or "
                f"(1, {weight.shape[1]}), got {bias.shape}"
            )
        arrays = (x, weight) if bias is None else (x, weight, bias)
        dtypes_match = all(array.dtype == x.dtype for array in arrays[1:])
        if not dtypes_match and self.implementation == "triton":
            raise RuntimeError(
                "cannot use implementation='triton' for Linear: input, weight, "
                "and bias must have the same dtype"
            )
        self.selected_implementation = (
            resolve_implementation(
                self.implementation,
                arrays,
                operation="Linear",
                dimensions={2},
            )
            if dtypes_match else "eager"
        )
        if self.selected_implementation == "triton":
            kernel = importlib.import_module("kernelleaf.kernels.linear")
            return kernel.forward(x, weight, bias)
        result = get_array_module(x).matmul(x, weight)
        return result if bias is None else result + bias

    def gradient(self, out_grad, node):
        x, weight, *bias_values = node.inputs
        grad_x = matmul(out_grad, transpose(weight))
        grad_weight = matmul(transpose(x), out_grad)
        if len(grad_weight.shape) > 2:
            grad_weight = summation(
                grad_weight, axes=tuple(range(len(grad_weight.shape) - 2))
            )
        if not bias_values:
            return grad_x, grad_weight
        bias = bias_values[0]
        axes = tuple(range(len(out_grad.shape) - 1))
        grad_bias = summation(out_grad, axes=axes).reshape(bias.shape)
        return grad_x, grad_weight, grad_bias


def linear(x, weight, bias=None, implementation="auto"):
    operation = Linear(implementation)
    return operation(x, weight) if bias is None else operation(x, weight, bias)


class Softmax(TensorOp):
    def __init__(self, axis=-1, implementation="auto"):
        self.axis = axis
        self.implementation = implementation
        self.selected_implementation = None

    def compute(self, x):
        axis = _normalize_axis(self.axis, x.ndim)
        self.selected_implementation = _resolve_reduction(
            self.implementation, (x,), "Softmax", axis
        )
        if self.selected_implementation == "triton":
            return importlib.import_module("kernelleaf.kernels.softmax").forward(x)
        xp = get_array_module(x)
        shifted = x - xp.max(x, axis=axis, keepdims=True)
        numerator = xp.exp(shifted)
        return numerator / xp.sum(numerator, axis=axis, keepdims=True)

    def gradient(self, out_grad, node):
        axis = _normalize_axis(self.axis, len(node.shape))
        weighted_sum = _sum_keepdim(out_grad * node, axis)
        return node * (out_grad - broadcast_to(weighted_sum, node.shape))


def softmax(x, axis=-1, implementation="auto"):
    return Softmax(axis, implementation)(x)


class LayerNorm(TensorOp):
    def __init__(self, epsilon=1e-5, implementation="auto"):
        self.epsilon = epsilon
        self.implementation = implementation
        self.selected_implementation = None

    def compute(self, x, weight, bias):
        _validate_norm_arrays(x, weight, bias, "LayerNorm")
        self.selected_implementation = _resolve_norm_implementation(
            self.implementation, (x, weight, bias), "LayerNorm"
        )
        if self.selected_implementation == "triton":
            kernel = importlib.import_module("kernelleaf.kernels.layernorm")
            return kernel.forward(x, weight, bias, self.epsilon)
        xp = get_array_module(x)
        mean = xp.mean(x, axis=-1, keepdims=True)
        variance = xp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / xp.sqrt(variance + self.epsilon) * weight + bias

    def gradient(self, out_grad, node):
        x, weight, _ = node.inputs
        columns = x.shape[-1]
        mean = _sum_keepdim(x, len(x.shape) - 1) / columns
        centered = x - broadcast_to(mean, x.shape)
        variance = _sum_keepdim(centered * centered, len(x.shape) - 1) / columns
        inverse_std = (variance + self.epsilon) ** -0.5
        normalized = centered * broadcast_to(inverse_std, x.shape)
        scaled_grad = out_grad * broadcast_to(weight, x.shape)
        sum_grad = _sum_keepdim(scaled_grad, len(x.shape) - 1)
        sum_grad_normalized = _sum_keepdim(
            scaled_grad * normalized, len(x.shape) - 1
        )
        grad_x = broadcast_to(inverse_std, x.shape) / columns * (
            columns * scaled_grad
            - broadcast_to(sum_grad, x.shape)
            - normalized * broadcast_to(sum_grad_normalized, x.shape)
        )
        leading_axes = tuple(range(len(x.shape) - 1))
        grad_weight = summation(out_grad * normalized, axes=leading_axes)
        grad_bias = summation(out_grad, axes=leading_axes)
        return grad_x, grad_weight, grad_bias


def layer_norm(x, weight, bias, epsilon=1e-5, implementation="auto"):
    return LayerNorm(epsilon, implementation)(x, weight, bias)


class RMSNorm(TensorOp):
    def __init__(self, epsilon=1e-5, implementation="auto"):
        self.epsilon = epsilon
        self.implementation = implementation
        self.selected_implementation = None

    def compute(self, x, weight):
        _validate_norm_arrays(x, weight, None, "RMSNorm")
        self.selected_implementation = _resolve_norm_implementation(
            self.implementation, (x, weight), "RMSNorm"
        )
        if self.selected_implementation == "triton":
            kernel = importlib.import_module("kernelleaf.kernels.rmsnorm")
            return kernel.forward(x, weight, self.epsilon)
        xp = get_array_module(x)
        mean_square = xp.mean(x * x, axis=-1, keepdims=True)
        return x / xp.sqrt(mean_square + self.epsilon) * weight

    def gradient(self, out_grad, node):
        x, weight = node.inputs
        columns = x.shape[-1]
        mean_square = _sum_keepdim(x * x, len(x.shape) - 1) / columns
        inverse_rms = (mean_square + self.epsilon) ** -0.5
        scaled_grad = out_grad * broadcast_to(weight, x.shape)
        projection = _sum_keepdim(scaled_grad * x, len(x.shape) - 1)
        grad_x = scaled_grad * broadcast_to(inverse_rms, x.shape) - (
            x
            * broadcast_to(inverse_rms ** 3, x.shape)
            * broadcast_to(projection, x.shape)
            / columns
        )
        normalized = x * broadcast_to(inverse_rms, x.shape)
        leading_axes = tuple(range(len(x.shape) - 1))
        grad_weight = summation(out_grad * normalized, axes=leading_axes)
        return grad_x, grad_weight


def rms_norm(x, weight, epsilon=1e-5, implementation="auto"):
    return RMSNorm(epsilon, implementation)(x, weight)


def _validate_norm_arrays(x, weight, bias, operation):
    if x.ndim < 1:
        raise ValueError(f"{operation} input must have at least one dimension")
    if weight.shape != (x.shape[-1],):
        raise ValueError(
            f"{operation} weight must have shape ({x.shape[-1]},), got {weight.shape}"
        )
    if bias is not None and bias.shape != weight.shape:
        raise ValueError(
            f"{operation} bias must have shape {weight.shape}, got {bias.shape}"
        )


def _resolve_norm_implementation(implementation, arrays, operation):
    if any(array.dtype != arrays[0].dtype for array in arrays[1:]):
        if implementation == "triton":
            raise RuntimeError(
                f"cannot use implementation='triton' for {operation}: input "
                "and affine parameters must share dtype"
            )
        return "eager"
    return resolve_implementation(
        implementation,
        arrays,
        operation=operation,
        last_dim_reduction=True,
    )
