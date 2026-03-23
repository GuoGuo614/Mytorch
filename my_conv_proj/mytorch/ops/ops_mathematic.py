"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return array_api.power(a, b)
        
    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lhs_grad = rhs * power(lhs, rhs - 1)
        rhs_grad = power(lhs, rhs) * log(lhs)
        return out_grad * lhs_grad, out_grad * rhs_grad

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0]
        if self.scalar == 0:
            # x^0 的导数是 0
            return mul_scalar(input_tensor, 0) * out_grad
        else:
            # 一般情况：n * x^(n-1)
            grad_coeff = mul_scalar(power_scalar(input_tensor, self.scalar - 1), self.scalar)
            return out_grad * grad_coeff


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * lhs * (-1) / (rhs * rhs)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            # 默认转置最后两个轴
            return array_api.swapaxes(a, -1, -2)
        else:
            # 指定轴转置
            axis1, axis2 = self.axes
            return array_api.swapaxes(a, axis1, axis2)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        result = array_api.reshape(a, self.shape)
        return result.astype(a.dtype) 

    def gradient(self, out_grad, node):
        # reshape 的逆操作就是 reshape 回原来的形状
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        reuslt = array_api.broadcast_to(a, shape=self.shape)
        return reuslt.astype(a.dtype)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        
        # 首先处理维度不匹配的情况
        grad = out_grad
        # 如果输入维度较少，先求和掉前面多出的维度
        ndims_added = len(self.shape) - len(input_shape)
        for i in range(ndims_added):
            grad = summation(grad, axes=(0,))
        
        # 然后处理size为1被广播的维度
        for i, (input_dim, output_dim) in enumerate(zip(input_shape, grad.shape)):
            if input_dim == 1 and output_dim > 1:
                grad = summation(grad, axes=(i,))
                grad = reshape(grad, grad.shape[:i] + (1,) + grad.shape[i:])
        
        return grad


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        result = array_api.sum(a, axis=self.axes)
        return result.astype(a.dtype)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        grad = out_grad
        
        # 如果指定了轴，需要在那些轴上添加维度
        if self.axes is not None:
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            # 对每个被求和的轴，添加维度1
            for axis in sorted(axes):
                grad = reshape(grad, grad.shape[:axis] + (1,) + grad.shape[axis:])

        return broadcast_to(grad, input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        result = array_api.matmul(a, b)
        target_dtype = a.dtype if a.dtype == b.dtype else array_api.result_type(a.dtype, b.dtype)
        return result.astype(target_dtype)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        
        # ∂L/∂A = ∂L/∂C @ B^T
        grad_lhs = matmul(out_grad, transpose(rhs))
        # ∂L/∂B = A^T @ ∂L/∂C  
        grad_rhs = matmul(transpose(lhs), out_grad)
        
        # 如果输入张量的维度少于输出梯度，需要求和掉多出的批量维度
        if len(grad_lhs.shape) > len(lhs.shape):
            # 计算需要求和的轴数
            axes_to_sum = tuple(range(len(grad_lhs.shape) - len(lhs.shape)))
            grad_lhs = summation(grad_lhs, axes=axes_to_sum)
            
        if len(grad_rhs.shape) > len(rhs.shape):
            # 计算需要求和的轴数  
            axes_to_sum = tuple(range(len(grad_rhs.shape) - len(rhs.shape)))
            grad_rhs = summation(grad_rhs, axes=axes_to_sum)
            
        return grad_lhs, grad_rhs


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        input = node.inputs[0]
        return divide(out_grad, input)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        input = node.inputs[0]
        return out_grad * exp(input)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(0, a)

    def gradient(self, out_grad, node):
        input_data = node.inputs[0].realize_cached_data()
        mask = (input_data > 0).astype(array_api.float32)
        
        return out_grad * Tensor(mask, device=out_grad.device)


def relu(a):
    return ReLU()(a)
