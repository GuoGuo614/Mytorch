from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *
from ..backend import get_array_module

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        xp = get_array_module(Z)
        max_vals = xp.max(Z, axis=1, keepdims=True)
        shifted = Z - max_vals
        sum_exp = xp.sum(xp.exp(shifted), axis=1, keepdims=True)
        log_sum_exp = max_vals + xp.log(sum_exp)
        return Z - log_sum_exp

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_tensor = node.inputs[0]

        # 计算 softmax (2D, axis=1)
        lse_result = logsumexp(input_tensor, axes=(1,))
        lse_broadcasted = broadcast_to(reshape(lse_result, (-1, 1)), input_tensor.shape)
        softmax_vals = exp(input_tensor - lse_broadcasted)

        # LogSoftmax 梯度
        out_grad_sum = summation(out_grad, axes=(1,))
        out_grad_sum_broadcasted = broadcast_to(reshape(out_grad_sum, (-1, 1)), input_tensor.shape)

        return out_grad - softmax_vals * out_grad_sum_broadcasted


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        xp = get_array_module(Z)
        if self.axes is None:
            max_val = xp.max(Z)
            return xp.log(xp.sum(xp.exp(Z - max_val))) + max_val
        else:
            max_vals = xp.max(Z, axis=self.axes, keepdims=True)
            exp_shifted = xp.exp(Z - max_vals)
            sum_exp = xp.sum(exp_shifted, axis=self.axes, keepdims=True)
            result = max_vals + xp.log(sum_exp)

            # 张量在 index 上求和，就把 index 的维度去掉
            result_shape = list(Z.shape)
            if isinstance(self.axes, int):
                axes_tuple = (self.axes,)
            else:
                axes_tuple = self.axes

            # 逆序删除
            for axis in sorted(axes_tuple, reverse=True):
                result_shape.pop(axis)

            result = xp.reshape(result, result_shape)

            return result

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_tensor = node.inputs[0]
        lse_result = logsumexp(input_tensor, self.axes)

        if self.axes is not None:
            # 需要将 lse_result 广播到 input_tensor 的形状
            lse_shape = list(input_tensor.shape)
            axes_tuple = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for axis in axes_tuple:
                lse_shape[axis] = 1
            lse_broadcasted = reshape(lse_result, lse_shape)
            lse_broadcasted = broadcast_to(lse_broadcasted, input_tensor.shape)
        else:
            lse_broadcasted = lse_result

        softmax_vals = exp(input_tensor - lse_broadcasted)

        if self.axes is not None:
            out_grad_shape = list(input_tensor.shape)
            axes_tuple = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for axis in axes_tuple:
                out_grad_shape[axis] = 1
            out_grad_reshaped = reshape(out_grad, out_grad_shape)
            out_grad_final = broadcast_to(out_grad_reshaped, input_tensor.shape)
        else:
            out_grad_final = out_grad

        return softmax_vals * out_grad_final


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
