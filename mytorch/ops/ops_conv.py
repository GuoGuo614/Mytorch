"""Convolution and pooling operations."""

from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..backend import get_array_module


class Conv2d(TensorOp):
    def __init__(self, stride: int = 1):
        self.stride = stride

    def compute(self, x: NDArray, weight: NDArray) -> NDArray:
        """
        2D卷积的前向传播（优化版本）
        x: (batch_size, in_channels, height, width)
        weight: (out_channels, in_channels, kernel_h, kernel_w)
        output: (batch_size, out_channels, out_h, out_w)
        """
        xp = get_array_module(x)
        batch_size, in_channels, H, W = x.shape
        out_channels, in_channels_w, kernel_h, kernel_w = weight.shape

        assert in_channels == in_channels_w, "Input channels mismatch"

        # 计算输出尺寸
        out_h = (H - kernel_h) // self.stride + 1
        out_w = (W - kernel_w) // self.stride + 1

        # 优化：减少循环，使用向量化操作
        output = xp.zeros((batch_size, out_channels, out_h, out_w), dtype=x.dtype)

        # 只对输出位置和输出通道循环
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + kernel_h
                w_end = w_start + kernel_w

                # 提取所有batch的窗口: (batch_size, in_channels, kernel_h, kernel_w)
                window = x[:, :, h_start:h_end, w_start:w_end]

                # 对每个输出通道计算
                for oc in range(out_channels):
                    # window: (batch_size, in_channels, kernel_h, kernel_w)
                    # weight[oc]: (in_channels, kernel_h, kernel_w)
                    # 广播相乘后求和: (batch_size, in_channels, kernel_h, kernel_w) -> (batch_size,)
                    output[:, oc, i, j] = xp.sum(window * weight[oc], axis=(1, 2, 3))

        return output

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        卷积的反向传播
        out_grad: (batch_size, out_channels, out_h, out_w)
        """
        x, weight = node.inputs
        x_data = x.realize_cached_data()
        weight_data = weight.realize_cached_data()

        batch_size, in_channels, H, W = x_data.shape
        out_channels, _, kernel_h, kernel_w = weight_data.shape
        _, _, out_h, out_w = out_grad.shape

        # 计算x的梯度
        xp = get_array_module(x_data)
        grad_x = xp.zeros_like(x_data)
        out_grad_data = out_grad.realize_cached_data()

        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + kernel_h
                        w_end = w_start + kernel_w

                        grad_x[b, :, h_start:h_end, w_start:w_end] += \
                            weight_data[oc] * out_grad_data[b, oc, i, j]

        # 计算weight的梯度
        grad_weight = xp.zeros_like(weight_data)

        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + kernel_h
                        w_end = w_start + kernel_w

                        window = x_data[b, :, h_start:h_end, w_start:w_end]
                        grad_weight[oc] += window * out_grad_data[b, oc, i, j]

        return Tensor(grad_x, device=x.device), Tensor(grad_weight, device=weight.device)


def conv2d(x, weight, stride=1):
    return Conv2d(stride)(x, weight)


class MaxPool2d(TensorOp):
    def __init__(self, kernel_size: int, stride: int):
        self.kernel_size = kernel_size
        self.stride = stride

    def compute(self, x: NDArray) -> NDArray:
        """
        2D最大池化的前向传播（优化版本）
        x: (batch_size, channels, height, width)
        output: (batch_size, channels, out_h, out_w)
        """
        xp = get_array_module(x)
        batch_size, channels, H, W = x.shape

        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        output = xp.zeros((batch_size, channels, out_h, out_w), dtype=x.dtype)

        # 优化：减少循环
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size

                # 提取所有batch和channel的窗口
                window = x[:, :, h_start:h_end, w_start:w_end]
                # (batch_size, channels, kernel_size, kernel_size) -> (batch_size, channels)
                output[:, :, i, j] = xp.max(window, axis=(2, 3))

        return output

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        最大池化的反向传播
        """
        x = node.inputs[0]
        x_data = x.realize_cached_data()
        out_grad_data = out_grad.realize_cached_data()

        batch_size, channels, H, W = x_data.shape
        _, _, out_h, out_w = out_grad_data.shape

        xp = get_array_module(x_data)
        grad_x = xp.zeros_like(x_data)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        window = x_data[b, c, h_start:h_end, w_start:w_end]
                        max_val = xp.max(window)

                        # 找到最大值的位置并传递梯度
                        mask = (window == max_val).astype(x_data.dtype)
                        grad_x[b, c, h_start:h_end, w_start:w_end] += \
                            mask * out_grad_data[b, c, i, j]

        return Tensor(grad_x, device=x.device)


def max_pool2d(x, kernel_size, stride):
    return MaxPool2d(kernel_size, stride)(x)


class AvgPool2d(TensorOp):
    def __init__(self, kernel_size: int, stride: int):
        self.kernel_size = kernel_size
        self.stride = stride

    def compute(self, x: NDArray) -> NDArray:
        """
        2D平均池化的前向传播（优化版本）
        x: (batch_size, channels, height, width)
        output: (batch_size, channels, out_h, out_w)
        """
        xp = get_array_module(x)
        batch_size, channels, H, W = x.shape

        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        output = xp.zeros((batch_size, channels, out_h, out_w), dtype=x.dtype)

        # 优化：减少循环
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size

                # 提取所有batch和channel的窗口
                window = x[:, :, h_start:h_end, w_start:w_end]
                # (batch_size, channels, kernel_size, kernel_size) -> (batch_size, channels)
                output[:, :, i, j] = xp.mean(window, axis=(2, 3))

        return output

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        平均池化的反向传播
        """
        x = node.inputs[0]
        x_data = x.realize_cached_data()
        out_grad_data = out_grad.realize_cached_data()

        batch_size, channels, H, W = x_data.shape
        _, _, out_h, out_w = out_grad_data.shape

        xp = get_array_module(x_data)
        grad_x = xp.zeros_like(x_data)
        pool_size = self.kernel_size * self.kernel_size

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        # 平均池化的梯度均匀分配到每个位置
                        grad_x[b, c, h_start:h_end, w_start:w_end] += \
                            out_grad_data[b, c, i, j] / pool_size

        return Tensor(grad_x, device=x.device)


def avg_pool2d(x, kernel_size, stride):
    return AvgPool2d(kernel_size, stride)(x)


class Pad(TensorOp):
    def __init__(self, pad_width):
        """
        pad_width: tuple of tuples, e.g., ((0,0), (0,0), (1,1), (1,1))
        """
        self.pad_width = pad_width

    def compute(self, x: NDArray) -> NDArray:
        """
        填充操作
        """
        xp = get_array_module(x)
        return xp.pad(x, self.pad_width, mode='constant', constant_values=0)

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        填充的反向传播：去掉填充的部分
        """
        out_grad_data = out_grad.realize_cached_data()

        # 构造slice来去掉填充
        slices = []
        for (pad_before, pad_after) in self.pad_width:
            if pad_after == 0:
                slices.append(slice(pad_before, None))
            else:
                slices.append(slice(pad_before, -pad_after))

        grad_x = out_grad_data[tuple(slices)]
        return Tensor(grad_x, device=out_grad.device)


def pad(x, pad_width):
    return Pad(pad_width)(x)


class Tanh(TensorOp):
    def compute(self, x: NDArray) -> NDArray:
        return get_array_module(x).tanh(x)

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        tanh的导数: 1 - tanh^2(x)
        """
        output = node.realize_cached_data()
        grad = 1 - output * output
        return out_grad * Tensor(grad, device=out_grad.device)


def tanh(x):
    return Tanh()(x)


class Sigmoid(TensorOp):
    def compute(self, x: NDArray) -> NDArray:
        return 1 / (1 + get_array_module(x).exp(-x))

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        sigmoid的导数: sigmoid(x) * (1 - sigmoid(x))
        """
        output = node.realize_cached_data()
        grad = output * (1 - output)
        return out_grad * Tensor(grad, device=out_grad.device)


def sigmoid(x):
    return Sigmoid()(x)


class Argmax(TensorOp):
    def __init__(self, axis: Optional[int] = None):
        self.axis = axis

    def compute(self, x: NDArray) -> NDArray:
        return get_array_module(x).argmax(x, axis=self.axis)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # argmax不可微，返回零梯度
        x = node.inputs[0]
        data = x.realize_cached_data()
        return Tensor(get_array_module(data).zeros_like(data), device=x.device)


def argmax(x, axis=None):
    return Argmax(axis)(x)
