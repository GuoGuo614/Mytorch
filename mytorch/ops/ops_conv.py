"""Convolution and pooling operations.

Conv2d uses NCHW inputs and OIHW weights.  The im2col implementation builds
bounded row chunks rather than materializing the complete column matrix.
"""

from typing import Optional
from ..autograd import NDArray
from ..autograd import Tensor, TensorOp
from ..backend import get_array_module


_CONV_IMPLEMENTATIONS = {"auto", "naive", "im2col"}
DEFAULT_IM2COL_MAX_BYTES = 64 * 1024 * 1024


def _pair(value, name):
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise TypeError(f"{name} must be an int or a pair of ints")
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


def _conv_shape(x, weight, stride):
    if x.ndim != 4:
        raise ValueError(f"Conv2d expects NCHW input with 4 dimensions, got {x.shape}")
    if weight.ndim != 4:
        raise ValueError(
            f"Conv2d expects OIHW weight with 4 dimensions, got {weight.shape}"
        )
    batch, in_channels, height, width = x.shape
    out_channels, weight_channels, kernel_h, kernel_w = weight.shape
    if kernel_h <= 0 or kernel_w <= 0:
        raise ValueError(f"Conv2d kernel dimensions must be positive, got {weight.shape}")
    if in_channels != weight_channels:
        raise ValueError(
            f"Conv2d channel mismatch: input has {in_channels}, "
            f"weight expects {weight_channels}"
        )
    if x.dtype != weight.dtype:
        raise TypeError(
            f"Conv2d requires matching dtypes, got {x.dtype} and {weight.dtype}"
        )
    stride_h, stride_w = _pair(stride, "stride")
    out_h = (height - kernel_h) // stride_h + 1
    out_w = (width - kernel_w) // stride_w + 1
    if batch <= 0 or out_channels <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError(
            "Conv2d kernel must fit a non-empty input; "
            f"input={x.shape}, weight={weight.shape}, stride={(stride_h, stride_w)}"
        )
    return (
        batch, in_channels, height, width, out_channels,
        kernel_h, kernel_w, out_h, out_w, stride_h, stride_w,
    )


def _im2col_row_bytes(x, weight):
    kernel_elements = weight.shape[1] * weight.shape[2] * weight.shape[3]
    return x.dtype.itemsize * (2 * kernel_elements + weight.shape[0])


def _im2col_supported(x, weight, max_im2col_bytes):
    return (
        x.dtype.kind == "f"
        and weight.dtype.kind == "f"
        and _im2col_row_bytes(x, weight) <= max_im2col_bytes
    )


def select_conv2d_implementation(x, weight, implementation="auto",
                                 max_im2col_bytes=DEFAULT_IM2COL_MAX_BYTES):
    """Resolve a Conv2d implementation for concrete backend arrays."""
    if implementation not in _CONV_IMPLEMENTATIONS:
        raise ValueError(
            f"implementation must be one of {sorted(_CONV_IMPLEMENTATIONS)}, "
            f"got {implementation!r}"
        )
    if max_im2col_bytes <= 0:
        raise ValueError("max_im2col_bytes must be positive")
    if implementation == "auto":
        return (
            "im2col" if _im2col_supported(x, weight, max_im2col_bytes)
            else "naive"
        )
    if implementation == "im2col" and (
        x.dtype.kind != "f" or weight.dtype.kind != "f"
    ):
        raise NotImplementedError(
            "Conv2d implementation='im2col' supports floating-point arrays only"
        )
    if implementation == "im2col" and (
        _im2col_row_bytes(x, weight) > max_im2col_bytes
    ):
        raise MemoryError(
            "one im2col work row exceeds max_im2col_bytes; increase the limit "
            "or use implementation='naive'/'auto'"
        )
    return implementation


class Conv2d(TensorOp):
    def __init__(self, stride=1, implementation="auto",
                 max_im2col_bytes=DEFAULT_IM2COL_MAX_BYTES):
        self.stride = stride
        self.implementation = implementation
        self.max_im2col_bytes = max_im2col_bytes
        self.selected_implementation = None
        self.chunk_rows = None

    def compute(self, x: NDArray, weight: NDArray) -> NDArray:
        shape = _conv_shape(x, weight, self.stride)
        self.selected_implementation = select_conv2d_implementation(
            x, weight, self.implementation, self.max_im2col_bytes
        )
        if self.selected_implementation == "naive":
            self.chunk_rows = None
            return self._forward_naive(x, weight, shape)
        return self._forward_im2col(x, weight, shape)

    def _forward_naive(self, x, weight, shape):
        xp = get_array_module(x)
        (batch, _, _, _, out_channels, kernel_h, kernel_w,
         out_h, out_w, stride_h, stride_w) = shape
        output = xp.zeros((batch, out_channels, out_h, out_w), dtype=x.dtype)
        for out_y in range(out_h):
            for out_x in range(out_w):
                window = x[
                    :, :,
                    out_y * stride_h:out_y * stride_h + kernel_h,
                    out_x * stride_w:out_x * stride_w + kernel_w,
                ]
                for out_channel in range(out_channels):
                    output[:, out_channel, out_y, out_x] = xp.sum(
                        window * weight[out_channel], axis=(1, 2, 3)
                    )
        return output

    def _rows_per_chunk(self, x, weight, total_rows):
        bytes_per_row = _im2col_row_bytes(x, weight)
        return max(1, min(total_rows, self.max_im2col_bytes // bytes_per_row))

    @staticmethod
    def _column_chunk(x, start, end, shape):
        xp = get_array_module(x)
        (_, in_channels, _, _, _, kernel_h, kernel_w,
         out_h, out_w, stride_h, stride_w) = shape
        positions = xp.arange(start, end, dtype=xp.int64)
        batch_index = positions // (out_h * out_w)
        spatial = positions % (out_h * out_w)
        base_h = (spatial // out_w) * stride_h
        base_w = (spatial % out_w) * stride_w
        kernel_index = xp.arange(
            in_channels * kernel_h * kernel_w, dtype=xp.int64
        )
        channel = kernel_index // (kernel_h * kernel_w)
        kernel_spatial = kernel_index % (kernel_h * kernel_w)
        offset_h = kernel_spatial // kernel_w
        offset_w = kernel_spatial % kernel_w
        columns = x[
            batch_index[:, None],
            channel[None, :],
            base_h[:, None] + offset_h[None, :],
            base_w[:, None] + offset_w[None, :],
        ]
        return columns, batch_index, base_h, base_w

    def _forward_im2col(self, x, weight, shape):
        xp = get_array_module(x)
        (batch, _, _, _, out_channels, _, _, out_h, out_w, _, _) = shape
        total_rows = batch * out_h * out_w
        self.chunk_rows = self._rows_per_chunk(x, weight, total_rows)
        output_rows = xp.empty((total_rows, out_channels), dtype=x.dtype)
        weight_rows = weight.reshape(out_channels, -1)
        for start in range(0, total_rows, self.chunk_rows):
            end = min(start + self.chunk_rows, total_rows)
            columns, _, _, _ = self._column_chunk(x, start, end, shape)
            output_rows[start:end] = columns @ weight_rows.T
        return output_rows.reshape(batch, out_h, out_w, out_channels).transpose(
            0, 3, 1, 2
        )

    def gradient(self, out_grad: Tensor, node: Tensor):
        x, weight = node.inputs
        x_data = x.realize_cached_data()
        weight_data = weight.realize_cached_data()
        out_grad_data = out_grad.realize_cached_data()
        shape = _conv_shape(x_data, weight_data, self.stride)
        implementation = self.selected_implementation or select_conv2d_implementation(
            x_data, weight_data, self.implementation, self.max_im2col_bytes
        )
        if implementation == "naive":
            grad_x, grad_weight = self._backward_naive(
                x_data, weight_data, out_grad_data, shape
            )
        else:
            grad_x, grad_weight = self._backward_im2col(
                x_data, weight_data, out_grad_data, shape
            )
        return Tensor(grad_x, device=x.device), Tensor(grad_weight, device=weight.device)


    def _backward_naive(self, x, weight, out_grad, shape):
        xp = get_array_module(x)
        (batch, _, _, _, out_channels, kernel_h, kernel_w,
         out_h, out_w, stride_h, stride_w) = shape
        grad_x = xp.zeros_like(x)
        grad_weight = xp.zeros_like(weight)
        for batch_index in range(batch):
            for out_channel in range(out_channels):
                for out_y in range(out_h):
                    for out_x in range(out_w):
                        h_start = out_y * stride_h
                        w_start = out_x * stride_w
                        upstream = out_grad[batch_index, out_channel, out_y, out_x]
                        grad_x[
                            batch_index, :,
                            h_start:h_start + kernel_h,
                            w_start:w_start + kernel_w,
                        ] += weight[out_channel] * upstream
                        grad_weight[out_channel] += x[
                            batch_index, :,
                            h_start:h_start + kernel_h,
                            w_start:w_start + kernel_w,
                        ] * upstream
        return grad_x, grad_weight

    def _backward_im2col(self, x, weight, out_grad, shape):
        xp = get_array_module(x)
        (batch, in_channels, _, _, out_channels, kernel_h, kernel_w,
         out_h, out_w, _, _) = shape
        total_rows = batch * out_h * out_w
        chunk_rows = self._rows_per_chunk(x, weight, total_rows)
        self.chunk_rows = chunk_rows
        weight_rows = weight.reshape(out_channels, -1)
        out_grad_rows = out_grad.transpose(0, 2, 3, 1).reshape(
            total_rows, out_channels
        )
        grad_x = xp.zeros_like(x)
        grad_weight_rows = xp.zeros_like(weight_rows)
        kernel_elements = in_channels * kernel_h * kernel_w
        for start in range(0, total_rows, chunk_rows):
            end = min(start + chunk_rows, total_rows)
            columns, batch_index, base_h, base_w = self._column_chunk(
                x, start, end, shape
            )
            upstream = out_grad_rows[start:end]
            grad_weight_rows += upstream.T @ columns
            del columns
            grad_columns = upstream @ weight_rows
            for kernel_index in range(kernel_elements):
                channel = kernel_index // (kernel_h * kernel_w)
                spatial = kernel_index % (kernel_h * kernel_w)
                offset_h = spatial // kernel_w
                offset_w = spatial % kernel_w
                xp.add.at(
                    grad_x,
                    (
                        batch_index,
                        channel,
                        base_h + offset_h,
                        base_w + offset_w,
                    ),
                    grad_columns[:, kernel_index],
                )
        return grad_x, grad_weight_rows.reshape(weight.shape)


def conv2d(x, weight, stride=1, implementation="auto",
           max_im2col_bytes=DEFAULT_IM2COL_MAX_BYTES):
    return Conv2d(stride, implementation, max_im2col_bytes)(x, weight)


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
