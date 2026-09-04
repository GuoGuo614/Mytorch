"""Triton implicit-GEMM Conv2d forward kernel.

Adapted to MyTorch's NCHW/OIHW TensorOp API and direct CuPy pointers from
GuoGuo614/MyTorch-1.  Backward remains on the optimized im2col/CuPy path.
"""

import cupy as cp
import triton
import triton.language as tl


@triton.jit
def _conv2d_forward_kernel(
    x_pointer,
    weight_pointer,
    output_pointer,
    batch: tl.constexpr,
    in_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    rows: tl.constexpr,
    reduction_size: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    x_pointer = tl.cast(x_pointer, tl.pointer_type(pointer_dtype))
    weight_pointer = tl.cast(weight_pointer, tl.pointer_type(pointer_dtype))
    output_pointer = tl.cast(output_pointer, tl.pointer_type(pointer_dtype))

    program = tl.program_id(0)
    programs_m = tl.cdiv(rows, BLOCK_M)
    programs_n = tl.cdiv(out_channels, BLOCK_N)
    programs_per_group = GROUP_M * programs_n
    group = program // programs_per_group
    first_program_m = group * GROUP_M
    group_m = tl.minimum(programs_m - first_program_m, GROUP_M)
    program_m = first_program_m + (program % programs_per_group) % group_m
    program_n = (program % programs_per_group) // group_m

    row_offsets = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    channel_offsets = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    image = row_offsets // (out_h * out_w)
    spatial = row_offsets % (out_h * out_w)
    output_y = spatial // out_w
    output_x = spatial % out_w
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    reduction_offsets = tl.arange(0, BLOCK_K)
    for block_start in range(0, tl.cdiv(reduction_size, BLOCK_K)):
        k_offsets = block_start * BLOCK_K + reduction_offsets
        input_channel = k_offsets // (kernel_h * kernel_w)
        kernel_spatial = k_offsets % (kernel_h * kernel_w)
        kernel_y = kernel_spatial // kernel_w
        kernel_x = kernel_spatial % kernel_w
        input_y = output_y[:, None] * stride_h + kernel_y[None, :]
        input_x = output_x[:, None] * stride_w + kernel_x[None, :]
        x_offsets = (
            ((image[:, None] * in_channels + input_channel[None, :]) * height
             + input_y) * width + input_x
        )
        x = tl.load(
            x_pointer + x_offsets,
            mask=(row_offsets[:, None] < rows)
            & (k_offsets[None, :] < reduction_size),
            other=0.0,
        )
        weight_offsets = (
            ((channel_offsets[None, :] * in_channels
              + input_channel[:, None]) * kernel_h + kernel_y[:, None])
            * kernel_w + kernel_x[:, None]
        )
        weight = tl.load(
            weight_pointer + weight_offsets,
            mask=(k_offsets[:, None] < reduction_size)
            & (channel_offsets[None, :] < out_channels),
            other=0.0,
        )
        accumulator += tl.dot(x, weight, input_precision="ieee")

    output_offsets = (
        ((image[:, None] * out_channels + channel_offsets[None, :]) * out_h
         + output_y[:, None]) * out_w + output_x[:, None]
    )
    tl.store(
        output_pointer + output_offsets,
        accumulator,
        mask=(row_offsets[:, None] < rows)
        & (channel_offsets[None, :] < out_channels),
    )


def forward(x, weight, shape):
    (batch, in_channels, height, width, out_channels, kernel_h, kernel_w,
     out_h, out_w, stride_h, stride_w) = shape
    rows = batch * out_h * out_w
    reduction_size = in_channels * kernel_h * kernel_w
    output = cp.empty((batch, out_channels, out_h, out_w), dtype=x.dtype)
    block_m = 32
    block_n = 32
    block_k = 32
    grid = (
        triton.cdiv(rows, block_m) * triton.cdiv(out_channels, block_n),
    )
    _conv2d_forward_kernel[grid](
        x.data.ptr,
        weight.data.ptr,
        output.data.ptr,
        batch,
        in_channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        stride_h,
        stride_w,
        rows,
        reduction_size,
        DTYPE_FLAG=0 if x.dtype == cp.float32 else 1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=4,
    )
    return output
