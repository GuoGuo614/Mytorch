"""Triton LayerNorm forward kernel using direct CuPy pointers.

Adapted from the MIT-licensed GuoGuo614/MyTorch-1 implementation.
"""

import cupy as cp
import triton
import triton.language as tl


@triton.jit
def _layernorm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    columns: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
):
    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    input_pointer = tl.cast(input_pointer, tl.pointer_type(pointer_dtype))
    weight_pointer = tl.cast(weight_pointer, tl.pointer_type(pointer_dtype))
    bias_pointer = tl.cast(bias_pointer, tl.pointer_type(pointer_dtype))
    output_pointer = tl.cast(output_pointer, tl.pointer_type(pointer_dtype))
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < columns
    values = tl.load(input_pointer + row * columns + offsets, mask=mask, other=0.0)
    values_fp32 = values.to(tl.float32)
    mean = tl.sum(values_fp32, axis=0) / columns
    centered = tl.where(mask, values_fp32 - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / columns
    normalized = centered * tl.rsqrt(variance + epsilon)
    weight = tl.load(weight_pointer + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_pointer + offsets, mask=mask, other=0.0)
    result = normalized * weight + bias
    tl.store(output_pointer + row * columns + offsets, result, mask=mask)


def forward(x, weight, bias, epsilon):
    columns = x.shape[-1]
    rows = x.size // columns
    output = cp.empty_like(x)
    block_size = triton.next_power_of_2(columns)
    num_warps = 4 if block_size < 2048 else 8
    _layernorm_forward_kernel[(rows,)](
        x.data.ptr,
        weight.data.ptr,
        bias.data.ptr,
        output.data.ptr,
        columns,
        epsilon,
        BLOCK_SIZE=block_size,
        DTYPE_FLAG=0 if x.dtype == cp.float32 else 1,
        num_warps=num_warps,
    )
    return output
