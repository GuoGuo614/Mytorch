"""Triton last-axis softmax forward using direct CuPy pointers.

Adapted from the MIT-licensed GuoGuo614/MyTorch-1 implementation.
"""

import cupy as cp
import triton
import triton.language as tl


@triton.jit
def _softmax_forward_kernel(
    input_pointer,
    output_pointer,
    columns: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
):
    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    input_pointer = tl.cast(input_pointer, tl.pointer_type(pointer_dtype))
    output_pointer = tl.cast(output_pointer, tl.pointer_type(pointer_dtype))
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < columns
    values = tl.load(input_pointer + row * columns + offsets, mask=mask, other=-float("inf"))
    numerator = tl.exp(values - tl.max(values, axis=0))
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(output_pointer + row * columns + offsets, result, mask=mask)


def forward(x):
    columns = x.shape[-1]
    rows = x.size // columns
    output = cp.empty_like(x)
    block_size = triton.next_power_of_2(columns)
    num_warps = 4 if block_size < 2048 else 8
    _softmax_forward_kernel[(rows,)](
        x.data.ptr,
        output.data.ptr,
        columns,
        BLOCK_SIZE=block_size,
        DTYPE_FLAG=0 if x.dtype == cp.float32 else 1,
        num_warps=num_warps,
    )
    return output
