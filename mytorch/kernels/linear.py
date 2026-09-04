"""Triton fused Linear+bias forward kernel.

Adapted to direct CuPy pointers from GuoGuo614/MyTorch-1.  The backward pass is
intentionally handled by eager GEMMs in the TensorOp adapter.
"""

import cupy as cp
import triton
import triton.language as tl


@triton.jit
def _linear_forward_kernel(
    x_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    x_pointer = tl.cast(x_pointer, tl.pointer_type(pointer_dtype))
    weight_pointer = tl.cast(weight_pointer, tl.pointer_type(pointer_dtype))
    output_pointer = tl.cast(output_pointer, tl.pointer_type(pointer_dtype))
    if HAS_BIAS:
        bias_pointer = tl.cast(bias_pointer, tl.pointer_type(pointer_dtype))

    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    column_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block_start in range(0, tl.cdiv(in_features, BLOCK_K)):
        k_offsets = block_start * BLOCK_K + reduction_offsets
        x = tl.load(
            x_pointer + row_offsets[:, None] * in_features + k_offsets[None, :],
            mask=(row_offsets[:, None] < rows) & (k_offsets[None, :] < in_features),
            other=0.0,
        )
        weight = tl.load(
            weight_pointer
            + k_offsets[:, None] * out_features
            + column_offsets[None, :],
            mask=(k_offsets[:, None] < in_features)
            & (column_offsets[None, :] < out_features),
            other=0.0,
        )
        accumulator += tl.dot(x, weight, input_precision="ieee")
    if HAS_BIAS:
        bias = tl.load(
            bias_pointer + column_offsets,
            mask=column_offsets < out_features,
            other=0.0,
        )
        accumulator += bias[None, :]
    tl.store(
        output_pointer + row_offsets[:, None] * out_features + column_offsets[None, :],
        accumulator,
        mask=(row_offsets[:, None] < rows)
        & (column_offsets[None, :] < out_features),
    )


def forward(x, weight, bias=None):
    rows, in_features = x.shape
    _, out_features = weight.shape
    output = cp.empty((rows, out_features), dtype=x.dtype)
    grid = (triton.cdiv(rows, 32), triton.cdiv(out_features, 32))
    _linear_forward_kernel[grid](
        x.data.ptr,
        weight.data.ptr,
        0 if bias is None else bias.data.ptr,
        output.data.ptr,
        rows,
        out_features,
        in_features,
        HAS_BIAS=bias is not None,
        DTYPE_FLAG=0 if x.dtype == cp.float32 else 1,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return output
