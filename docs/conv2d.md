# Conv2d implementations

KernelLeaf Conv2d keeps one public layout on every backend:

- input: NCHW `(batch, in_channels, height, width)`;
- weight: OIHW `(out_channels, in_channels, kernel_height, kernel_width)`;
- bias in `nn.Conv2d`: `(1, out_channels, 1, 1)`;
- output: NCHW.

For each spatial dimension, after `nn.Conv2d` applies symmetric padding, the
output size is `floor((input - kernel) / stride) + 1`. Integer and two-element
kernel, stride, and padding values are supported. Dilation and grouped
convolution are intentionally outside the current API.

## Selection

`implementation="naive"` preserves the Python-loop correctness baseline.
`implementation="im2col"` gathers convolution windows into row chunks and
uses the active backend's matrix multiplication. `implementation="triton"`
uses an implicit-GEMM CUDA forward kernel: input-window indices are generated
inside the kernel, so it does not allocate an im2col matrix. `auto` selects
Triton only when its complete support contract is met, then im2col, then naive.

| implementation | CPU forward/backward | CUDA forward | CUDA backward |
| --- | --- | --- | --- |
| `naive` | naive / naive | naive | naive |
| `im2col` | chunked im2col / im2col | chunked im2col | im2col GEMMs + CuPy col2im kernel |
| `triton` | unsupported | Triton implicit-GEMM | optimized im2col/CuPy fallback |
| `auto` | im2col or naive | Triton when supported | selected path's documented backward |

This is therefore a **Triton forward path**, not yet a full Triton Conv2d.
`Conv2d.selected_implementation`/`nn.Conv2d.last_implementation` record the
forward path. `Conv2d.backward_implementation` and
`nn.Conv2d.last_backward_implementation` expose the backward plan/result.

The Triton path requires CUDA capability 7.0+, Triton, contiguous NCHW/OIHW
CuPy arrays on one device, matching float16/float32 dtypes, kernel dimensions
up to 7, stride dimensions up to 4, and a reduction dimension
`in_channels * kernel_height * kernel_width <= 4096`. Forced Triton reports a
clear error for unsupported inputs; `auto` falls back before execution. Kernel
execution errors and incorrect results are never silently retried on another
path.

`nn.Conv2d` materializes symmetric padding before dispatch, so the Triton
kernel receives a contiguous padded tensor. Integer or two-dimensional padding
continues to work at module level, including boundary tiles and non-square
kernels/strides. Dilation and groups remain outside the public API.

## Temporary memory

The complete im2col matrix is never materialized. Forward and backward split
the `batch * output_height * output_width` rows according to
`max_im2col_bytes`, which defaults to 64 MiB. The limit covers the column and
column-gradient workspaces; output and final gradient arrays are necessarily
allocated at their full tensor sizes. Explicit im2col raises `MemoryError` if
even one work row exceeds the limit, while auto falls back to naive.

## Benchmark

Run `python -m benchmarks.bench_conv --device cpu` or use `--device cuda`.
On CUDA, `--implementation all --suite all` compares naive, im2col, and Triton
over reproducible small/medium/large cases; `--suite single --shape N C H W`
benchmarks a custom case. Every implementation gets the same warmup count and
explicit device synchronization.
Each JSON line reports input/weight/output shapes, requested and selected
implementation, device, dtype, warmup/repeat counts, median, P95, peak-memory
estimate, and im2col chunk rows. CPU memory uses `tracemalloc`; CUDA memory is
the retained high-water delta of CuPy's default memory pool.
