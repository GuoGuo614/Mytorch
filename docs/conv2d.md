# Conv2d implementations

MyTorch Conv2d keeps one public layout on every backend:

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
uses the active backend's matrix multiplication. `implementation="auto"`
selects im2col for supported floating-point shapes and otherwise falls back to
naive. `Conv2d.selected_implementation` and `nn.Conv2d.last_implementation`
record the resolved path for tests and diagnostics.

## Temporary memory

The complete im2col matrix is never materialized. Forward and backward split
the `batch * output_height * output_width` rows according to
`max_im2col_bytes`, which defaults to 64 MiB. The limit covers the column and
column-gradient workspaces; output and final gradient arrays are necessarily
allocated at their full tensor sizes. Explicit im2col raises `MemoryError` if
even one work row exceeds the limit, while auto falls back to naive.

## Benchmark

Run `python -m benchmarks.bench_conv --device cpu` or use `--device cuda`.
Each JSON line reports input/weight/output shapes, requested and selected
implementation, device, dtype, warmup/repeat counts, median, P95, peak-memory
estimate, and im2col chunk rows. CPU memory uses `tracemalloc`; CUDA memory is
the retained high-water delta of CuPy's default memory pool.
