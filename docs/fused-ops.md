# V4 fused operators

KernelLeaf V4 adds optional Triton forward kernels for `Linear`, `Softmax`,
`LayerNorm`, and `RMSNorm`. Their public APIs accept
`implementation="auto" | "eager" | "triton"`.

- `auto` selects Triton only for supported contiguous CUDA inputs; otherwise it
  uses the existing eager implementation.
- `eager` always uses CuPy or NumPy array operations.
- `triton` requires a supported CUDA input and raises a descriptive error
  instead of silently falling back.

All four V4 kernels fuse **forward only**. Backward remains the transparent
TensorOp formula; Linear backward uses eager GEMMs. Benchmarks therefore label
backward as eager and must not be interpreted as fully fused training results.

The Triton path supports contiguous float16/float32 arrays on CUDA capability
7.0 or newer. Softmax and both norms operate on the last axis and limit that
axis to 65,536 elements. Linear's Triton path accepts 2-D inputs and matching
input/weight/bias dtypes. Unsupported axes, layouts, sizes, shapes, dtypes, or
devices fall back only in `auto` mode.

CuPy allocation pointers and its current CUDA stream are passed directly to
Triton. The runtime registers a CuPy-backed Triton driver, so neither PyTorch
nor DLPack is involved.

Install and benchmark on NVIDIA CUDA:

```bash
pip install -e ".[dev,cuda,triton]"
python -m benchmarks.bench_fused_ops --operator all --implementation all
```

The kernels were adapted to the current TensorOp architecture from the
MIT-licensed [GuoGuo614/MyTorch-1](https://github.com/GuoGuo614/MyTorch-1)
fused-operator implementations. The driver integration follows the public
[Triton Windows](https://github.com/triton-lang/triton-windows) distribution
without copying its runtime implementation.
