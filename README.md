# KernelLeaf

KernelLeaf is a compact educational deep-learning framework built around an
explicit `TensorOp.compute` / `TensorOp.gradient` computation graph. NumPy is
the baseline CPU backend; CuPy provides an optional, lazily loaded CUDA array
backend. PyTorch is not a runtime dependency.

## Install

CPU development environment:

```bash
pip install -e ".[dev]"
```

Optional NVIDIA CUDA 12 support:

```bash
pip install -e ".[dev,cuda]"
```

Install the optional Triton fused kernels with:

```bash
pip install -e ".[dev,cuda,triton]"
```

The CUDA extra installs CuPy plus CUDA runtime/NVRTC component wheels; an
NVIDIA driver is still required. Apple Silicon cannot run the CUDA/CuPy path;
use the NumPy CPU backend there.

## Quick start

```python
import numpy as np
import kernelleaf as kl

x = kl.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
loss = (x * x).sum()
loss.backward()

print(loss.numpy())   # 14.0
print(x.grad.numpy()) # [2. 4. 6.]
```

Device transfers are explicit:

```python
x_cpu = kl.Tensor([1, 2, 3], dtype="float32")
if kl.is_cuda_available():
    x_gpu = x_cpu.cuda(0)
    x_cpu_again = x_gpu.cpu()
```

Modules recursively move parameters, running-statistic buffers, and existing
gradients. State loading preserves the destination device and dtype:

```python
model = kl.nn.Linear(3, 2).to(kl.cuda(0))
state = model.state_dict()
model.load_state_dict(state)
```

## Examples

The MNIST entry points do not download data automatically:

```bash
python -m apps.mlp_mnist --help
python -m apps.lenet5_mnist --help
python -m examples.device_smoke
```

Conv2d supports `implementation="naive"`, `"im2col"`, `"triton"`, or `"auto"`.
On supported CUDA inputs, the default `auto` path selects the V5 Triton
implicit-GEMM forward kernel; its backward pass explicitly uses the optimized
im2col/CuPy-kernel path. CPU and unsupported CUDA inputs fall back to bounded
im2col, then naive:

```python
conv = kl.nn.Conv2d(3, 16, 3, padding=1, implementation="auto")
```

| Conv2d path | CPU forward | CUDA forward | backward |
| --- | --- | --- | --- |
| `naive` | yes | yes | naive |
| `im2col` | yes | yes | im2col; CUDA col2im uses a CuPy kernel |
| `triton` | no | yes, supported shapes/dtypes only | im2col/CuPy fallback (not Triton) |
| `auto` | im2col → naive | Triton → im2col → naive | follows the selected forward path above |

Run the V3 benchmark without writing result files:

```bash
python -m benchmarks.bench_conv --device cpu
python -m benchmarks.bench_conv --device cuda
python -m benchmarks.bench_conv --device cuda --suite all --implementation all
```

See `docs/conv2d.md` for layout, output-shape, selection, and memory details.

V4 provides forward-fused Triton implementations of Linear, Softmax,
LayerNorm, and RMSNorm. Their backward passes intentionally remain eager.
See `docs/fused-ops.md` for dispatch constraints and benchmarking.

V5 extends the same CuPy-backed Triton runtime and explicit dispatch model to
Conv2d forward. See `docs/conv2d.md` for the forward/backward support matrix,
shape limits, fallbacks, correctness coverage, and benchmark protocol.

V6 adds deterministic bounded asynchronous data loading while retaining
`num_workers=0` as the synchronous baseline:

```python
loader = kl.data.DataLoader(
    dataset, batch_size=128, shuffle=True, seed=7,
    num_workers=2, prefetch_factor=2, drop_last=True,
)
```

It also includes a lightweight reader for the official CIFAR-10 Python files
and a synthetic throughput/TinyCNN benchmark. See `docs/dataloader.md` for
worker cleanup, exception propagation, pinned-memory transfer boundaries,
CIFAR-10 download instructions, and benchmark commands.

V7 adds framework `BatchNorm2d`/global adaptive average pooling, portable NPZ
model+optimizer checkpoints, and a new KernelLeaf dual-head lightweight ResNet in
`apps/autodrive`. Its manifest-only dataset uses per-map grouped splits
(frame-number blocks for legacy data, real runs for new collections).
See `apps/autodrive/README.md` for legacy DonkeyCar manifest
conversion, training, resume, and evaluation commands. The Paddle directory is
reference code only. The AutoDrive package now also loads paired JSON/NPZ
artifacts for safe, smoothed, dual-head closed-loop driving through an isolated
Gym DonkeyCar adapter. V8 adds keyboard steering/dynamic-throttle collection,
timestamped per-run records, image/label auditing, per-map grouped splitting,
and steering/throttle Grad-CAM using KernelLeaf autograd. Run
`python -m apps.autodrive --help` for the unified entry point and see the app
README for the complete workflow. No real-track result is claimed.

Expected files are under `data/MNIST/raw/`. Unit tests use synthetic inputs and
do not require the dataset.

## Repository layout

- `kernelleaf/`: canonical framework and NumPy/CuPy device API
- `apps/`: canonical runnable examples
- `examples/`: small backend and migration smoke programs
- `benchmarks/`: reproducible command-line microbenchmarks
- `tests/`: canonical CPU/CUDA tests
- `docs/`: architecture and migration notes

The framework is adapted from the MIT-licensed Needle educational project and
the earlier `GuoGuo614/MyTorch-1` interface design. See `LICENSE` and the
architecture boundary document for scope details.
