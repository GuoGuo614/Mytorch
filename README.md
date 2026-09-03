# MyTorch

MyTorch is a compact educational deep-learning framework built around an
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

The CUDA extra installs CuPy plus CUDA runtime/NVRTC component wheels; an
NVIDIA driver is still required. Apple Silicon cannot run the CUDA/CuPy path;
use the NumPy CPU backend there.

## Quick start

```python
import numpy as np
import mytorch as mt

x = mt.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
loss = (x * x).sum()
loss.backward()

print(loss.numpy())   # 14.0
print(x.grad.numpy()) # [2. 4. 6.]
```

Device transfers are explicit:

```python
x_cpu = mt.Tensor([1, 2, 3], dtype="float32")
if mt.is_cuda_available():
    x_gpu = x_cpu.cuda(0)
    x_cpu_again = x_gpu.cpu()
```

Modules recursively move parameters, running-statistic buffers, and existing
gradients. State loading preserves the destination device and dtype:

```python
model = mt.nn.Linear(3, 2).to(mt.cuda(0))
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

Conv2d supports `implementation="naive"`, `"im2col"`, or `"auto"`. The
default `auto` path currently selects bounded-memory im2col when supported:

```python
conv = mt.nn.Conv2d(3, 16, 3, padding=1, implementation="auto")
```

Run the V3 benchmark without writing result files:

```bash
python -m benchmarks.bench_conv --device cpu
python -m benchmarks.bench_conv --device cuda
```

See `docs/conv2d.md` for layout, output-shape, selection, and memory details.

Expected files are under `data/MNIST/raw/`. Unit tests use synthetic inputs and
do not require the dataset.

## Repository layout

- `mytorch/`: canonical framework and NumPy/CuPy device API
- `apps/`: canonical runnable examples
- `examples/`: small backend and migration smoke programs
- `benchmarks/`: reproducible command-line microbenchmarks
- `tests/`: canonical CPU/CUDA tests
- `docs/`: architecture and migration notes
- `MyTorch_分阶段迁移_Codex任务书.md`: staged V0-V12 migration plan

The framework is adapted from the MIT-licensed Needle educational project and
the earlier `GuoGuo614/MyTorch-1` interface design. See `LICENSE` and the
architecture boundary document for scope details.
