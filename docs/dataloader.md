# V6 asynchronous DataLoader and CIFAR-10

## Loader contract

`DataLoader(..., num_workers=0)` is the synchronous baseline. A positive
`num_workers` starts thread workers that fetch raw dataset batches through a
bounded producer/consumer pipeline. `prefetch_factor * num_workers` is both
the queue capacity and maximum number of submitted-but-not-yielded batches, so
out-of-order completion cannot create an unbounded reorder buffer.

Each iterator owns its workers. Normal epoch completion, worker exceptions,
conversion exceptions, `KeyboardInterrupt`, explicit `close()`, and iterator
destruction request shutdown and join the threads. If application code keeps a
reference to a partially consumed iterator, close it explicitly or use it as a
context manager:

```python
loader = DataLoader(dataset, batch_size=128, num_workers=2,
                    prefetch_factor=2, shuffle=True, seed=7)
with iter(loader) as batches:
    for inputs, labels in batches:
        if finished_early:
            break
```

Results are reordered by sampler sequence before they are yielded. The sampler
uses a private RNG derived from `(seed, epoch)`, so synchronous and asynchronous
loaders with the same configuration produce the same order in every epoch.
`drop_last=True` removes the incomplete final batch without crossing epoch
boundaries. Randomized transforms should use their own deterministic randomness
when reproducible augmentation values—not only sample order—are required.

Workers do not create MyTorch tensors or access CUDA. Tensor construction and
host-to-device transfer happen in the main consumer thread immediately before
a batch is returned. `pin_memory=True` lazily imports CuPy only for a CUDA
device, copies NumPy arrays through pinned host allocations, and then transfers
them to the requested device. It is a no-op on CPU and does not affect CPU-only
installation requirements.

## CIFAR-10

`CIFAR10Dataset` reads the official Python batch format and performs no network
access. Download `cifar-10-python.tar.gz` from
<https://www.cs.toronto.edu/~kriz/cifar.html>, verify/extract it, and pass either
the extracted `cifar-10-batches-py` directory or its parent as `root`.

```python
from mytorch.data.datasets import CIFAR10Dataset

train = CIFAR10Dataset("data/CIFAR10", train=True, transforms=[...])
test = CIFAR10Dataset("data/CIFAR10", train=False)
```

Images are decoded as HWC for transforms and returned as normalized contiguous
CHW float32 arrays by default. `layout="HWC"` and `normalize=False` are optional.
No CIFAR data or generated dataset is stored in this repository.

## Benchmark

The benchmark reports Python/platform/backend versions, complete loader
configuration, raw loader throughput, and one warmed-up TinyCNN training epoch:

```bash
python -m benchmarks.bench_dataloader --device cpu --num-workers 0 2
python -m benchmarks.bench_dataloader --device cuda --num-workers 0 2 --pin-memory
```

`--delay-ms` models host decode/I/O work in the synthetic dataset. Results are
measurements of the current machine and configuration; no presentation numbers
are embedded in the benchmark.
