"""Compare synchronous and bounded asynchronous DataLoader throughput."""

import argparse
import json
import platform
import sys
import time

import numpy as np

import mytorch as mt
import mytorch.nn as nn
from mytorch.data import DataLoader, Dataset


class SyntheticCIFAR(Dataset):
    def __init__(self, samples, seed, delay_ms=0.0):
        rng = np.random.default_rng(seed)
        self.images = rng.normal(size=(samples, 3, 32, 32)).astype(np.float32)
        self.labels = rng.integers(0, 10, size=samples, dtype=np.int64)
        self.delay = delay_ms / 1000.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, indices):
        if self.delay:
            time.sleep(self.delay)
        return self.images[indices], self.labels[indices]

    def get_batch(self, indices):
        return self[indices]


class TinyCNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1, device=device)
        self.pool = nn.MaxPool2d(2)
        self.linear = nn.Linear(8 * 16 * 16, 10, device=device)

    def forward(self, inputs):
        values = self.pool(nn.ReLU()(self.conv(inputs)))
        return self.linear(values.reshape((values.shape[0], -1)))


def synchronize(device):
    if device.kind == "cuda":
        device.xp.cuda.get_current_stream().synchronize()


def make_loader(args, dataset, device, workers):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
        num_workers=workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
        seed=args.seed,
        pin_memory=args.pin_memory,
    )


def measure_loader(args, dataset, device, workers):
    loader = make_loader(args, dataset, device, workers)
    total = 0
    started = time.perf_counter()
    for _ in range(args.loader_epochs):
        for images, _ in loader:
            total += images.shape[0]
    synchronize(device)
    elapsed = time.perf_counter() - started
    return total / elapsed, elapsed


def measure_cnn_epoch(args, dataset, device, workers):
    np.random.seed(args.seed)
    model = TinyCNN(device)
    optimizer = mt.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SoftmaxLoss()
    warmup_loader = make_loader(args, dataset, device, workers)
    warmup_images, warmup_labels = next(iter(warmup_loader))
    optimizer.reset_grad()
    warmup_loss = loss_fn(model(warmup_images), warmup_labels)
    warmup_loss.backward()
    optimizer.step()
    synchronize(device)
    loader = make_loader(args, dataset, device, workers)
    started = time.perf_counter()
    samples = 0
    for images, labels in loader:
        optimizer.reset_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        samples += images.shape[0]
    synchronize(device)
    return time.perf_counter() - started, samples


def environment(device):
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "device": str(device),
    }
    if device.kind == "cuda":
        cp = device.xp
        result.update({
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "gpu": cp.cuda.runtime.getDeviceProperties(device.index)["name"].decode(),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", nargs="+", type=int, default=(0, 2))
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--loader-epochs", type=int, default=3)
    parser.add_argument("--delay-ms", type=float, default=2.0,
                        help="synthetic host decode/I/O delay per batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0 or args.loader_epochs <= 0:
        parser.error("samples and loader-epochs must be positive")
    device = mt.cuda(0) if args.device == "cuda" else mt.cpu()
    dataset = SyntheticCIFAR(args.samples, args.seed, args.delay_ms)
    common = environment(device)
    for workers in args.num_workers:
        throughput, loader_seconds = measure_loader(
            args, dataset, device, workers
        )
        epoch_seconds, epoch_samples = measure_cnn_epoch(
            args, dataset, device, workers
        )
        print(json.dumps({
            **common,
            "samples": args.samples,
            "batch_size": args.batch_size,
            "num_workers": workers,
            "prefetch_factor": args.prefetch_factor,
            "queue_capacity": workers * args.prefetch_factor,
            "shuffle": True,
            "drop_last": True,
            "seed": args.seed,
            "pin_memory": args.pin_memory,
            "synthetic_delay_ms": args.delay_ms,
            "loader_epochs": args.loader_epochs,
            "loader_seconds": loader_seconds,
            "loader_samples_per_second": throughput,
            "tiny_cnn_epoch_samples": epoch_samples,
            "tiny_cnn_epoch_seconds": epoch_seconds,
        }))


if __name__ == "__main__":
    main()
