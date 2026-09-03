"""Residual MLP MNIST example."""

from pathlib import Path

import numpy as np

import mytorch as torch
import mytorch.nn as nn
from mytorch.data import DataLoader
from mytorch.data.datasets import MNISTDataset


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1,
                  device=None, dtype="float32"):
    branch = nn.Sequential(
        nn.Linear(dim, hidden_dim, device=device, dtype=dtype),
        norm(hidden_dim, device=device, dtype=dtype),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim, device=device, dtype=dtype),
        norm(dim, device=device, dtype=dtype),
    )
    return nn.Sequential(nn.Residual(branch), nn.ReLU())


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
    device=None,
    dtype="float32",
):
    block_dim = hidden_dim // 2
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(dim, hidden_dim, device=device, dtype=dtype),
        nn.ReLU(),
        *[
            ResidualBlock(
                hidden_dim, block_dim, norm, drop_prob, device, dtype
            )
            for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes, device=device, dtype=dtype),
    )


def epoch(dataloader, model, opt=None):
    model.train() if opt is not None else model.eval()
    loss_fn = nn.SoftmaxLoss()
    total_loss = 0.0
    total_errors = 0
    total_samples = 0
    for inputs, labels in dataloader:
        if opt is not None:
            opt.reset_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        if opt is not None:
            loss.backward()
            opt.step()
        predictions = torch.ops.argmax(logits, axis=1).numpy()
        labels_numpy = labels.numpy()
        total_errors += int(np.sum(predictions != labels_numpy))
        total_loss += float(loss.numpy()) * inputs.shape[0]
        total_samples += inputs.shape[0]
    return total_errors / total_samples, total_loss / total_samples


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=torch.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir=None,
    device=None,
):
    np.random.seed(4)
    device = torch.cpu() if device is None else device
    data_dir = (
        Path(__file__).resolve().parents[1] / "data" / "MNIST" / "raw"
        if data_dir is None else Path(data_dir)
    )
    train_set = MNISTDataset(
        data_dir / "train-images-idx3-ubyte.gz",
        data_dir / "train-labels-idx1-ubyte.gz",
    )
    test_set = MNISTDataset(
        data_dir / "t10k-images-idx3-ubyte.gz",
        data_dir / "t10k-labels-idx1-ubyte.gz",
    )
    train_loader = DataLoader(train_set, batch_size, shuffle=True, device=device)
    test_loader = DataLoader(test_set, batch_size, device=device)
    model = MLPResNet(784, hidden_dim=hidden_dim, device=device)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_metrics = test_metrics = None
    for _ in range(epochs):
        train_metrics = epoch(train_loader, model, opt)
        test_metrics = epoch(test_loader, model)
    return train_metrics, test_metrics


if __name__ == "__main__":
    print(train_mnist())
