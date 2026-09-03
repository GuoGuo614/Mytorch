import os
import struct
import sys
import time

import numpy as np

sys.path.append("./python")

import needle as ndl
import needle.nn as nn


def _load_mnist_images(path):
    with open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">4I", f.read(16))
        assert magic == 2051, f"Invalid MNIST image file: {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(num_images, rows, cols).astype(np.float32) / 255.0
    return images[:, None, :, :]


def _load_mnist_labels(path):
    with open(path, "rb") as f:
        magic, num_labels = struct.unpack(">2I", f.read(8))
        assert magic == 2049, f"Invalid MNIST label file: {path}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int32)


def load_mnist_raw(root="data/MNIST/raw"):
    train_images = _load_mnist_images(os.path.join(root, "train-images-idx3-ubyte"))
    train_labels = _load_mnist_labels(os.path.join(root, "train-labels-idx1-ubyte"))
    test_images = _load_mnist_images(os.path.join(root, "t10k-images-idx3-ubyte"))
    test_labels = _load_mnist_labels(os.path.join(root, "t10k-labels-idx1-ubyte"))
    return (train_images, train_labels), (test_images, test_labels)


class LeNet5(nn.Module):
    """
    LeNet-5 style model for MNIST classification.
    This implementation uses stride-2 convolutions for downsampling.
    """

    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv(1, 6, 5, stride=1, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv(6, 16, 5, stride=2, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv(16, 120, 5, stride=2, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120 * 7 * 7, 84, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(84, 10, device=device, dtype=dtype),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MNISTConvNet(nn.Module):
    """PyTorch-style MNIST CNN using the merged pooling operators."""

    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=0, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=0, device=device, dtype=dtype),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def epoch_general_mnist(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    if opt is None:
        model.eval()
    else:
        model.train()

    correct = 0
    total_loss = 0.0
    total_samples = 0

    for X, y in dataloader:
        if opt is not None:
            opt.reset_grad()

        logits = model(X)
        loss = loss_fn(logits, y)

        if opt is not None:
            loss.backward()
            opt.step()

        labels = y.numpy().astype(np.int32)
        preds = np.argmax(logits.numpy(), axis=1)
        batch_size = labels.shape[0]

        correct += np.sum(preds == labels)
        total_loss += float(loss.numpy()) * batch_size
        total_samples += batch_size

    return correct / total_samples, total_loss / total_samples


def train_mnist(
    model,
    train_loader,
    test_loader,
    n_epochs=5,
    optimizer=ndl.optim.Adam,
    lr=1e-3,
    weight_decay=1e-4,
):
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        start_time = time.time()
        train_acc, train_loss = epoch_general_mnist(train_loader, model, opt=opt)
        test_acc, test_loss = epoch_general_mnist(test_loader, model, opt=None)
        end_time = time.time()
        print(
            f"Epoch {epoch}: "
            f"time={end_time - start_time:.2f}s, "
            f"train_acc={train_acc:.4f}, train_loss={train_loss:.4f}, "
            f"test_acc={test_acc:.4f}, test_loss={test_loss:.4f}"
        )


def build_mnist_dataloaders(batch_size=128, device=None, root="data/MNIST/raw"):
    (train_images, train_labels), (test_images, test_labels) = load_mnist_raw(root=root)

    train_dataset = ndl.data.NDArrayDataset(train_images, train_labels)
    test_dataset = ndl.data.NDArrayDataset(test_images, test_labels)

    train_loader = ndl.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, device=device
    )
    test_loader = ndl.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, device=device
    )
    return train_loader, test_loader


if __name__ == "__main__":
    np.random.seed(0)

    device = ndl.cuda() if ndl.cuda().enabled() else ndl.cpu()
    train_loader, test_loader = build_mnist_dataloaders(
        batch_size=128, device=device, root="data/MNIST/raw"
    )

    model = LeNet5(device=device, dtype="float32")
    train_mnist(
        model,
        train_loader,
        test_loader,
        n_epochs=5,
        optimizer=ndl.optim.Adam,
        lr=1e-3,
        weight_decay=1e-4,
    )
