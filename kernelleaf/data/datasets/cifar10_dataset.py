"""Minimal reader for the official CIFAR-10 Python batch files."""

from pathlib import Path
import pickle

import numpy as np

from ..data_basic import Dataset


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


class CIFAR10Dataset(Dataset):
    """Read CIFAR-10 without downloading data or importing another framework.

    ``root`` may point at either the extracted ``cifar-10-batches-py`` folder
    or its parent. Transforms receive individual HWC images before the optional
    CHW layout conversion.
    """

    def __init__(self, root, train=True, transforms=None, *,
                 layout="CHW", normalize=True):
        super().__init__(transforms)
        if layout not in {"CHW", "HWC"}:
            raise ValueError("layout must be 'CHW' or 'HWC'")
        root = Path(root)
        nested = root / "cifar-10-batches-py"
        self.root = nested if nested.is_dir() else root
        self.train = bool(train)
        self.layout = layout
        self.normalize = bool(normalize)
        names = ([f"data_batch_{index}" for index in range(1, 6)]
                 if self.train else ["test_batch"])
        missing = [str(self.root / name) for name in names
                   if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "CIFAR-10 batch files are missing: " + ", ".join(missing)
                + f". Download and extract {CIFAR10_URL} under {root}."
            )
        images = []
        labels = []
        for name in names:
            with (self.root / name).open("rb") as file:
                batch = pickle.load(file, encoding="latin1")
            data = batch.get("data", batch.get(b"data"))
            batch_labels = batch.get("labels", batch.get(b"labels"))
            if data is None or batch_labels is None:
                raise ValueError(f"invalid CIFAR-10 batch file: {self.root / name}")
            data = np.asarray(data, dtype=np.uint8)
            if data.ndim != 2 or data.shape[1] != 3 * 32 * 32:
                raise ValueError(f"invalid CIFAR-10 image shape in {self.root / name}")
            images.append(data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            labels.append(np.asarray(batch_labels, dtype=np.int64))
        self.images = np.concatenate(images, axis=0)
        self.labels = np.concatenate(labels, axis=0)
        if len(self.images) != len(self.labels):
            raise ValueError("CIFAR-10 image and label counts do not match")

    def __len__(self):
        return len(self.images)

    def _image(self, index):
        image = self.images[index]
        image = image.astype(np.float32)
        if self.normalize:
            image /= 255.0
        image = self.apply_transforms(image)
        if self.layout == "CHW":
            image = np.asarray(image).transpose(2, 0, 1)
        return np.ascontiguousarray(image)

    def __getitem__(self, index):
        if np.ndim(index) == 0:
            integer = int(index)
            return self._image(integer), self.labels[integer]
        indices = np.asarray(index, dtype=np.int64).reshape(-1)
        images = np.stack([self._image(int(integer)) for integer in indices])
        return images, self.labels[indices]

    def get_batch(self, indices):
        return self[indices]
