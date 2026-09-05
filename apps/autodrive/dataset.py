"""Manifest-only DonkeyCar dataset with deterministic train augmentation."""

import json
from pathlib import Path

import numpy as np

from kernelleaf.data import Dataset
from .manifest import validate_record


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def preprocess_rgb_frame(frame, image_size, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """Convert one simulator RGB HWC frame to normalized float32 CHW."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "AutoDrive preprocessing requires Pillow; install the autodrive extra"
        ) from error
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an HWC RGB frame, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("frame contains non-finite values")
    if array.dtype != np.uint8:
        scale = 255.0 if array.size and float(array.max()) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    height, width = (int(value) for value in image_size)
    image = Image.fromarray(array, mode="RGB")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((width, height), resampling)
    values = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    values = (values - mean) / std
    return np.ascontiguousarray(values.transpose(2, 0, 1))


class AutoDriveDataset(Dataset):
    def __init__(self, manifest_path, split, *, image_size=(60, 80),
                 mean=DEFAULT_MEAN, std=DEFAULT_STD, augment=None, seed=0,
                 maps=None):
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.image_size = tuple(int(value) for value in image_size)
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive integers")
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) \
                or np.any(self.std <= 0):
            raise ValueError("mean/std must contain three values and std must be positive")
        self.augment = split == "train" if augment is None else bool(augment)
        if split != "train" and self.augment:
            raise ValueError("augmentation is only allowed for the training split")
        self.seed = int(seed)
        self.epoch = 0
        self.requested_maps = None if maps is None else set(maps)
        self.records = []
        with self.manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    validate_record(record)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid manifest line {line_number}: {error}"
                    ) from error
                if record["split"] == split and (
                    self.requested_maps is None
                    or record["map_name"] in self.requested_maps
                ):
                    self.records.append(record)
        if not self.records:
            suffix = "" if maps is None else f" for maps {sorted(self.requested_maps)}"
            raise ValueError(f"manifest contains no {split!r} records{suffix}")

    @property
    def map_names(self):
        return sorted({record["map_name"] for record in self.records})

    @property
    def normalization(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @property
    def has_recorded_throttle(self):
        return any(
            record.get("label_source") != "default_throttle"
            for record in self.records
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def _path(self, record):
        path = Path(record["image_path"])
        return path if path.is_absolute() else self.manifest_path.parent / path

    def __getitem__(self, index):
        try:
            from PIL import Image, ImageEnhance
        except ImportError as error:
            raise RuntimeError(
                "AutoDriveDataset requires Pillow; install the autodrive extra"
            ) from error
        record = self.records[int(index)]
        path = self._path(record)
        if not path.is_file():
            raise FileNotFoundError(f"manifest image does not exist: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            image = image.resize(
                (self.image_size[1], self.image_size[0]), resampling
            )
            steering = float(record["steering"])
            if self.augment:
                rng = np.random.default_rng(np.random.SeedSequence([
                    self.seed, self.epoch, int(index)
                ]))
                if rng.random() < 0.5:
                    transpose = getattr(Image, "Transpose", Image)
                    image = image.transpose(transpose.FLIP_LEFT_RIGHT)
                    steering = -steering
                image = ImageEnhance.Brightness(image).enhance(
                    float(rng.uniform(0.85, 1.15))
                )
            array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        array = np.ascontiguousarray(array.transpose(2, 0, 1))
        return (
            array,
            np.asarray([steering], dtype=np.float32),
            np.asarray([record["throttle"]], dtype=np.float32),
        )
