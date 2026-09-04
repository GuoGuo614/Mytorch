"""Portable NPZ checkpoints for models, optimizers, and training metadata."""

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .autograd import Tensor
from .backend import asnumpy


FORMAT_VERSION = 1


def _encode(value, arrays):
    if isinstance(value, Tensor):
        value = value.realize_cached_data()
    if isinstance(value, np.ndarray) or type(value).__module__.split(".", 1)[0] == "cupy":
        key = f"array_{len(arrays):06d}"
        arrays[key] = asnumpy(value).copy()
        return {"__array__": key}
    if isinstance(value, dict):
        return {str(key): _encode(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, arrays) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"checkpoint metadata cannot encode {type(value).__name__}")


def _decode(value, archive):
    if isinstance(value, dict) and set(value) == {"__array__"}:
        return archive[value["__array__"]].copy()
    if isinstance(value, dict):
        return {key: _decode(item, archive) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, archive) for item in value]
    return value


def save_checkpoint(path, model, optimizer=None, *, epoch=0, config=None,
                    normalization=None):
    """Atomically save a portable, pickle-free training checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    payload = {
        "format_version": FORMAT_VERSION,
        "epoch": int(epoch),
        "config": {} if config is None else config,
        "normalization": {} if normalization is None else normalization,
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
    }
    encoded = _encode(payload, arrays)
    arrays["__metadata__"] = np.asarray(
        json.dumps(encoded, ensure_ascii=False, sort_keys=True)
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=destination.name + ".",
            suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            np.savez_compressed(file, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def inspect_checkpoint(path):
    """Read lightweight checkpoint metadata without constructing a model."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError("not a MyTorch checkpoint: metadata is missing")
        encoded = json.loads(str(archive["__metadata__"].item()))
    version = encoded.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version {version}; "
            f"expected {FORMAT_VERSION}"
        )
    return {
        "epoch": int(encoded["epoch"]),
        "config": encoded.get("config", {}),
        "normalization": encoded.get("normalization", {}),
        "has_optimizer": encoded.get("optimizer") is not None,
    }


def load_checkpoint(path, model, optimizer=None, *, strict=True):
    """Load parameters and optional optimizer state, returning run metadata."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError("not a MyTorch checkpoint: metadata is missing")
        encoded = json.loads(str(archive["__metadata__"].item()))
        payload = _decode(encoded, archive)
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version {version}; "
            f"expected {FORMAT_VERSION}"
        )
    model.load_state_dict(payload["model"], strict=strict)
    saved_optimizer = payload.get("optimizer")
    if optimizer is not None:
        if saved_optimizer is None:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(saved_optimizer)
    return {
        "epoch": int(payload["epoch"]),
        "config": payload.get("config", {}),
        "normalization": payload.get("normalization", {}),
        "has_optimizer": saved_optimizer is not None,
    }
