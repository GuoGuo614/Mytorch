"""JSON inference sidecars paired with MyTorch NPZ checkpoints."""

import json
import os
from pathlib import Path
import tempfile


INFERENCE_CONFIG_VERSION = 1


def build_inference_config(checkpoint_path, training_config, normalization, *,
                           epoch, metrics=None, config_path=None):
    destination = Path(config_path) if config_path else Path(
        checkpoint_path
    ).with_suffix(".json")
    checkpoint = Path(checkpoint_path).resolve()
    relative_checkpoint = os.path.relpath(
        checkpoint, destination.resolve().parent
    ).replace("\\", "/")
    return {
        "format_version": INFERENCE_CONFIG_VERSION,
        "checkpoint": relative_checkpoint,
        "model": {
            "base_channels": training_config["base_channels"],
            "blocks": training_config.get("blocks", [1, 1, 1]),
            "throttle_min": training_config["throttle_min"],
            "throttle_max": training_config["throttle_max"],
        },
        "preprocessing": {
            "image_size": training_config["image_size"],
            "mean": normalization["mean"],
            "std": normalization["std"],
            "color_order": "RGB",
        },
        "control": {
            "throttle_mode": training_config.get("throttle_mode", "predicted"),
            "fixed_throttle": training_config.get("fixed_throttle", 0.2),
            "steering_min": -1.0,
            "steering_max": 1.0,
            "throttle_min": training_config["throttle_min"],
            "throttle_max": training_config["throttle_max"],
            "steering_smoothing": 0.35,
            "throttle_smoothing": 0.25,
            "startup_throttle": min(
                0.2, training_config["throttle_max"]
            ),
            "failure_mode": "stop",
            "safe_throttle": 0.0,
            "max_consecutive_failures": 3,
        },
        "training": {
            "epoch": int(epoch),
            "manifest": training_config.get("manifest"),
            "metrics": metrics or {},
        },
    }


def validate_inference_config(config):
    if config.get("format_version") != INFERENCE_CONFIG_VERSION:
        raise ValueError("unsupported AutoDrive inference config version")
    for section in ("checkpoint", "model", "preprocessing", "control"):
        if section not in config:
            raise ValueError(f"AutoDrive config is missing {section!r}")
    preprocessing = config["preprocessing"]
    if preprocessing.get("color_order") != "RGB":
        raise ValueError("AutoDrive inference currently requires RGB frames")
    if len(preprocessing.get("image_size", [])) != 2:
        raise ValueError("preprocessing.image_size must contain height and width")
    control = config["control"]
    for key in ("steering_smoothing", "throttle_smoothing"):
        if not 0 < float(control[key]) <= 1:
            raise ValueError(f"control.{key} must be in (0, 1]")
    if control.get("failure_mode") not in {"stop", "fixed"}:
        raise ValueError("control.failure_mode must be 'stop' or 'fixed'")
    if control.get("throttle_mode", "predicted") not in {"predicted", "fixed"}:
        raise ValueError("control.throttle_mode must be 'predicted' or 'fixed'")
    if int(control.get("max_consecutive_failures", 0)) <= 0:
        raise ValueError("control.max_consecutive_failures must be positive")
    return config


def save_inference_config(path, config):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_inference_config(config)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=destination.name + ".", suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(config, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_inference_config(path):
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as file:
        config = validate_inference_config(json.load(file))
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = source.parent / checkpoint
    result = dict(config)
    result["checkpoint"] = str(checkpoint.resolve())
    result["config_path"] = str(source)
    return result
