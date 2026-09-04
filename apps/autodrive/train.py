"""Train the V7 MyTorch dual-head AutoDrive ResNet."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import mytorch as mt
import mytorch.nn as nn
from mytorch.data import DataLoader

from .dataset import AutoDriveDataset, DEFAULT_MEAN, DEFAULT_STD
from .model import AutoDriveResNet
from .artifacts import build_inference_config, save_inference_config


def mse_loss(prediction, target):
    difference = prediction - target
    return (difference * difference).sum() / prediction.shape[0]


def train_epoch(model, loader, optimizer, lambda_throttle=1.0):
    model.train()
    totals = {"loss": 0.0, "steer_loss": 0.0, "throttle_loss": 0.0}
    samples = 0
    started = time.perf_counter()
    for images, steering_target, throttle_target in loader:
        optimizer.reset_grad()
        steering, throttle = model(images)
        steering_loss = mse_loss(steering, steering_target)
        throttle_loss = mse_loss(throttle, throttle_target)
        loss = steering_loss + lambda_throttle * throttle_loss
        loss.backward()
        optimizer.step()
        batch = images.shape[0]
        totals["loss"] += float(loss.numpy()) * batch
        totals["steer_loss"] += float(steering_loss.numpy()) * batch
        totals["throttle_loss"] += float(throttle_loss.numpy()) * batch
        samples += batch
    if not samples:
        raise ValueError("training loader produced no samples")
    return {
        **{key: value / samples for key, value in totals.items()},
        "samples": samples,
        "epoch_seconds": time.perf_counter() - started,
    }


def evaluate_model(model, loader, lambda_throttle=1.0):
    model.eval()
    totals = {
        "loss": 0.0,
        "steer_loss": 0.0,
        "throttle_loss": 0.0,
        "steer_mae": 0.0,
        "throttle_mae": 0.0,
    }
    samples = 0
    for images, steering_target, throttle_target in loader:
        steering, throttle = model(images)
        steering_loss = mse_loss(steering, steering_target)
        throttle_loss = mse_loss(throttle, throttle_target)
        batch = images.shape[0]
        totals["steer_loss"] += float(steering_loss.numpy()) * batch
        totals["throttle_loss"] += float(throttle_loss.numpy()) * batch
        totals["loss"] += float(
            steering_loss.numpy() + lambda_throttle * throttle_loss.numpy()
        ) * batch
        totals["steer_mae"] += np.abs(
            steering.numpy() - steering_target.numpy()
        ).sum()
        totals["throttle_mae"] += np.abs(
            throttle.numpy() - throttle_target.numpy()
        ).sum()
        samples += batch
    if not samples:
        raise ValueError("validation loader produced no samples")
    return {**{key: float(value) / samples for key, value in totals.items()},
            "samples": samples}


def _configuration(args):
    return {
        "manifest": str(args.manifest),
        "image_size": [args.image_height, args.image_width],
        "base_channels": args.base_channels,
        "blocks": [1, 1, 1],
        "throttle_min": args.throttle_min,
        "throttle_max": args.throttle_max,
        "lambda_throttle": args.lambda_throttle,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "maps": args.maps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/DonkeyCar/manifest.jsonl")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-throttle", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=60)
    parser.add_argument("--image-width", type=int, default=80)
    parser.add_argument("--throttle-min", type=float, default=0.0)
    parser.add_argument("--throttle-max", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--maps", nargs="+", default=None,
        help="manifest map names to train; omitted means all maps",
    )
    parser.add_argument("--checkpoint", default="checkpoints/autodrive_v7.npz")
    parser.add_argument(
        "--run-config", default=None,
        help="JSON inference config; defaults to the checkpoint path with .json",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.lambda_throttle < 0:
        parser.error("epochs must be positive and lambda-throttle non-negative")
    np.random.seed(args.seed)
    device = mt.cuda(0) if args.device == "cuda" else mt.cpu()
    image_size = (args.image_height, args.image_width)
    resume_info = mt.inspect_checkpoint(args.checkpoint) if args.resume else None
    normalization = (
        resume_info["normalization"] if resume_info is not None else {}
    )
    mean = normalization.get("mean", DEFAULT_MEAN)
    std = normalization.get("std", DEFAULT_STD)
    train_dataset = AutoDriveDataset(
        args.manifest, "train", image_size=image_size, augment=True,
        seed=args.seed, mean=mean, std=std, maps=args.maps,
    )
    validation_dataset = AutoDriveDataset(
        args.manifest, "val", image_size=image_size, augment=False,
        seed=args.seed, mean=mean, std=std, maps=args.maps,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "device": device,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": device.kind == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, seed=args.seed, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )
    config = _configuration(args)
    if train_dataset.map_names != validation_dataset.map_names:
        raise ValueError(
            "training and validation splits contain different maps: "
            f"train={train_dataset.map_names}, val={validation_dataset.map_names}"
        )
    config["maps"] = train_dataset.map_names
    config["throttle_mode"] = (
        "predicted" if train_dataset.has_recorded_throttle else "fixed"
    )
    config["fixed_throttle"] = 0.2
    model = AutoDriveResNet(
        base_channels=args.base_channels,
        throttle_min=args.throttle_min,
        throttle_max=args.throttle_max,
        device=device,
    )
    optimizer = mt.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 1
    if args.resume:
        metadata = mt.load_checkpoint(
            args.checkpoint, model, optimizer=optimizer
        )
        saved = metadata["config"]
        for key in ("image_size", "base_channels", "blocks",
                    "throttle_min", "throttle_max", "maps"):
            if saved.get(key) != config.get(key):
                raise ValueError(
                    f"resume configuration mismatch for {key}: "
                    f"checkpoint={saved.get(key)!r}, current={config.get(key)!r}"
                )
        start_epoch = metadata["epoch"] + 1
        train_loader.set_epoch(start_epoch - 1)
    for epoch in range(start_epoch, args.epochs + 1):
        training = train_epoch(
            model, train_loader, optimizer, args.lambda_throttle
        )
        validation = evaluate_model(
            model, validation_loader, args.lambda_throttle
        )
        record = {
            "epoch": epoch,
            "lr": optimizer.lr,
            "train": training,
            "val": validation,
        }
        print(json.dumps(record, sort_keys=True))
        mt.save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            epoch=epoch,
            config=config,
            normalization=train_dataset.normalization,
        )
        config_path = args.run_config or str(Path(args.checkpoint).with_suffix(".json"))
        inference_config = build_inference_config(
            args.checkpoint,
            config,
            train_dataset.normalization,
            epoch=epoch,
            metrics=record,
            config_path=config_path,
        )
        save_inference_config(config_path, inference_config)


if __name__ == "__main__":
    main()
