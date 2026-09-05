"""Evaluate a V7 AutoDrive checkpoint on a manifest validation split."""

import argparse
import json

import kernelleaf as kl
from kernelleaf.data import DataLoader

from .dataset import AutoDriveDataset
from .model import AutoDriveResNet
from .train import (
    default_checkpoint_path, evaluate_model, selected_manifest_maps,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/DonkeyCar/manifest.jsonl")
    parser.add_argument(
        "--checkpoint", default=None,
        help="NPZ path; defaults to checkpoints/autodrive_<maps>.npz",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=60)
    parser.add_argument("--image-width", type=int, default=80)
    parser.add_argument("--throttle-min", type=float, default=0.0)
    parser.add_argument("--throttle-max", type=float, default=1.0)
    parser.add_argument("--lambda-throttle", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--maps", nargs="+", default=None,
        help="map names to evaluate; defaults to the checkpoint map set",
    )
    args = parser.parse_args()
    selected_maps = selected_manifest_maps(args.manifest, args.maps)
    if args.checkpoint is None:
        args.checkpoint = str(default_checkpoint_path(selected_maps))
    device = kl.cuda(0) if args.device == "cuda" else kl.cpu()
    checkpoint = kl.inspect_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    normalization = checkpoint["normalization"]
    image_size = tuple(config.get(
        "image_size", [args.image_height, args.image_width]
    ))
    model = AutoDriveResNet(
        base_channels=config.get("base_channels", args.base_channels),
        blocks=config.get("blocks", (1, 1, 1)),
        throttle_min=config.get("throttle_min", args.throttle_min),
        throttle_max=config.get("throttle_max", args.throttle_max),
        device=device,
    )
    metadata = kl.load_checkpoint(args.checkpoint, model)
    dataset = AutoDriveDataset(
        args.manifest, "val", image_size=image_size, augment=False,
        mean=normalization.get("mean", (0.485, 0.456, 0.406)),
        std=normalization.get("std", (0.229, 0.224, 0.225)),
        maps=args.maps if args.maps is not None else config.get("maps"),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, device=device,
        num_workers=args.num_workers, pin_memory=device.kind == "cuda",
    )
    metrics = evaluate_model(
        model, loader, config.get("lambda_throttle", args.lambda_throttle)
    )
    print(json.dumps({"checkpoint": metadata, "metrics": metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
