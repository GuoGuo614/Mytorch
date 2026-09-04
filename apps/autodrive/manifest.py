"""Create a grouped JSONL manifest from the legacy DonkeyCar image folders."""

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = {
    "image_path", "steering", "throttle", "map_name", "split", "run_id"
}
MAP_NAMES = {
    "data_circuit": "circuit",
    "data_generated_track_thro": "generated-track",
    "data_mountain_track_thro": "mountain-track",
}


def validate_record(record):
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"manifest record is missing fields: {missing}")
    if record["split"] not in {"train", "val"}:
        raise ValueError(f"invalid manifest split: {record['split']!r}")
    float(record["steering"])
    float(record["throttle"])


def _legacy_labels(path, default_throttle):
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"legacy filename has no steering label: {path.name}")
    steering = float(parts[-2] if len(parts) >= 3 else parts[-1])
    if len(parts) >= 3:
        return steering, float(parts[-1]), "legacy_filename"
    return steering, float(default_throttle), "default_throttle"


def _legacy_frame_index(path):
    try:
        return int(path.stem.split("_", 1)[0])
    except ValueError as error:
        raise ValueError(
            f"legacy filename has no numeric frame index: {path.name}"
        ) from error


def build_legacy_records(data_root, *, val_ratio=0.2, seed=256,
                         default_throttle=0.2, group_size=500, maps=None):
    """Import legacy labels and create a map-stratified grouped split.

    The flattened legacy folders no longer retain their original drive IDs.
    Consecutive frame-number blocks are therefore used as conservative pseudo
    runs.  Every selected map contributes blocks to both train and validation.
    """
    root = Path(data_root).resolve()
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    run_directories = [
        directory for directory in sorted(root.iterdir())
        if directory.is_dir() and any(directory.rglob("*.jpg"))
    ]
    requested_maps = None if maps is None else set(maps)
    available_maps = {
        MAP_NAMES.get(directory.name, directory.name)
        for directory in run_directories
    }
    if requested_maps is not None:
        unknown = sorted(requested_maps - available_maps)
        if unknown:
            raise ValueError(
                f"unknown maps {unknown}; available maps: {sorted(available_maps)}"
            )
        run_directories = [
            directory for directory in run_directories
            if MAP_NAMES.get(directory.name, directory.name) in requested_maps
        ]
    if not run_directories:
        raise ValueError("no legacy image directories were selected")

    pending = []
    groups_by_map = {}
    for directory in run_directories:
        map_name = MAP_NAMES.get(directory.name, directory.name)
        for image in sorted(directory.rglob("*.jpg")):
            frame_index = _legacy_frame_index(image)
            block_start = frame_index // group_size * group_size
            run_id = (
                f"{map_name}:legacy-frames-"
                f"{block_start:05d}-{block_start + group_size - 1:05d}"
            )
            groups_by_map.setdefault(map_name, set()).add(run_id)
            pending.append((directory, image, map_name, run_id, frame_index))

    rng = np.random.default_rng(seed)
    validation_runs = set()
    for map_name in sorted(groups_by_map):
        groups = sorted(groups_by_map[map_name])
        if len(groups) < 2:
            raise ValueError(
                f"map {map_name!r} has only {len(groups)} frame block; "
                "reduce group_size so it can contribute to both splits"
            )
        validation_count = max(
            1, min(len(groups) - 1, round(len(groups) * val_ratio))
        )
        validation_runs.update(rng.permutation(groups)[:validation_count])

    records = []
    for directory, image, map_name, run_id, frame_index in pending:
        split = "val" if run_id in validation_runs else "train"
        steering, throttle, source = _legacy_labels(image, default_throttle)
        records.append({
            "image_path": image.relative_to(root).as_posix(),
            "steering": steering,
            "throttle": throttle,
            "map_name": map_name,
            "split": split,
            "run_id": run_id,
            "frame_index": frame_index,
            "label_source": source,
        })
    return records


def write_manifest(records, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            validate_record(record)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/DonkeyCar")
    parser.add_argument("--output", default="data/DonkeyCar/manifest.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--default-throttle", type=float, default=0.2)
    parser.add_argument(
        "--group-size", type=int, default=500,
        help="legacy frame numbers per pseudo-run block",
    )
    parser.add_argument(
        "--maps", nargs="+", default=None,
        help="map names to include; omitted means every detected map",
    )
    args = parser.parse_args()
    records = build_legacy_records(
        args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        default_throttle=args.default_throttle,
        group_size=args.group_size,
        maps=args.maps,
    )
    write_manifest(records, args.output)
    counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "val")
    }
    map_counts = {
        map_name: {
            split: sum(
                record["map_name"] == map_name and record["split"] == split
                for record in records
            )
            for split in ("train", "val")
        }
        for map_name in sorted({record["map_name"] for record in records})
    }
    print(json.dumps({"output": args.output, **counts, "maps": map_counts}))


if __name__ == "__main__":
    main()
