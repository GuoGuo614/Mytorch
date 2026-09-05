"""Validate images/labels and split complete runs within each map."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .manifest import write_manifest


def audit_sources(sources):
    from PIL import Image
    records, errors, duplicates = [], [], []
    hashes = {}
    for source in sources:
        source = Path(source).resolve()
        with source.open(encoding="utf-8") as file:
            for number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                location = f"{source}:{number}"
                try:
                    record = json.loads(line)
                    for key in ("image_path", "map_name", "run_id", "steering", "throttle"):
                        if key not in record:
                            raise ValueError(f"missing {key}")
                    for key in ("map_name", "run_id"):
                        if not isinstance(record[key], str) or not record[key]:
                            raise ValueError(f"invalid {key}")
                    steering, throttle = float(record["steering"]), float(record["throttle"])
                    if not np.isfinite([steering, throttle]).all():
                        raise ValueError("non-finite label")
                    if not -1 <= steering <= 1 or not 0 <= throttle <= 1:
                        raise ValueError("label outside physical range")
                    path = (source.parent / record["image_path"]).resolve()
                    with Image.open(path) as image:
                        image.load()
                        pixels = np.asarray(image.convert("RGB"))
                    digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
                    if digest in hashes:
                        duplicates.append({"first": hashes[digest], "duplicate": location})
                    else:
                        hashes[digest] = location
                    records.append({**record, "image_path": str(path),
                                    "steering": steering, "throttle": throttle})
                except (ValueError, TypeError, KeyError, OSError) as error:
                    errors.append({"location": location, "error": str(error)})
    groups = defaultdict(set)
    for record in records:
        if "split" in record:
            groups[(record["map_name"], record["run_id"])].add(record["split"])
    distribution = {}
    for key in ("steering", "throttle"):
        values = np.asarray([record[key] for record in records])
        distribution[key] = ({"min": float(values.min()), "max": float(values.max()),
                              "mean": float(values.mean()), "std": float(values.std()),
                              "histogram": np.histogram(values, bins=10)[0].tolist()}
                             if values.size else {})
    report = {"valid_samples": len(records), "errors": errors,
              "duplicates": duplicates, "maps": dict(Counter(r["map_name"] for r in records)),
              "distribution": distribution,
              "split_leakage": [list(key) for key, splits in groups.items() if len(splits) > 1]}
    return records, report


def group_split(records, val_ratio=0.2, seed=256):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")
    maps = defaultdict(set)
    for record in records:
        maps[record["map_name"]].add(record["run_id"])
    rng = np.random.default_rng(seed)
    validation = set()
    for name, runs in sorted(maps.items()):
        if len(runs) < 2:
            raise ValueError(f"map {name!r} needs at least two runs; collect another drive")
        count = max(1, min(len(runs) - 1, round(len(runs) * val_ratio)))
        validation.update((name, run) for run in rng.permutation(sorted(runs))[:count])
    return [{**record, "split": "val" if (record["map_name"], record["run_id"])
             in validation else "train"} for record in records]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--collection-root")
    source.add_argument("--manifest")
    parser.add_argument("--output", help="write a new grouped training manifest")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=256)
    args = parser.parse_args()
    sources = ([Path(args.manifest)] if args.manifest else
               sorted(Path(args.collection_root).rglob("records.jsonl")))
    if not sources:
        parser.error("no records.jsonl files found")
    records, report = audit_sources(sources)
    print(json.dumps(report, sort_keys=True))
    if report["errors"]:
        raise SystemExit("invalid records found; fix them before exporting")
    if report["split_leakage"] and not args.output:
        raise SystemExit("a run appears in multiple splits")
    if args.output:
        if report["duplicates"]:
            raise SystemExit("duplicate images found; resolve them before splitting to avoid leakage")
        result = group_split(records, args.val_ratio, args.seed)
        destination = Path(args.output).resolve()
        if destination in {Path(path).resolve() for path in sources}:
            parser.error("output must differ from source files")
        write_manifest(result, destination)
        print(json.dumps({"output": str(destination), "splits": dict(Counter(
            r["split"] for r in result))}))


if __name__ == "__main__":
    main()
