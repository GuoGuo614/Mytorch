"""Shared simulator and keyboard control configuration (no simulator imports)."""

from dataclasses import dataclass, asdict
import math
import re


MAP_ENVS = {name: f"donkey-{name}-v0" for name in (
    "generated-track", "mountain-track", "warren-track", "warehouse", "circuit"
)}
MAP_ENVS["circuit"] = "donkey-circuit-launch-track-v0"


def map_slug(map_name):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(map_name)).strip("-").lower()
    if not slug:
        raise ValueError(f"map name cannot form a filename: {map_name!r}")
    return slug


def map_set_slug(map_names):
    names = sorted({map_slug(name) for name in map_names})
    if not names:
        raise ValueError("at least one map is required for artifact naming")
    return "__".join(names)


def add_simulator_arguments(parser):
    parser.add_argument("--map", choices=sorted(MAP_ENVS), default="mountain-track")
    parser.add_argument("--env-name", default=None, help="override map environment ID")


def environment_name(args):
    return args.env_name or MAP_ENVS[args.map]


@dataclass
class CollectionConfig:
    base_throttle: float = 0.2
    max_throttle: float = 0.5
    throttle_rise: float = 0.2
    throttle_decay: float = 0.1
    steering_rate: float = 2.0
    steering_return: float = 3.0
    steering_limit: float = 1.0
    fps: float = 20.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("collection parameters must be finite")
        if not 0 <= self.base_throttle <= self.max_throttle <= 1:
            raise ValueError("require 0 <= base_throttle <= max_throttle <= 1")
        if not 0 < self.steering_limit <= 1 or self.fps <= 0:
            raise ValueError("invalid steering_limit or fps")
        if min(self.throttle_rise, self.throttle_decay,
               self.steering_rate, self.steering_return) < 0:
            raise ValueError("control rates must be non-negative")
