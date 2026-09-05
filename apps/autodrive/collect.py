"""Collect timestamped RGB/action pairs using A/D, W and space to brake."""

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .config import (
    CollectionConfig, add_simulator_arguments, environment_name, map_slug,
)
from .drive import GymDonkeyAdapter


class ManualControl:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.steering = 0.0
        self.throttle = self.config.base_throttle

    def update(self, keys, dt):
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        c = self.config
        direction = int("d" in keys) - int("a" in keys)
        if direction:
            self.steering += direction * c.steering_rate * dt
        else:
            self.steering = np.sign(self.steering) * max(
                0, abs(self.steering) - c.steering_return * dt)
        self.steering = float(np.clip(self.steering, -c.steering_limit, c.steering_limit))
        if "space" in keys:
            self.throttle = 0.0
        elif "w" in keys:
            self.throttle = min(c.max_throttle, self.throttle + c.throttle_rise * dt)
        elif self.throttle >= c.base_throttle:
            self.throttle = max(c.base_throttle, self.throttle - c.throttle_decay * dt)
        else:
            self.throttle = min(c.base_throttle, self.throttle + c.throttle_rise * dt)
        return np.asarray([self.steering, self.throttle], dtype=np.float32)


class Keyboard:
    """Optional pygame window: focus it to control the simulator."""
    def __init__(self):
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("keyboard collection requires pygame") from error
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("AutoDrive: A/D steer, W accelerate, SPACE brake, ESC quit")

    def poll(self, frame):
        p = self.pg
        for event in p.event.get():
            if event.type == p.QUIT:
                return {"quit"}
        pressed = p.key.get_pressed()
        keys = {name for name, code in (("a", p.K_a), ("d", p.K_d),
                ("w", p.K_w), ("space", p.K_SPACE), ("quit", p.K_ESCAPE)) if pressed[code]}
        surface = p.surfarray.make_surface(np.asarray(frame).transpose(1, 0, 2))
        self.screen.blit(p.transform.scale(surface, self.screen.get_size()), (0, 0))
        p.display.flip()
        return keys

    def close(self):
        self.pg.quit()


def count_collected_images(output):
    root = Path(output)
    if not root.exists():
        return 0
    return sum(
        path.is_file() and path.suffix.lower() == ".png"
        for path in root.rglob("*")
    )


def collect(adapter, key_source, output, map_name, config, *, max_samples=10000,
            max_steps=None, realtime=True, env_name=None):
    """Write pre-action frames with the exact action sent to the adapter.

    Each episode is a distinct run. Raw records receive a split only during
    audit, once at least two runs per map have been collected.
    """
    from PIL import Image
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when specified")
    root = Path(output)
    existing = count_collected_images(root)
    control = ManualControl(config)
    count = 0
    writer = None
    stop_reason = "sample_limit" if existing >= max_samples else "keyboard"
    try:
        if existing >= max_samples:
            return {
                "existing_samples": existing,
                "collected_samples": 0,
                "total_samples": existing,
                "target_samples": max_samples,
                "stop_reason": "sample_limit",
            }
        frame = adapter.reset()
        new_run = True
        while existing + count < max_samples and (
            max_steps is None or count < max_steps
        ):
            started = time.monotonic()
            keys = key_source.poll(frame)
            if "quit" in keys:
                break
            if new_run:
                run_id = uuid.uuid4().hex
                directory = root / f"{map_slug(map_name)}_{run_id}"
                directory.mkdir(parents=True, exist_ok=False)
                metadata = {"run_id": run_id, "map_name": map_name,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "control": asdict(config), "env_name": env_name,
                            "action_alignment": "pre_action_frame",
                            "target_samples": max_samples,
                            "samples_before_session": existing}
                (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                writer = (directory / "records.jsonl").open("x", encoding="utf-8")
                index = 0
                new_run = False
            action = control.update(keys, 1.0 / config.fps)
            timestamp = datetime.now(timezone.utc).isoformat()
            pixels = np.asarray(frame).copy()
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError("collection requires uint8 HWC RGB frames")
            next_frame, _, done, _ = adapter.step(action)
            image_name = f"{index:08d}.png"
            Image.fromarray(pixels).save(directory / image_name)
            record = {"image_path": image_name, "steering": float(action[0]),
                      "throttle": float(action[1]), "map_name": map_name,
                      "run_id": run_id, "timestamp": timestamp, "frame_index": index,
                      "label_source": "recorded_control"}
            writer.write(json.dumps(record) + "\n")
            writer.flush()
            count += 1
            index += 1
            frame = next_frame
            if done:
                writer.close()
                writer = None
                new_run = True
                control.reset()
                frame = adapter.reset()
            if realtime:
                time.sleep(max(0, 1.0 / config.fps - (time.monotonic() - started)))
        if existing + count >= max_samples:
            stop_reason = "sample_limit"
        elif max_steps is not None and count >= max_steps:
            stop_reason = "step_limit"
        return {
            "existing_samples": existing,
            "collected_samples": count,
            "total_samples": existing + count,
            "target_samples": max_samples,
            "stop_reason": stop_reason,
        }
    finally:
        if writer is not None:
            writer.close()
        try:
            adapter.step(np.zeros(2, dtype=np.float32))
        except Exception:
            # A disconnected simulator cannot receive the final stop command.
            pass
        finally:
            try:
                adapter.close()
            finally:
                key_source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_simulator_arguments(parser)
    parser.add_argument("--output", default="data/DonkeyCar/collected")
    parser.add_argument(
        "--max-samples", type=int, default=10000,
        help="cumulative PNG target under --output (default: 10000)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="optional limit for samples collected by this invocation",
    )
    defaults = CollectionConfig()
    for field in fields(defaults):
        parser.add_argument("--" + field.name.replace("_", "-"), type=float,
                            default=getattr(defaults, field.name))
    args = parser.parse_args()
    config = CollectionConfig(**{f.name: getattr(args, f.name) for f in fields(defaults)})
    keyboard = Keyboard()
    try:
        adapter = GymDonkeyAdapter(environment_name(args))
    except BaseException:
        keyboard.close()
        raise
    print(json.dumps(collect(adapter, keyboard, args.output, args.map, config,
                           max_samples=args.max_samples, max_steps=args.max_steps,
                           env_name=environment_name(args))))


if __name__ == "__main__":
    main()
