"""Run closed-loop DonkeyCar driving from JSON + KernelLeaf NPZ artifacts."""

import argparse
import json

import numpy as np

import kernelleaf as kl

from .artifacts import load_inference_config
from .dataset import preprocess_rgb_frame
from .model import AutoDriveResNet
from .config import add_simulator_arguments, environment_name


class AutoDrivePolicy:
    """Load artifacts and turn RGB frames into safe, smoothed actions."""

    def __init__(self, config_path, *, checkpoint=None, device=None):
        self.config = load_inference_config(config_path)
        self.device = device or kl.cpu()
        model_config = self.config["model"]
        self.model = AutoDriveResNet(
            base_channels=model_config["base_channels"],
            blocks=model_config.get("blocks", (1, 1, 1)),
            throttle_min=model_config["throttle_min"],
            throttle_max=model_config["throttle_max"],
            device=self.device,
        )
        self.checkpoint = checkpoint or self.config["checkpoint"]
        self.checkpoint_metadata = kl.load_checkpoint(
            self.checkpoint, self.model
        )
        saved_model = self.checkpoint_metadata.get("config", {})
        for key in ("base_channels", "blocks", "throttle_min", "throttle_max"):
            if key in saved_model and saved_model[key] != model_config.get(key):
                raise ValueError(
                    f"JSON/NPZ model configuration mismatch for {key}: "
                    f"json={model_config.get(key)!r}, npz={saved_model[key]!r}"
                )
        saved_normalization = self.checkpoint_metadata.get("normalization", {})
        for key in ("mean", "std"):
            if key in saved_normalization and not np.allclose(
                saved_normalization[key], self.config["preprocessing"][key]
            ):
                raise ValueError(
                    f"JSON/NPZ preprocessing mismatch for {key}"
                )
        self.model.eval()
        self.preprocessing = self.config["preprocessing"]
        self.control = self.config["control"]
        self.previous_steering = 0.0
        self.previous_throttle = float(self.control["startup_throttle"])
        self.last_prediction = None

    def reset(self):
        self.previous_steering = 0.0
        self.previous_throttle = float(self.control["startup_throttle"])

    @staticmethod
    def _smooth(previous, current, factor):
        return (1.0 - factor) * previous + factor * current

    def predict(self, frame):
        values = preprocess_rgb_frame(
            frame,
            self.preprocessing["image_size"],
            self.preprocessing["mean"],
            self.preprocessing["std"],
        )
        inputs = kl.Tensor(
            values[None, ...], device=self.device, requires_grad=False
        )
        steering_tensor, throttle_tensor = self.model(inputs)
        raw_steering = float(steering_tensor.numpy().reshape(-1)[0])
        raw_throttle = float(throttle_tensor.numpy().reshape(-1)[0])
        if not np.isfinite(raw_steering) or not np.isfinite(raw_throttle):
            raise FloatingPointError("model produced a non-finite control value")
        steering = float(np.clip(
            raw_steering,
            self.control["steering_min"],
            self.control["steering_max"],
        ))
        throttle = float(np.clip(
            raw_throttle,
            self.control["throttle_min"],
            self.control["throttle_max"],
        ))
        if self.control.get("throttle_mode", "predicted") == "fixed":
            throttle = float(np.clip(
                self.control.get("fixed_throttle", 0.2),
                0.0,
                self.control["throttle_max"],
            ))
        steering = self._smooth(
            self.previous_steering,
            steering,
            float(self.control["steering_smoothing"]),
        )
        throttle = self._smooth(
            self.previous_throttle,
            throttle,
            float(self.control["throttle_smoothing"]),
        )
        steering = float(np.clip(
            steering,
            self.control["steering_min"],
            self.control["steering_max"],
        ))
        throttle = float(np.clip(
            throttle,
            self.control["throttle_min"],
            self.control["throttle_max"],
        ))
        self.previous_steering = steering
        self.previous_throttle = throttle
        self.last_prediction = {
            "raw_steering": raw_steering,
            "raw_throttle": raw_throttle,
            "steering": steering,
            "throttle": throttle,
        }
        return np.asarray([steering, throttle], dtype=np.float32)

    def safe_action(self):
        if self.control["failure_mode"] == "stop":
            throttle = 0.0
        else:
            throttle = float(np.clip(
                self.control["safe_throttle"],
                0.0,
                self.control["throttle_max"],
            ))
        return np.asarray([0.0, throttle], dtype=np.float32)

    def startup_action(self):
        throttle = float(np.clip(
            self.control["startup_throttle"],
            0.0,
            self.control["throttle_max"],
        ))
        return np.asarray([0.0, throttle], dtype=np.float32)


class GymDonkeyAdapter:
    """Optional gym-donkeycar adapter; imports simulator packages lazily."""

    def __init__(self, env_name):
        try:
            import gym
            import gym_donkeycar  # noqa: F401 - registers environments
        except ImportError as error:
            raise RuntimeError(
                "DonkeyCar simulation requires gym and gym-donkeycar"
            ) from error
        self.env_name = env_name
        self.environment = gym.make(env_name)

    @staticmethod
    def _observation(reset_result):
        return reset_result[0] if isinstance(reset_result, tuple) else reset_result

    def reset(self):
        return self._observation(self.environment.reset())

    def step(self, action):
        result = self.environment.step(np.asarray(action, dtype=np.float32))
        if len(result) == 5:
            frame, reward, terminated, truncated, info = result
            return frame, reward, bool(terminated or truncated), info
        frame, reward, done, info = result
        return frame, reward, bool(done), info

    def close(self):
        close = getattr(self.environment, "close", None)
        if close is not None:
            close()


class ClosedLoopDriver:
    def __init__(self, adapter, policy, *, max_steps=6000, log_interval=50):
        if max_steps <= 0 or log_interval <= 0:
            raise ValueError("max_steps and log_interval must be positive")
        self.adapter = adapter
        self.policy = policy
        self.max_steps = int(max_steps)
        self.log_interval = int(log_interval)

    @staticmethod
    def _is_frame(value):
        return isinstance(value, np.ndarray) and value.ndim == 3 \
            and value.shape[-1] == 3

    def _reset_frame(self):
        self.policy.reset()
        frame = self.adapter.reset()
        if not self._is_frame(frame):
            frame, _, _, _ = self.adapter.step(self.policy.startup_action())
        if not self._is_frame(frame):
            raise ValueError("simulator reset did not provide an RGB frame")
        return frame

    def run(self):
        statistics = {
            "steps": 0,
            "episodes": 0,
            "inference_failures": 0,
            "stopped_for_safety": False,
        }
        consecutive_failures = 0
        try:
            frame = self._reset_frame()
            for step in range(self.max_steps):
                error = None
                try:
                    action = self.policy.predict(frame)
                    consecutive_failures = 0
                except Exception as prediction_error:
                    error = prediction_error
                    statistics["inference_failures"] += 1
                    consecutive_failures += 1
                    action = self.policy.safe_action()
                frame, reward, done, info = self.adapter.step(action)
                statistics["steps"] += 1
                if step % self.log_interval == 0 or error is not None:
                    print(json.dumps({
                        "step": step,
                        "steering": float(action[0]),
                        "throttle": float(action[1]),
                        "reward": float(reward),
                        "prediction_error": None if error is None else str(error),
                    }, sort_keys=True))
                if consecutive_failures >= int(
                    self.policy.control["max_consecutive_failures"]
                ):
                    statistics["stopped_for_safety"] = True
                    break
                if done:
                    statistics["episodes"] += 1
                    frame = self._reset_frame()
            return statistics
        finally:
            self.adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    add_simulator_arguments(parser)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()
    device = kl.cuda(0) if args.device == "cuda" else kl.cpu()
    policy = AutoDrivePolicy(
        args.config, checkpoint=args.checkpoint, device=device
    )
    adapter = GymDonkeyAdapter(environment_name(args))
    statistics = ClosedLoopDriver(
        adapter, policy, max_steps=args.max_steps,
        log_interval=args.log_interval,
    ).run()
    print(json.dumps({"summary": statistics}, sort_keys=True))


if __name__ == "__main__":
    main()
