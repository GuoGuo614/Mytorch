import json

import numpy as np
import pytest

import mytorch as mt
from apps.autodrive.artifacts import (
    build_inference_config,
    load_inference_config,
    save_inference_config,
)
from apps.autodrive.dataset import preprocess_rgb_frame
from apps.autodrive.drive import AutoDrivePolicy, ClosedLoopDriver
from apps.autodrive.model import AutoDriveResNet


def _artifacts(tmp_path, *, checkpoint_config=None):
    model_config = {
        "manifest": "synthetic.jsonl",
        "image_size": [16, 20],
        "base_channels": 2,
        "blocks": [1, 1, 1],
        "throttle_min": 0.1,
        "throttle_max": 0.5,
        "lambda_throttle": 1.0,
    }
    normalization = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    checkpoint = tmp_path / "drive.npz"
    model = AutoDriveResNet(
        base_channels=2, throttle_min=0.1, throttle_max=0.5
    )
    mt.save_checkpoint(
        checkpoint,
        model,
        epoch=4,
        config=checkpoint_config or model_config,
        normalization=normalization,
    )
    config_path = tmp_path / "drive.json"
    config = build_inference_config(
        checkpoint,
        model_config,
        normalization,
        epoch=4,
        metrics={"val": {"steer_mae": 0.2}},
        config_path=config_path,
    )
    save_inference_config(config_path, config)
    return config_path, checkpoint, config


def test_inference_json_roundtrip_and_relative_checkpoint(tmp_path):
    config_path, checkpoint, config = _artifacts(tmp_path)
    loaded = load_inference_config(config_path)
    assert loaded["checkpoint"] == str(checkpoint.resolve())
    assert loaded["model"] == config["model"]
    assert loaded["training"]["epoch"] == 4
    assert loaded["training"]["metrics"]["val"]["steer_mae"] == 0.2


def test_policy_loads_json_npz_and_predicts_bounded_smoothed_controls(tmp_path):
    config_path, _, _ = _artifacts(tmp_path)
    policy = AutoDrivePolicy(config_path, device=mt.cpu())
    frame = np.full((120, 160, 3), 127, dtype=np.uint8)
    first = policy.predict(frame)
    second = policy.predict(frame)
    assert first.shape == (2,)
    assert -1 <= first[0] <= 1
    assert 0.1 <= first[1] <= 0.5
    assert -1 <= second[0] <= 1
    assert 0.1 <= second[1] <= 0.5
    assert policy.last_prediction["raw_steering"] != first[0]
    policy.control["failure_mode"] = "stop"
    np.testing.assert_array_equal(policy.safe_action(), [0.0, 0.0])
    policy.control["failure_mode"] = "fixed"
    policy.control["safe_throttle"] = 0.15
    np.testing.assert_allclose(policy.safe_action(), [0.0, 0.15])


def test_policy_uses_fixed_point_two_when_training_has_no_throttle(tmp_path):
    config_path, _, _ = _artifacts(tmp_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config["control"]["throttle_mode"] = "fixed"
    config["control"]["fixed_throttle"] = 0.2
    config["control"]["throttle_smoothing"] = 1.0
    save_inference_config(config_path, config)
    policy = AutoDrivePolicy(config_path)
    action = policy.predict(np.zeros((120, 160, 3), dtype=np.uint8))
    assert action[1] == pytest.approx(0.2)


def test_policy_rejects_mismatched_json_and_checkpoint(tmp_path):
    mismatched = {
        "manifest": "synthetic.jsonl",
        "image_size": [16, 20],
        "base_channels": 3,
        "blocks": [1, 1, 1],
        "throttle_min": 0.1,
        "throttle_max": 0.5,
    }
    config_path, _, _ = _artifacts(
        tmp_path, checkpoint_config=mismatched
    )
    with pytest.raises(ValueError, match="JSON/NPZ model configuration mismatch"):
        AutoDrivePolicy(config_path)


def test_frame_preprocessing_validates_rgb_and_is_finite():
    frame = np.full((12, 16, 3), 0.5, dtype=np.float32)
    result = preprocess_rgb_frame(frame, (8, 10))
    assert result.shape == (3, 8, 10)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    with pytest.raises(ValueError, match="HWC RGB"):
        preprocess_rgb_frame(np.zeros((12, 16)), (8, 10))
    invalid = frame.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        preprocess_rgb_frame(invalid, (8, 10))


class MockAdapter:
    def __init__(self, done_steps=()):
        self.done_steps = set(done_steps)
        self.actions = []
        self.reset_count = 0
        self.closed = False

    def reset(self):
        self.reset_count += 1
        return np.zeros((12, 16, 3), dtype=np.uint8)

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        step = len(self.actions)
        return (
            np.full((12, 16, 3), step, dtype=np.uint8),
            1.0,
            step in self.done_steps,
            {},
        )

    def close(self):
        self.closed = True


class MockPolicy:
    def __init__(self, failures=()):
        self.failures = set(failures)
        self.calls = 0
        self.reset_count = 0
        self.control = {"max_consecutive_failures": 2}

    def reset(self):
        self.reset_count += 1

    def predict(self, frame):
        self.calls += 1
        if self.calls in self.failures:
            raise RuntimeError("mock inference failure")
        return np.asarray([0.25, 0.3], dtype=np.float32)

    def safe_action(self):
        return np.asarray([0.0, 0.0], dtype=np.float32)

    def startup_action(self):
        return np.asarray([0.0, 0.2], dtype=np.float32)


def test_closed_loop_steps_resets_and_closes_adapter(capsys):
    adapter = MockAdapter(done_steps={2})
    policy = MockPolicy()
    statistics = ClosedLoopDriver(
        adapter, policy, max_steps=4, log_interval=10
    ).run()
    assert statistics == {
        "steps": 4,
        "episodes": 1,
        "inference_failures": 0,
        "stopped_for_safety": False,
    }
    assert adapter.reset_count == 2
    assert policy.reset_count == 2
    assert adapter.closed
    assert len(adapter.actions) == 4
    assert json.loads(capsys.readouterr().out)["step"] == 0


def test_closed_loop_uses_safe_action_and_stops_after_repeated_failures():
    adapter = MockAdapter()
    policy = MockPolicy(failures={1, 2, 3})
    statistics = ClosedLoopDriver(
        adapter, policy, max_steps=10, log_interval=50
    ).run()
    assert statistics["steps"] == 2
    assert statistics["inference_failures"] == 2
    assert statistics["stopped_for_safety"]
    assert adapter.closed
    for action in adapter.actions:
        np.testing.assert_array_equal(action, [0.0, 0.0])


@pytest.mark.skipif(not mt.is_cuda_available(), reason="CUDA is unavailable")
def test_policy_loads_checkpoint_and_predicts_on_cuda(tmp_path):
    config_path, _, _ = _artifacts(tmp_path)
    device = mt.cuda(0)
    policy = AutoDrivePolicy(config_path, device=device)
    action = policy.predict(np.zeros((120, 160, 3), dtype=np.uint8))
    device.xp.cuda.get_current_stream().synchronize()
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert policy.model.steering_head.weight.device == device
