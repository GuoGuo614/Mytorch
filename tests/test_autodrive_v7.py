import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import mytorch as mt
import mytorch.nn as nn
from mytorch.data import DataLoader
from apps.autodrive.dataset import AutoDriveDataset
from apps.autodrive.manifest import (
    REQUIRED_FIELDS, build_legacy_records, write_manifest,
)
from apps.autodrive.model import AutoDriveResNet
from apps.autodrive.train import mse_loss, train_epoch


def _image(path, value):
    pixels = np.full((12, 16, 3), value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def _tiny_manifest(tmp_path):
    records = []
    for index in range(6):
        path = tmp_path / f"frame_{index}.png"
        _image(path, 30 + index * 20)
        records.append({
            "image_path": path.name,
            "steering": (index - 2.5) / 5,
            "throttle": 0.2 + index * 0.02,
            "map_name": "synthetic",
            "run_id": "train-run" if index < 4 else "val-run",
            "split": "train" if index < 4 else "val",
        })
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)
    return manifest


def test_batchnorm2d_and_global_pool_forward_backward_and_eval():
    rng = np.random.default_rng(2)
    values = mt.Tensor(
        rng.normal(size=(3, 4, 5, 7)).astype(np.float32), requires_grad=True
    )
    norm = nn.BatchNorm2d(4)
    output = norm(values)
    pooled = nn.AdaptiveAvgPool2d(1)(output)
    assert pooled.shape == (3, 4, 1, 1)
    pooled.sum().backward()
    assert values.grad is not None
    assert norm.weight.grad is not None
    assert not np.allclose(norm.running_var.numpy(), np.ones(4))
    norm.eval()
    assert norm(values).shape == values.shape
    with pytest.raises(NotImplementedError, match="output_size=1"):
        nn.AdaptiveAvgPool2d(2)


def test_legacy_import_creates_grouped_manifest_without_run_leakage(tmp_path):
    circuit = tmp_path / "data_circuit"
    mountain = tmp_path / "data_mountain_track_thro"
    circuit.mkdir()
    mountain.mkdir()
    _image(circuit / "0_0.2500.jpg", 80)
    _image(circuit / "1_-0.1000.jpg", 90)
    _image(mountain / "0_0.5000_0.3000.jpg", 100)
    _image(mountain / "1_-0.2000_0.2500.jpg", 110)
    records = build_legacy_records(
        tmp_path, seed=11, default_throttle=0.2, group_size=1
    )
    assert all(REQUIRED_FIELDS <= set(record) for record in records)
    train_runs = {record["run_id"] for record in records
                  if record["split"] == "train"}
    val_runs = {record["run_id"] for record in records
                if record["split"] == "val"}
    assert train_runs and val_runs and train_runs.isdisjoint(val_runs)
    for map_name in {record["map_name"] for record in records}:
        assert {
            record["split"] for record in records
            if record["map_name"] == map_name
        } == {"train", "val"}
    circuit_records = [record for record in records
                       if record["map_name"] == "circuit"]
    assert {record["throttle"] for record in circuit_records} == {0.2}
    assert {record["label_source"] for record in circuit_records} == {
        "default_throttle"
    }
    manifest = tmp_path / "legacy_manifest.jsonl"
    write_manifest(records, manifest)
    circuit_split = circuit_records[0]["split"]
    split_dataset = AutoDriveDataset(
        manifest, circuit_split, image_size=(8, 10), augment=False
    )
    expected_recorded = any(
        record.get("label_source") != "default_throttle"
        for record in records if record["split"] == circuit_split
    )
    assert split_dataset.has_recorded_throttle is expected_recorded
    circuit_only = AutoDriveDataset(
        manifest, circuit_split, image_size=(8, 10), augment=False,
        maps=["circuit"],
    )
    assert circuit_only.map_names == ["circuit"]


def test_manifest_dataset_preprocess_and_train_only_augmentation(tmp_path):
    manifest = _tiny_manifest(tmp_path)
    train = AutoDriveDataset(
        manifest, "train", image_size=(8, 10), augment=True, seed=5
    )
    validation = AutoDriveDataset(
        manifest, "val", image_size=(8, 10), augment=False, seed=5
    )
    train.set_epoch(3)
    first = train[0]
    repeated = train[0]
    for left, right in zip(first, repeated):
        np.testing.assert_array_equal(left, right)
    assert first[0].shape == (3, 8, 10)
    validation.set_epoch(9)
    before = validation[0]
    validation.set_epoch(10)
    after = validation[0]
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match="only allowed"):
        AutoDriveDataset(manifest, "val", augment=True)


def _training_values(batch=4, device=None):
    rng = np.random.default_rng(4)
    inputs = mt.Tensor(
        rng.normal(size=(batch, 3, 16, 16)).astype(np.float32),
        device=device,
        requires_grad=False,
    )
    steering = mt.Tensor(
        np.zeros((batch, 1), dtype=np.float32), device=device,
        requires_grad=False,
    )
    throttle = mt.Tensor(
        np.full((batch, 1), 0.3, dtype=np.float32), device=device,
        requires_grad=False,
    )
    return inputs, steering, throttle


def _step(model, optimizer, values):
    inputs, steering_target, throttle_target = values
    steering, throttle = model(inputs)
    loss = mse_loss(steering, steering_target) + mse_loss(
        throttle, throttle_target
    )
    result = float(loss.numpy())
    optimizer.reset_grad()
    loss.backward()
    optimizer.step()
    return result


def test_dual_head_shapes_ranges_gradients_and_loss_decrease():
    np.random.seed(4)
    model = AutoDriveResNet(
        base_channels=2, throttle_min=0.1, throttle_max=0.5
    )
    optimizer = mt.optim.Adam(model.parameters(), lr=1e-3)
    values = _training_values()
    steering, throttle = model(values[0])
    assert steering.shape == throttle.shape == (4, 1)
    assert np.all((-1 <= steering.numpy()) & (steering.numpy() <= 1))
    assert np.all((0.1 <= throttle.numpy()) & (throttle.numpy() <= 0.5))
    initial = _step(model, optimizer, values)
    assert all(parameter.grad is not None for parameter in model.parameters())
    for _ in range(4):
        final = _step(model, optimizer, values)
    assert final < initial
    assert model.steering_head.weight.grad is not None
    assert model.throttle_head.weight.grad is not None
    assert model.stem.modules[0].weight.grad is not None


def test_synthetic_manifest_runs_one_training_epoch(tmp_path):
    manifest = _tiny_manifest(tmp_path)
    dataset = AutoDriveDataset(
        manifest, "train", image_size=(8, 10), augment=False
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, seed=3)
    model = AutoDriveResNet(base_channels=2)
    optimizer = mt.optim.Adam(model.parameters(), lr=1e-3)
    metrics = train_epoch(model, loader, optimizer, lambda_throttle=0.5)
    assert metrics["samples"] == 4
    assert metrics["loss"] >= 0
    assert metrics["steer_loss"] >= 0
    assert metrics["throttle_loss"] >= 0


def test_checkpoint_restores_model_optimizer_and_metadata(tmp_path):
    np.random.seed(8)
    model = AutoDriveResNet(base_channels=2, throttle_min=0.1, throttle_max=0.5)
    optimizer = mt.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    values = _training_values()
    _step(model, optimizer, values)
    path = tmp_path / "resume.npz"
    mt.save_checkpoint(
        path, model, optimizer, epoch=3,
        config={"base_channels": 2},
        normalization={"mean": [0.1, 0.2, 0.3], "std": [1, 1, 1]},
    )
    assert mt.inspect_checkpoint(path) == {
        "epoch": 3,
        "config": {"base_channels": 2},
        "normalization": {
            "mean": [0.1, 0.2, 0.3], "std": [1, 1, 1]
        },
        "has_optimizer": True,
    }
    restored = AutoDriveResNet(
        base_channels=2, throttle_min=0.1, throttle_max=0.5
    )
    restored_optimizer = mt.optim.Adam(restored.parameters(), lr=9e-2)
    metadata = mt.load_checkpoint(path, restored, restored_optimizer)
    assert metadata == {
        "epoch": 3,
        "config": {"base_channels": 2},
        "normalization": {
            "mean": [0.1, 0.2, 0.3], "std": [1, 1, 1]
        },
        "has_optimizer": True,
    }
    assert restored_optimizer.t == optimizer.t == 1
    assert restored_optimizer.lr == optimizer.lr
    model.eval()
    restored.eval()
    for expected, actual in zip(model(values[0]), restored(values[0])):
        np.testing.assert_allclose(expected.numpy(), actual.numpy())
    _step(model, optimizer, values)
    _step(restored, restored_optimizer, values)
    for name, expected in model.state_dict().items():
        np.testing.assert_allclose(
            expected.numpy(), restored.state_dict()[name].numpy(),
            rtol=1e-6, atol=1e-7,
        )


@pytest.mark.skipif(not mt.is_cuda_available(), reason="CUDA is unavailable")
def test_autodrive_cuda_forward_backward_optimizer_step():
    device = mt.cuda(0)
    model = AutoDriveResNet(
        base_channels=2, throttle_min=0.1, throttle_max=0.5, device=device
    )
    optimizer = mt.optim.Adam(model.parameters(), lr=1e-3)
    loss = _step(model, optimizer, _training_values(batch=2, device=device))
    device.xp.cuda.get_current_stream().synchronize()
    assert np.isfinite(loss)
    assert model.steering_head.weight.device == device
    assert model.throttle_head.weight.grad.device == device
