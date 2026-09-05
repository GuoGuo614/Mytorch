import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import mytorch as mt
from apps.autodrive.audit import audit_sources, group_split
from apps.autodrive.collect import ManualControl, collect, count_collected_images
from apps.autodrive.config import CollectionConfig, MAP_ENVS, map_set_slug
from apps.autodrive.gradcam import grad_cam, overlay_heatmap
from apps.autodrive.model import AutoDriveResNet
from apps.autodrive.train import default_checkpoint_path, selected_manifest_maps


def test_dynamic_throttle_and_steering():
    control = ManualControl(CollectionConfig())
    np.testing.assert_allclose(control.update({'w', 'a'}, 1), [-1, .4])
    np.testing.assert_allclose(control.update({'w', 'd'}, 10), [1, .5])
    np.testing.assert_allclose(control.update(set(), 1), [0, .4])
    np.testing.assert_allclose(control.update(set(), 10), [0, .2])
    assert control.update({'space', 'w'}, 1)[1] == 0
    assert control.update(set(), 1)[1] == pytest.approx(.2)
    with pytest.raises(ValueError):
        control.update(set(), float('nan'))
    with pytest.raises(ValueError):
        CollectionConfig(base_throttle=.8, max_throttle=.2)


class Adapter:
    def __init__(self):
        self.index = 0
        self.actions = []
        self.closed = False

    def reset(self):
        self.index += 1
        return np.full((12, 16, 3), self.index, np.uint8)

    def step(self, action):
        self.actions.append(action.copy())
        self.index += 1
        return np.full((12, 16, 3), self.index, np.uint8), 0, len(self.actions) % 2 == 0, {}

    def close(self):
        self.closed = True


class Keys:
    closed = False

    def poll(self, frame):
        return {'w', 'd'}

    def close(self):
        self.closed = True


def test_collection_alignment_metadata_and_group_split(tmp_path):
    adapter, keys = Adapter(), Keys()
    result = collect(adapter, keys, tmp_path, 'warehouse', CollectionConfig(),
                     max_samples=4, realtime=False)
    assert result == {'existing_samples': 0, 'collected_samples': 4,
                      'total_samples': 4, 'target_samples': 4,
                      'stop_reason': 'sample_limit'}
    assert count_collected_images(tmp_path) == 4
    assert adapter.closed and keys.closed
    np.testing.assert_array_equal(adapter.actions[-1], [0, 0])
    sources = sorted(tmp_path.rglob('records.jsonl'))
    assert len(sources) == 2
    assert all(source.parent.name.startswith('warehouse_') for source in sources)
    records, report = audit_sources(sources)
    assert not report['errors'] and not report['duplicates']
    assert report['maps'] == {'warehouse': 4}
    first = min(records, key=lambda r: r['timestamp'])
    assert np.asarray(Image.open(first['image_path']))[0, 0, 0] == 1
    np.testing.assert_allclose([first['steering'], first['throttle']], adapter.actions[0])
    for source in sources:
        metadata = json.loads(source.with_name('metadata.json').read_text())
        assert metadata['control']['throttle_decay'] == .1
        assert metadata['action_alignment'] == 'pre_action_frame'
        assert metadata['target_samples'] == 4
    split = group_split(records)
    assert {r['split'] for r in split} == {'train', 'val'}
    train_runs = {r['run_id'] for r in split if r['split'] == 'train'}
    assert not train_runs & {r['run_id'] for r in split if r['split'] == 'val'}
    assert group_split(records) == split
    with pytest.raises(ValueError, match='two runs'):
        group_split([first])


def test_audit_detects_bad_labels_images_duplicates(tmp_path):
    Image.fromarray(np.ones((3, 4, 3), np.uint8)).save(tmp_path / 'a.png')
    valid = dict(image_path='a.png', steering=.1, throttle=.2, run_id='one', map_name='circuit')
    source = tmp_path / 'records.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in [valid, valid,
        {**valid, 'steering': float('nan')}, {**valid, 'image_path': 'missing.png'},
        {'image_path': 'a.png'}]))
    records, report = audit_sources([source])
    assert len(records) == 2
    assert len(report['duplicates']) == 1
    assert len(report['errors']) == 3


def test_each_map_gets_eighty_twenty_run_split():
    records = [dict(map_name=name, run_id=str(run), frame_index=frame)
               for name in ('generated-track', 'mountain-track', 'warren-track', 'warehouse')
               for run in range(10) for frame in range(3)]
    result = group_split(records)
    for name in {r['map_name'] for r in records}:
        for split, count in [('train', 8), ('val', 2)]:
            assert len({r['run_id'] for r in result
                        if r['map_name'] == name and r['split'] == split}) == count


def test_collection_cleans_up_on_keyboard_failure(tmp_path):
    class BrokenKeys(Keys):
        def poll(self, frame):
            raise RuntimeError('keyboard failure')
    adapter, keys = Adapter(), BrokenKeys()
    with pytest.raises(RuntimeError, match='keyboard failure'):
        collect(adapter, keys, tmp_path, 'warehouse', CollectionConfig(), realtime=False)
    assert adapter.closed and keys.closed
    np.testing.assert_array_equal(adapter.actions[-1], [0, 0])


def test_collection_resumes_until_cumulative_image_limit(tmp_path):
    old_run = tmp_path / 'old-run'
    old_run.mkdir()
    for index in range(3):
        Image.fromarray(np.full((2, 2, 3), index, np.uint8)).save(
            old_run / f'{index:08d}.png'
        )
    adapter, keys = Adapter(), Keys()
    result = collect(adapter, keys, tmp_path, 'warehouse', CollectionConfig(),
                     max_samples=5, realtime=False)
    assert result == {'existing_samples': 3, 'collected_samples': 2,
                      'total_samples': 5, 'target_samples': 5,
                      'stop_reason': 'sample_limit'}
    assert count_collected_images(tmp_path) == 5
    new_metadata = next(path for path in tmp_path.rglob('metadata.json'))
    assert json.loads(new_metadata.read_text())['samples_before_session'] == 3


def test_map_names_determine_default_weight_filename(tmp_path):
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join([
        json.dumps({'map_name': 'mountain-track'}),
        json.dumps({'map_name': 'generated-track'}),
    ]))
    maps = selected_manifest_maps(manifest)
    assert maps == ['generated-track', 'mountain-track']
    assert map_set_slug(maps) == 'generated-track__mountain-track'
    assert default_checkpoint_path(maps).as_posix() == (
        'checkpoints/autodrive_generated-track__mountain-track.npz'
    )
    assert selected_manifest_maps(manifest, ['mountain-track']) == ['mountain-track']
    with pytest.raises(ValueError, match='unknown maps'):
        selected_manifest_maps(manifest, ['warehouse'])


@pytest.mark.parametrize('head', ['steering', 'throttle'])
@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
def test_gradcam_matches_head_gradient_and_restores_state(head, device_name):
    if device_name == 'cuda' and not mt.is_cuda_available():
        pytest.skip('CUDA unavailable')
    device = mt.cpu() if device_name == 'cpu' else mt.cuda(0)
    np.random.seed(3)
    model = AutoDriveResNet(base_channels=2, device=device)
    inputs = mt.Tensor(np.random.default_rng(5).normal(size=(1, 3, 16, 24)).astype('float32'),
                       device=device, requires_grad=True)
    model.eval()
    features = model.forward_features(inputs)
    # For GAP + one linear head, Grad-CAM weights are output activation
    # derivative times linear weights divided by spatial area.
    outputs = model.forward_heads(features)
    selected = 0 if head == 'steering' else 1
    value = float(outputs[selected].numpy().item())
    derivative = 1 - value * value if selected == 0 else value * (1 - value)
    layer = model.steering_head if selected == 0 else model.throttle_head
    weights = layer.weight.numpy().reshape(-1) * derivative / np.prod(features.shape[2:])
    expected = np.maximum((features.numpy()[0] * weights[:, None, None]).sum(0), 0)
    expected /= max(float(expected.max()), 1e-12)
    model.train()
    before = {name: tensor.numpy().copy() for name, tensor in model.state_dict().items()}
    heatmap = grad_cam(model, inputs, head)
    np.testing.assert_allclose(heatmap, expected, atol=1e-5)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, tensor in model.state_dict().items():
        np.testing.assert_array_equal(tensor.numpy(), before[name])
    overlay = overlay_heatmap(np.zeros((45, 80, 3), np.uint8), heatmap)
    assert overlay.shape == (45, 80, 3) and overlay.dtype == np.uint8


def test_optional_simulator_imports_and_cli():
    code = "import sys; import apps.autodrive.collect, apps.autodrive.gradcam; assert all(x not in sys.modules for x in ['gym', 'gym_donkeycar', 'pygame', 'torch'])"
    subprocess.run([sys.executable, '-c', code], check=True)
    for command in ('collect', 'audit', 'drive', 'gradcam'):
        subprocess.run([sys.executable, '-m', 'apps.autodrive', command, '--help'], check=True,
                       capture_output=True)
    assert {'generated-track', 'mountain-track', 'warren-track', 'warehouse'} <= MAP_ENVS.keys()
