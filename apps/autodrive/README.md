# MyTorch V7 AutoDrive

This directory is the new MyTorch implementation. `apps/auto-drive(paddle)` is
kept only as a Paddle reference and its checkpoints/models are not used here.
V7 covers manifest conversion, preprocessing, a dual-head lightweight ResNet,
training, evaluation, resumable checkpoints, and JSON-configured closed-loop
inference. Data collection and Grad-CAM remain later work.

## 1. Create the manifest

The existing `data/DonkeyCar` folders are legacy data whose filenames contain
labels. Convert them once:

```bash
python -m apps.autodrive.manifest \
  --data-root data/DonkeyCar \
  --output data/DonkeyCar/manifest.jsonl \
  --seed 256 --default-throttle 0.2 --group-size 500
```

Every JSONL record contains `image_path`, `steering`, `throttle`, `map_name`,
`run_id`, and `split`. Training and evaluation read only these fields and never
infer labels from filenames. With no `--maps` argument, all detected maps are
included. Each map is split independently, so every map contributes both
training and validation samples.

The old folders flattened several drives together and reused frame numbers, so
their true drive boundaries cannot be recovered. The converter groups each 500
consecutive frame numbers into a conservative pseudo-run and keeps every such
block in only one split. Change the block size with `--group-size`. This avoids
splitting the same numerical neighborhood between train and validation, though
it cannot recreate metadata that the legacy export discarded. `data_circuit`
has no recorded throttle, so its records explicitly use `default_throttle` and
carry `label_source="default_throttle"`.

For newly collected data, write the manifest directly with a distinct `run_id`
per drive; real run boundaries are preferable to legacy pseudo-runs.

## 2. Train and resume

Install the lightweight image dependency with `pip install -e ".[autodrive]"`.
The default preprocessing resizes RGB images to 60x80, normalizes them with the
documented mean/std, and applies deterministic brightness/horizontal-flip
augmentation to training only. A horizontal flip also negates steering.

```bash
python -m apps.autodrive.train --device cpu --epochs 10
python -m apps.autodrive.train --device cuda --epochs 10 --num-workers 2
```

Those commands train one shared model with all maps in the manifest. To train a
map-specific model, select one or more manifest map names explicitly:

```bash
python -m apps.autodrive.train --device cuda --epochs 10 \
  --maps circuit mountain-track
```

The selected map set is saved in the checkpoint configuration and reused by
evaluation unless `--maps` overrides it.

The model has a shared small ResNet backbone and independent steering/throttle
heads. Steering uses `tanh` and is in `[-1, 1]`. Throttle uses a scaled sigmoid
and defaults to `[0, 1]`; change it with `--throttle-min/--throttle-max`.

Each epoch prints JSON containing total, steering, and throttle losses,
learning rate, epoch time, sample counts, and validation MAE. The training loss
is:

```text
steering_mse + lambda_throttle * throttle_mse
```

Checkpoints are atomic, pickle-free compressed NPZ files containing model
parameters/buffers, Adam state (`m`, `v`, step), epoch, configuration, and
normalization statistics. Resume restores all of them and starts at the next
epoch; architecture/range/preprocessing mismatches are rejected:

```bash
python -m apps.autodrive.train \
  --device cuda --epochs 20 --resume \
  --checkpoint checkpoints/autodrive_v7.npz
```

Every completed epoch writes two paired artifacts by default:

- `checkpoints/autodrive_v7.npz`: model weights, BN buffers, optimizer state,
  epoch, training configuration, and normalization;
- `checkpoints/autodrive_v7.json`: checkpoint location, model construction,
  preprocessing, physical control ranges, smoothing, and safety policy.

The JSON checkpoint path is relative to the JSON file, so the pair can be moved
together. Loading rejects mismatched JSON/NPZ architecture or normalization.

## 3. Evaluate

```bash
python -m apps.autodrive.evaluate \
  --device cuda --checkpoint checkpoints/autodrive_v7.npz
```

This reports validation total/sub-losses plus steering and throttle MAE. V7
tests use generated tiny images; no dataset, checkpoint, or benchmark result is
committed to the repository.

## 4. Closed-loop automatic driving

The simulator dependency is isolated behind `GymDonkeyAdapter`; importing the
model, dataset, policy, or tests does not import Gym or launch a simulator.
Install simulator dependencies only on the driving machine:

```bash
pip install -e ".[autodrive,simulator]"
```

Start the DonkeyCar simulator on the selected track, then run:

```bash
python -m apps.autodrive.drive \
  --config checkpoints/autodrive_v7.json \
  --device cuda \
  --env-name donkey-mountain-track-v0 \
  --max-steps 6000
```

For each RGB frame the policy applies the saved resize/normalization, loads the
paired NPZ weights into the dual-head model, predicts steering and throttle,
clips both to configured physical ranges, and applies independent exponential
smoothing. It uses the model's throttle output continuously; there is no
unconditional fixed-throttle driving mode. If the complete training split has
no recorded throttle and uses legacy default labels, training writes
`throttle_mode="fixed"` and driving uses 0.2 instead; datasets with real
throttle labels use `throttle_mode="predicted"`.

Edit the JSON `control` section to choose steering/throttle limits, smoothing,
startup throttle, and failure behavior. `failure_mode="stop"` sends `[0, 0]` on
an inference error. `failure_mode="fixed"` sends the configured safe throttle.
Repeated failures stop the loop, and all exits close the Gym environment.

The adapter accepts both four-value legacy Gym and five-value newer step
results. Closed-loop behavior is covered with a mock simulator, but real track
completion and lap time have not been validated and are not claimed here.
