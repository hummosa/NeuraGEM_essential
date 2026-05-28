# Rotating Targets Task

Implementation of the predictive-inference paradigm from Yu et al. (2025), "A parietal grid-like code rotates with cognitive maps but lags rapid behavioral transfer."

---

## Task Description

Five shield colors each have a fixed target location on a circular arena. An alien attacks the location of the active shield color; the model must predict where the attack will land. After several mini-blocks (a *state-block*), all target locations rotate simultaneously by a fixed angle — without any explicit signal. The key behavioral feature is **zero-shot transfer**: once the model observes the new location for one color, it should infer all others' new locations instantly, because the relational structure (evenly-spaced targets on a circle) is preserved under rotation.

---

## Data Representation

Each paper-trial (one color shown, one attack position observed) maps to **two sequential timesteps** using `input_size = n_colors + 2 = 7`:

```
t   (cue):     [color_onehot (5 dims),  0.0,      0.0     ]
t+1 (outcome): [0.0, 0.0, 0.0, 0.0, 0.0,  attack_x, attack_y]
```

With frame prediction (`predict_first_frame=False`), the model at step t must predict step t+1. At cue timesteps, this means the model must output the attack position given the shield color — the primary learning signal. At outcome timesteps, the model predicts the next color (random permutation order; adds irreducible noise to the loss).

### Target geometry

Base targets at 0° rotation — `n_colors` points evenly spaced on a unit circle of radius `target_radius`:

```python
angle_c = (2π / n_colors) * c      # c = 0, 1, ..., n_colors-1
base_target_c = target_radius * [cos(angle_c), sin(angle_c)]
```

At rotation θ (radians), all targets are rotated by the same 2×2 rotation matrix:

```python
R = [[cos θ, -sin θ],
     [sin θ,  cos θ]]
rotated_targets = (R @ base_targets.T).T
```

Each attack is sampled independently as:
```
attack_c = rotated_target_c + noise_std * N(0, I₂)
```

---

## Block Structure

| Concept | Value (defaults) |
|---|---|
| Mini-block | One random permutation of all `n_colors` colors; each color → 2 timesteps |
| State-block | `n_miniblocks_per_state_block` mini-blocks at a fixed rotation |
| `block_size` | `n_miniblocks_per_state_block × n_colors × 2 = 8 × 5 × 2 = 80` timesteps |

Rotations cycle through `config.train_rotations` across state-blocks. E.g., with `train_rotations=[0.0, 90.0]`, blocks alternate between 0° and 90°.

### Context IDs

| ID | Value |
|---|---|
| `llcid[t]` | Rotation angle in **radians** (e.g., 0.0 or π/2). Continuous float; used directly as oracle Z. |
| `hlcid[t]` | Block index (float). |

---

## Train / Test Rotation Split

Two dataset classes handle the split:

| Class | Registered as | Rotations used |
|---|---|---|
| `RotatingTargetsDataset` | `'rotating_targets'` | `config.train_rotations` |
| `RotatingTargetsTestDataset` | `'rotating_targets_test'` | `config.test_rotations` (falls back to `train_rotations` if empty) |

`create_datasets_and_loaders` checks for `config.test_dataset_name` and instantiates train and test sets from their respective registry keys. All existing tasks are unaffected by this addition.

---

## Configuration — `RotatingTargetsConfig`

```python
from configs import RotatingTargetsConfig
cfg = RotatingTargetsConfig()
```

### Task-specific fields

| Field | Default | Description |
|---|---|---|
| `n_colors` | `5` | Number of shield colors |
| `n_miniblocks_per_state_block` | `8` | Mini-blocks per state-block |
| `noise_std` | `0.04` | Std of attack Gaussian (units of target radius) |
| `target_radius` | `0.5` | Circle radius for base targets |
| `train_rotations` | `[0.0, 90.0]` | Rotation angles (°) cycling across training blocks |
| `test_rotations` | `[]` | Novel rotation angles (°) for transfer test; empty → uses train_rotations |
| `dataset_name` | `'rotating_targets'` | Train dataset key |
| `test_dataset_name` | `'rotating_targets_test'` | Test dataset key |

### Derived / architecture fields

| Field | Default | Derivation |
|---|---|---|
| `input_size` | `7` | `n_colors + 2` |
| `output_size` | `7` | `n_colors + 2` |
| `block_size` | `80` | `n_miniblocks × n_colors × 2` |
| `seq_len` | `10` | `n_colors × 2` (one full mini-block) |
| `latent_dims` | `[1]` | Scalar Z; matches continuous radian llcid |
| `predict_first_frame` | `False` | Cue → attack frame prediction |
| `pass_previous_latent` | `True` | Carry Z across batches |

---

## Usage

### Basic training

```python
from configs import RotatingTargetsConfig
from train_and_infer_functions import train_model

cfg = RotatingTargetsConfig()
cfg.train_rotations = [0.0, 90.0]
cfg.test_rotations  = [45.0, 135.0, 225.0, 315.0]

logger, model, cfg, figs = train_model(cfg, seed=0)
```

### Oracle Z baseline (upper bound)

Pass the true rotation angle directly as Z, bypassing Z optimization:

```python
cfg.what_latent_to_use = 'context_ids'   # llcid (radians) used as oracle Z
```

Since `latent_dims=[1]` and `llcid` is a continuous radian angle, the oracle Z is a direct numeric representation of the context — a natural upper bound for learned Z.

### Zero-shot transfer evaluation

```python
cfg.test_rotations = [45.0, 135.0, 225.0, 315.0]
cfg.no_of_steps_in_weight_space = 0   # freeze weights
logger_test, _, _, _ = train_model(cfg, seed=0)
```

After the first observation of any color under the new rotation, Z should update to represent the new context and prediction error on subsequent colors should drop.

---

## Expected Behavior

- **During training**: loss decreases; Z clusters at two distinct values corresponding to 0° and 90°.
- **At state-block boundaries**: prediction error spikes on the first few cue → attack trials (the model hasn't updated Z yet), then drops rapidly.
- **Zero-shot transfer test**: after observing one color's new location, prediction error on all other colors should drop substantially below the random baseline, even for rotation angles never seen during training.

---

## See Also

- [datasets.md](datasets.md) — `BaseTaskDataset` template, `DATASET_REGISTRY`, and `create_datasets_and_loaders`
- [configs.md](configs.md) — Base `Config` class fields and `_validate()`
- [algorithm_predictive_learning.md](algorithm_predictive_learning.md) — Training loop, Z optimization, latent carryover
