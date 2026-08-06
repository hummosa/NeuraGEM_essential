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
| State-block | `block_size // (n_colors × 2)` mini-blocks at a fixed rotation |
| `block_size` | `n_miniblocks_per_state_block × n_colors = 50 × 5 = 250` timesteps |

Rotations cycle through `config.train_rotations` across state-blocks. E.g., with `train_rotations=[0.0, 90.0]`, blocks alternate between 0° and 90°.

> **`n_miniblocks_per_state_block` is not the mini-block count.** The config computes `block_size` without the ×2 that two-timesteps-per-trial requires, while the dataset derives the actual count as `block_size // (n_colors × 2)`. At defaults that is 25 mini-blocks, not 50. A run script that wants the field to mean what it says must recompute:
> ```python
> cfg.block_size = cfg.n_miniblocks_per_state_block * cfg.n_colors * 2
> ```
> `run_rotating_targets_comparison.py` does exactly this.

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
| `n_miniblocks_per_state_block` | `50` | Sets `block_size`; the realised mini-block count is half this — see Block Structure |
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
| `hidden_size` | `64` | — |
| `block_size` | `250` | `n_miniblocks_per_state_block × n_colors` |
| `seq_len` | `5` | Half a mini-block; a full one would be `n_colors × 2` |
| `output_loss_mask` | `[0,0,0,0,0,1,1]` | Loss on the `(x, y)` attack coords only, not the color one-hot |
| `latent_dims` | `[2]` | `Z_dim`; must be ≥ `len(train_rotations)` when using oracle Z |
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

### Oracle Z baselines (upper bounds)

`what_latent_to_use='context_ids'` feeds the true rotation in as Z and bypasses Z optimization. The context variable here is the rotation angle, which the dataset emits as `llcid` in radians; `config.oracle_context_encoding` chooses how that value becomes a Z vector. The two encodings answer different questions:

```python
# Identity oracle — one slot per trained rotation
cfg.what_latent_to_use       = 'context_ids'
cfg.oracle_context_encoding  = 'one_hot'
cfg.latent_dims              = [len(cfg.train_rotations)]

# Metric oracle — 0 rad → 0, 360° → 1
cfg.what_latent_to_use       = 'context_ids'
cfg.oracle_context_encoding  = 'normalized'
cfg.latent_dims              = [1]
cfg.latent_activation        = 'none'   # softmax over 1 dim is a constant
```

| Encoding | Z | Carries | Trained rotations `[0, 120, 215]`, novel 45° |
|---|---|---|---|
| `one_hot` | `Z_dim = len(train_rotations)` | Identity only | 45° collapses onto 0°'s slot — no distance between contexts |
| `normalized` | `Z_dim = 1`, `(θ−lo)/(hi−lo)` | Angle magnitude | 45° → 0.125, between 0° (0.0) and 120° (0.333) |
| `circular` | `Z_dim = 2`, `[(1+cos θ)/2, (1+sin θ)/2]` | Angle on a ring | Same, and 359° stays adjacent to 0° |

`one_hot` matches each `llcid` to the nearest entry of `config.oracle_context_values` — a property defaulting to `deg2rad(train_rotations)`, so it tracks a run script's override. `normalized` and `circular` use `config.oracle_context_range`, here `(0, 2π)`. Mismatched `Z_dim` raises at model construction with the value to use.

Two caveats when reading the results:

> **`normalized` is discontinuous at the wrap.** 359° → 0.997 and 0° → 0.0 are maximally far apart in Z despite being the same context. `circular` has no such seam.
>
> **`normalized` puts Z = 0 at θ = 0.** The gate is multiplicative, so a 0° block zeroes the hidden state for its full duration. Prefer `circular`, or exclude 0.0 from `train_rotations`, if 0° blocks look anomalous.

Phase 3 forces `what_latent_to_use='self'` for every condition, so an oracle's Phase-3 advantage lies entirely in the weights it learned under ground-truth labels. That switch only produces inference because `test_no_of_steps_in_latent_space=1` also re-enables LU — an oracle trains with `no_of_steps_in_latent_space=0`, so without it Phase 3 would run with Z pinned at its initial zeros.

Tasks whose `context_ids` are already integer class labels (e.g. flanker slots `0.0`–`4.0`) need no table — with `oracle_context_values=None` the raw id is used directly as the slot index.

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

- [rotation_decoding.md](rotation_decoding.md) — decoding the rotation angle from Z vs. hidden activity
- [rotation_geometry.md](rotation_geometry.md) — what *kind* of rotation code it is (RSA)
- [datasets.md](datasets.md) — `BaseTaskDataset` template, `DATASET_REGISTRY`, and `create_datasets_and_loaders`
- [configs.md](configs.md) — Base `Config` class fields and `_validate()`
- [algorithm_predictive_learning.md](algorithm_predictive_learning.md) — Training loop, Z optimization, latent carryover
