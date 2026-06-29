# Flanker Task — Implementation & Analysis Guide

Paper reference: *Cortical β Power Reflects a Neural Implementation of Decision Boundary Collapse*, JNeurosci 2024.

---

## File Map

| File | Role |
|---|---|
| `configs.py` | Config classes for each stage |
| `datasets.py` | Dataset classes + `DATASET_REGISTRY` |
| `run_flanker.py` | Main experiment script (`#%%` cells, one per stage/analysis) |
| `flanker_analyses.py` | Reusable trial extraction, selection, plotting, and helper utilities |
| `train_and_infer_functions.py` | `train_model()` training loop |
| `functions_and_utils.py` | `plot_logger_panels()`, `Logger` class |

---

## Task Structure

Five slots: `[far_left(0), near_left(1), center(2), near_right(3), far_right(4)]`.

**Stage 1**: Target slot rotates across blocks (`block_idx % 5`). One companion slot drawn randomly per trial; companion direction correlated with target with probability `p_corr_by_distance[|companion − target|]`.

**Stages 2–4**: Target always center (slot 2). All other active slots are flankers.

Each timestep: active slots receive `direction × signal_strength + N(0, arrow_noise_std)`, inactive slots receive `N(0, bg_noise_std)`.

---

## Input / Loss Masking

6-dimensional input: `[obs_slot_0 … obs_slot_4 | true_direction]`.

```python
input_feed_mask  = [1, 1, 1, 1, 1, 0]   # model never sees dim 5
output_loss_mask = [0, 0, 0, 0, 0, 1]   # loss only on dim 5 (direction)
```

`logger.inputs[:, -1]` = unmasked ground-truth direction (always ±1.0).

---

## Speed Pressure

`predict_first_frame=True` → t=0 gets a zero frame; response window starts at t=1.

```
temporal_loss_weights = [0, 1.0, e^-λ, e^-2λ, e^-3λ]   (λ=0.7 default)
                      = [0, 1.0, 0.50, 0.25, 0.12]
```

`temporal_decay_factor=0` → uniform weights.

---

## DataLoader Output & Oracle Z

DataLoader yields `(data, context_ids, hlcids)`. Meaning varies by stage:

| Field | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| `context_ids` | `float(target_slot)` → oracle Z | `2.0` always (unused in `'self'` mode) | congruency flag (1.0/0.0) for block shading | congruency flag (1.0/0.0) for block shading |
| `hlcids` | congruency flag (1.0/0.0) | congruency flag (1.0/0.0) | block type (0.0–3.0) | trial type (0.0–3.0) |

`what_latent_to_use = 'context_ids'`: training loop passes `context_ids` as oracle Z; `_get_Z_slice` converts the slot integer to a one-hot of length `Z_dim=5`.  
`what_latent_to_use = 'self'`: model infers Z via LU; `context_ids` is ignored.

---

## Key Config Parameters (`FlankerTaskConfig` defaults)

| Parameter | Value | Notes |
|---|---|---|
| `arrows_duration` | 5 | Timesteps per trial = `seq_len` = `stride` |
| `trials_per_context_block` | 2 | Trials per block in Stage 1 |
| `signal_strength` | 1.4 | Arrow amplitude |
| `arrow_noise_std` | 1.0 | Noise on active slots |
| `bg_noise_std` | 0.1 | Noise on inactive slots |
| `p_corr_by_distance` | `[1.0, 0.65, 0.55, 0.25, 0.1]` | Companion correlation by slot distance |
| `temporal_decay_factor` | 0.7 | Speed pressure; 0 = uniform |
| `what_latent_to_use` | `'context_ids'` | `'context_ids'` (oracle) or `'self'` (learned) |
| `latent_dims` | `[5]` | Z_dim=5, one per slot |
| `Z_lr` | 0.3 | Latent update learning rate |
| `hidden_size` | 64 | RNN hidden units (set in `run_flanker.py`) |

---

## Staged Training

| Stage | Config class | `what_latent_to_use` | Goal |
|---|---|---|---|
| 1 | `FlankerTaskConfig` | `'context_ids'` | Pretrain RNN with oracle slot identity; target rotates across blocks so model learns all 5 Z patterns |
| 2 | `FlankerTaskStage2Config` | `'self'` | Frozen weights; Z self-organizes on alternating congruent/incongruent full-flanker blocks |
| 3 | `FlankerTaskStage3Config` | `'self'` | Frozen weights; 4 block types cross distance × congruency (near/far × cong/incong) |
| 4 | `FlankerTaskStage4Config` | `'self'` | Frozen weights; fully randomized trials; sequential history and post-error analyses |

### Stage 3 block types (`hlcids` encodes 0–3)

| Block | Flanker slots | Direction | Label |
|---|---|---|---|
| 0 | 1, 3 (near) | = target | near-congruent |
| 1 | 1, 3 (near) | ≠ target | near-incongruent |
| 2 | 0, 4 (far) | = target | far-congruent |
| 3 | 0, 4 (far) | ≠ target | far-incongruent |

**Key Stage 3 prediction**: near flanker effect (near-cong − near-incong accuracy) > far flanker effect. Reason: `p_corr_by_distance` gave near slots higher weight during Stage 1 training, so the model weights near-slot signals more heavily.

### Stage 4 trial types (same coding as Stage 3, drawn i.i.d. each trial)

Same 0–3 encoding. Trials are fully randomized (no blocks); `block_size = arrows_duration = 5` so each DataLoader batch is exactly 1 trial.

---

## `train_model()` Signature

```python
logger, model, config, figs = train_model(
    config,
    seed=0,
    save_models=False,
    load_models=False,
    run_test_phase=True,      # False to skip Phase 3 (test); use for Stage 3+
    pretrained_model=None,    # pass a trained model to skip construction/loading
)
```

**Phases run:**
- **Phase 1** (passive): WU only, no LU. Skipped if `config.add_passive_learning_phase=False` (default).
- **Phase 2** (active): WU + LU for `blocked_phase_length` timesteps. Always runs.
- **Phase 3** (test): inference-only + feedforward baseline. Skipped when `run_test_phase=False`.

---

## Pattern: Passing a Model Between Stages

Always do these three things when creating a new stage config:

```python
from configs import FlankerTaskStageNConfig
from flanker_analyses import sync_gating, mirror_to_model

stageN_config = FlankerTaskStageNConfig(experiment_to_run='default')
stageN_config.run_name = 'flanker_stageN_v1'
stageN_config.Z_lr = 0.4
stageN_config.no_of_steps_in_latent_space = 1

sync_gating(stageN_config, config)         # copy pre/post/add/mul gating from Stage 1 config
mirror_to_model(model_prev, stageN_config) # patch model.config.Z_lr and no_of_steps_in_latent_space

model_prev.set_Z(torch.randn_like(model_prev.Z) * 0.2)  # reset Z before new stage

loggerN, modelN, stageN_config, figs = train_model(
    stageN_config, seed=stageN_config.env_seed,
    save_models=False, load_models=False,
    pretrained_model=model_prev,
    run_test_phase=False,
)
```

---

## Adding a New Stage

1. **`datasets.py`** — subclass `BaseTaskDataset`, implement `generate_sequences()`, register:
   ```python
   class FlankerTaskStageNDataset(BaseTaskDataset):
       def generate_sequences(self):
           ...   # returns (data_seq, context_ids_seq, hlcid_seq)
   DATASET_REGISTRY['flanker_stageN'] = FlankerTaskStageNDataset
   ```

2. **`configs.py`** — subclass the nearest parent, override what changes:
   ```python
   class FlankerTaskStageNConfig(FlankerTaskStage2Config):
       def __init__(self, experiment_to_run='default'):
           super().__init__(experiment_to_run)
           self.dataset_name = 'flanker_stageN'
           self.n_training_contexts = ...
           self.no_of_blocks = self.n_training_contexts
           self.blocked_phase_length = self.no_of_blocks * self.block_size
           self.update_export_path()
           self._validate()
   ```

3. **`run_flanker.py`** — add `#%%` cells for training and analysis following the stage-passing pattern above.

---

## `flanker_analyses.py` — Full API

Import at top of script:
```python
from flanker_analyses import (
    extract_trials, select_trials, build_history_groups,
    plot_accuracy_by_timestep, plot_rt_distribution, plot_z_by_timestep,
    plot_scalar_bars, sync_gating, mirror_to_model,
)
```

### `extract_trials(logger, config, rt_threshold=0.5)`

Unpacks a logger into a per-trial dict. Call once after `train_model`.

```python
trials = extract_trials(logger, config)
```

**Returned dict keys:**

| Key | Shape | Description |
|---|---|---|
| `correct` | `(n, ad)` | 1.0 if sign-correct at each timestep |
| `output_traj` | `(n, ad)` | raw model output (dim 5) per timestep |
| `z_traj` | `(n, ad, Z_dim)` | Z values per timestep |
| `trial_type` | `(n,)` | hlcids value per trial (float); meaning is stage-dependent |
| `context_id` | `(n,)` | context_ids value per trial (float) |
| `true_dir` | `(n,)` | true direction ±1.0 |
| `is_correct` | `(n,)` | bool — correct at final timestep |
| `response_side` | `(n,)` | sign of output at final timestep (+1/−1/0) |
| `rt` | `(n,)` | first timestep where `\|output\| > threshold`; `ad` if never |
| `rt_threshold` | float | threshold used |
| `ad` | int | `arrows_duration` |
| `n_trials` | int | total trials |

---

### `select_trials(trials, trial_type=None, is_correct=None, response_side=None)`

Returns a boolean mask `(n_trials,)`.

```python
incong_mask  = select_trials(trials, trial_type=[1, 3])         # near- and far-incongruent
error_mask   = select_trials(trials, is_correct=False)
right_trials = select_trials(trials, response_side=1.0)
```

`trial_type` matches against `trials['trial_type']` (hlcids). Values depend on stage:
- Stage 1/2: `1.0` = congruent, `0.0` = incongruent
- Stage 3/4: `0` = near-cong, `1` = near-incong, `2` = far-cong, `3` = far-incong

---

### `build_history_groups(trials, n_back=2, current_mask=None, congruent_types=(0, 2))`

Groups trials by the congruency of the preceding `n_back` trials. Returns a dict of boolean masks keyed by history tuples `(bool, bool)` ordered oldest-to-most-recent.

```python
incong_mask = select_trials(trials, trial_type=[1, 3])
groups = build_history_groups(trials, n_back=2, current_mask=incong_mask)
# groups[(True, True)]  — CC→I: both preceding trials congruent (least adapted)
# groups[(False, False)] — II→I: both preceding incongruent (most adapted)
```

`congruent_types`: trial_type values treated as congruent. Default `(0, 2)` for Stage 3/4.  
For Stage 1/2 where `trial_type` is the congruency flag directly, use `congruent_types=(1,)`.

---

### Plot functions

All plot functions take `specs = [(mask, label, color), ...]` and an optional `linestyles` list.

```python
plot_accuracy_by_timestep(ax, trials, specs, config, linestyles=None)
    # Mean accuracy per timestep. Draws axhline(0.5) and axvspan for pre-response region.

plot_rt_distribution(ax, trials, specs, config, fit_gaussian=True, linestyles=None,
                     undecided='extra_bin')
    # Empirical PMF (markers + connecting line) + optional Gaussian fit (dashed).
    # PMF always sums to 1.0 — undecided trials are never silently dropped.
    # undecided='extra_bin' : adds a labelled 'und.' bin at t=ad (default).
    # undecided='last_bin'  : stacks undecided trials into the t=ad-1 bin.

plot_z_by_timestep(ax, trials, specs, z_dim, config, linestyles=None)
    # Mean Z[z_dim] per timestep. Use z_dim=2 for center slot.

plot_scalar_bars(ax, trials, specs, measure, group_spacing=None)
    # Bar chart mean ± SEM. measure options:
    #   'accuracy'       — correctness at RT-crossing timestep
    #   'rt'             — RT in timesteps (arrows_duration for undecided)
    #   int              — Z dim value at RT-crossing timestep
    #   ('z_end', int)   — Z dim value at final timestep
    # group_spacing=[2,4] inserts gaps before bar indices 2 and 4.
    # Returns x_positions array for further annotation.
```

**Common spec construction pattern:**

```python
conditions = [
    (0.0, 'near-cong',   '#4393c3', '-'),
    (1.0, 'near-incong', '#4393c3', '--'),
    (2.0, 'far-cong',    '#d6604d', '-'),
    (3.0, 'far-incong',  '#d6604d', '--'),
]
specs = [(select_trials(trials, trial_type=bt), lbl, c) for bt, lbl, c, _ in conditions]
ls    = [ls for _, _, _, ls in conditions]
plot_accuracy_by_timestep(ax, trials, specs, config, linestyles=ls)
```

---

### Config / model helpers

```python
sync_gating(stage_config, from_config)
    # Copies pre_gating, post_gating, use_add_gating, use_mul_gating from from_config.
    # Required because stage configs are freshly constructed with default values.

mirror_to_model(model, stage_config, attrs=('Z_lr', 'no_of_steps_in_latent_space'))
    # Patches model.config attributes so the LU loop reads the new values.
    # Required because model.config is a direct reference to Stage 1's config object.
```

---

## Post-Error / Sequential Analysis Pattern (Stage 4)

Use `np.roll` for one-trial lookback:

```python
prev_is_error   = np.roll(~trials4['is_correct'], 1);  prev_is_error[0]   = False
prev_trial_type = np.roll(trials4['trial_type'],  1);  prev_trial_type[0] = -1
prev_correct    = np.roll(trials4['is_correct'],  1);  prev_correct[0]    = False

near_err = prev_is_error & np.isin(prev_trial_type, [0, 1])
far_err  = prev_is_error & np.isin(prev_trial_type, [2, 3])
```

Always set index 0 to False to prevent end-of-array wraparound contamination.

---

## Analysis Outputs (`run_flanker.py`)

**Stage 1**  
`flanker_pretrain_results.pdf` — behavior, latent_2d, corrects panels.  
`flanker_accuracy_by_timestep.pdf` — accuracy by training third; congruent vs. incongruent; RT distribution (late trials).

**Stage 2**  
`flanker_stage2_results.pdf` — latent_2d, corrects panels.  
`flanker_stage2_accuracy_by_timestep.pdf` — accuracy by phase; congruent vs. incongruent; RT distribution.  
`flanker_stage2_rt_correct_vs_wrong.pdf` — RT by correct/wrong within each congruency.

**Stage 3**  
`flanker_stage3_panels.pdf` — behavior, latent_2d (first-cycle block types annotated), corrects.  
`flanker_stage3_results.pdf` — 2×2 accuracy + RT distribution for near/far × cong/incong.

**Stage 4**  
`flanker_stage4_history_effects.pdf` — sequential congruency effect: accuracy, RT, Z center-dim for history groups (CC→I, IC→I, CI→I, II→I).  
`flanker_stage4_post_error.pdf` — post-error line plots: accuracy, RT, Z by trial-A type (near-error / far-error / correct).  
`flanker_stage4_post_error_bars.pdf` — same 3 conditions as mean ± SEM bar charts.  
`flanker_stage4_post_error_diagnostic.pdf` — post-error × trial-B congruency crossover diagnostic (6-condition 2×3 bar plot).

Correctness definition: `(inputs[:, -1] * outputs[:, -1]) > 0` (sign match, dim 5).

---

## Critical Gotchas

**1. `model.config` is a direct reference, not a copy.**  
When Stage N's `train_model` receives `pretrained_model=model`, `model.config` still points to Stage 1's config object. Parameters the model reads at LU time (`Z_lr`, `no_of_steps_in_latent_space`) must be explicitly patched:
```python
mirror_to_model(model, stageN_config)   # always call this
```

**2. Stage configs don't inherit runtime gating changes.**  
`FlankerTaskStage2Config()` is freshly constructed with base-class defaults. User's runtime choice (`gating = 'post'` etc. set on `config`) is NOT propagated automatically:
```python
sync_gating(stageN_config, config)      # always call this
```

**3. `trial_type` field meaning is stage-dependent.**  
`extract_trials` puts `hlcids` into `trials['trial_type']`. In Stages 1/2 hlcids = congruency (0/1). In Stages 3/4 hlcids = block/trial type (0–3). Pass appropriate `congruent_types` to `build_history_groups`.

**4. DataLoader uses `shuffle=False`.**  
Trial order is preserved exactly as generated. This is intentional and required for the sequential history analysis. Stage 4 trials are i.i.d. by construction (not by shuffling).

**5. `run_test_phase=False` for Stages 3+.**  
Stages 3 and 4 only need Phase 2 (blocked learning). Passing `run_test_phase=False` skips Phase 3 and keeps the logger cleaner for analysis.

**6. `blocked_phase_length` overrides `n_training_contexts`.**  
`train_model` recalculates `no_of_blocks = int(blocked_phase_length / block_size)` at runtime. Setting `blocked_phase_length` in `run_flanker.py` after config construction takes precedence. Always update `blocked_phase_length` whenever you change `block_size`.

**7. `model.set_Z(torch.randn_like(model.Z) * 0.2)` before each new stage.**  
Stage 1's oracle-guided Z is not a suitable starting point for self-organization in later stages. Resetting ensures LU starts fresh.
