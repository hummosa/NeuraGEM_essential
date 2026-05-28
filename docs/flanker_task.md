# Flanker Task — Implementation Guide

Paper: *Cortical β Power Reflects a Neural Implementation of Decision Boundary Collapse*, JNeurosci 2024.

---

## Task Structure

Five slots: `[far_left(0), near_left(1), center(2), near_right(3), far_right(4)]`.

**Stage 1**: Target slot rotates across blocks (`block_idx % 5`). One companion slot drawn randomly; companion direction correlated with target per `p_corr_by_distance[|companion − target|]`. ~62% congruent, ~38% incongruent overall.

**Stages 2 & 3**: Target always center (slot 2). All other active slots are flankers.

Each timestep: active slots get `direction × signal_strength + N(0, arrow_noise_std)`, inactive slots get `N(0, bg_noise_std)`.

`p_corr_by_distance = [1.0, 0.80, 0.60, 0.45, 0.30]` (indexed by slot distance; distance 0 unused).

---

## Input / Loss Masking

6-dimensional input: `[obs_slot_0 … obs_slot_4 | true_direction]`.

```python
input_feed_mask  = [1, 1, 1, 1, 1, 0]   # hide true direction from model
output_loss_mask = [0, 0, 0, 0, 0, 1]   # loss only on dim 5
```

`logger.inputs[:, -1]` = unmasked ground-truth direction (always ±1.0).

---

## Speed Pressure

`predict_first_frame=True` → t=0 gets a zero frame; response window starts at t=1.

```
temporal_loss_weights = [0, 1.0, e^-λ, e^-2λ, e^-3λ]   (default λ=0.7)
                      = [0, 1.0, 0.50, 0.25, 0.12]
```

Set `temporal_decay_factor=0` for uniform weights.

---

## DataLoader Output & Oracle Z

DataLoader yields `(data, context_ids, hlcids)`. Meaning varies by stage:

| Field | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| `context_ids` | `float(target_slot)` → oracle Z | `2.0` always (unused in `'self'` mode) | congruency flag (1.0 / 0.0) for block shading |
| `hlcids` | congruency flag (1.0 / 0.0) | congruency flag (1.0 / 0.0) | block type (0.0–3.0) |

`what_latent_to_use = 'context_ids'`: training loop passes `context_ids` as oracle Z; `_get_Z_slice` converts the slot integer to a one-hot of length `Z_dim=5`.  
`what_latent_to_use = 'self'`: model infers Z via LU; `context_ids` is ignored.

---

## Key Config Parameters

| Parameter | Default | Notes |
|---|---|---|
| `arrows_duration` | 5 | Timesteps per trial = `seq_len` = `stride` |
| `trials_per_context_block` | 20 | Trials per block |
| `n_training_contexts` | 500 | Total blocks (Stage 1) |
| `what_latent_to_use` | `'context_ids'` | `'context_ids'` (oracle) or `'self'` (learned) |
| `temporal_decay_factor` | 0.7 | Speed pressure; 0 = uniform |
| `p_corr_by_distance` | `[1,0.8,0.6,0.45,0.3]` | Companion correlation by distance |
| `signal_strength` | 1.0 | Arrow amplitude |
| `arrow_noise_std` | 1.5 | Noise on active slots |
| `latent_dims` | `[5]` | Z_dim=5, one per slot |

---

## Analysis Outputs (`run_flanker.py`)

**Stage 1**  
`flanker_pretrain_results.pdf` — behavior, latent_2d, corrects panels.  
`flanker_accuracy_by_timestep.pdf` — 3-panel: accuracy by training third; congruent vs. incongruent accuracy; RT proxy CDF (late trials, split by congruency).

**Stage 2**  
`flanker_stage2_results.pdf` — latent_2d, corrects panels.  
`flanker_stage2_accuracy_by_timestep.pdf` — accuracy by phase; congruent vs. incongruent; RT proxy CDF.  
`flanker_stage2_rt_correct_vs_wrong.pdf` — RT CDF split by correct/wrong within each congruency.

**Stage 3**  
`flanker_stage3_panels.pdf` — behavior, latent_2d (first-cycle block types annotated), corrects.  
`flanker_stage3_results.pdf` — 2×2 accuracy by timestep + RT proxy CDF for near/far × cong/incong.

Correctness: `(ii[:, -1] * oi[:, -1]) > 0` (sign match between true and predicted direction).

---

## Staged Training

| Stage | Config class | `what_latent_to_use` | Goal |
|---|---|---|---|
| 1 | `FlankerTaskConfig` | `'context_ids'` | Pretrain RNN with oracle slot identity; target rotates across blocks so model learns all 5 Z patterns |
| 2 | `FlankerTaskStage2Config` | `'self'` | Frozen weights; Z self-organizes on alternating congruent/incongruent blocks. Expected: Z flat for congruent, peaked on center dim for incongruent |
| 3 | `FlankerTaskStage3Config` | `'self'` | Frozen weights; 4 block types cross distance × congruency; key prediction is near flanker effect > far flanker effect |
| 4 | — | — | RT proxy via confidence output neuron or REINFORCE stop signal (see paper) |

### Stage 3 dataset details

`FlankerTaskStage3Dataset` — block type cycles `block_idx % 4`, crossing distance × congruency.

| Block | Flanker slots | Flanker direction | Label |
|---|---|---|---|
| 0 | 1, 3 (near) | = target | near-congruent |
| 1 | 1, 3 (near) | ≠ target | near-incongruent |
| 2 | 0, 4 (far) | = target | far-congruent |
| 3 | 0, 4 (far) | ≠ target | far-incongruent |

`hlcids` encodes block type (0.0–3.0). The `latent_2d` panel labels the first full cycle.

### Stage 3 analysis note

**Do not collapse across congruency when comparing near vs. far.** Report as a 2×2:

- Near flanker effect = (near-cong accuracy) − (near-incong accuracy)
- Far flanker effect = (far-cong accuracy) − (far-incong accuracy)
- **Key prediction**: near flanker effect > far flanker effect (interaction), because Stage 1 assigned higher correlation weight to nearby slots (`p_corr_by_distance`), so the model learned to weight near-slot signals more heavily.

Same logic applies to RT proxy: near-incongruent should produce the slowest decisions, far-congruent the fastest.

Z dynamics: incongruent blocks should drive Z toward a sharper center-slot peak; near-incongruent should converge faster than far-incongruent because near-slot prediction error is larger.

---

## Critical: `model.config` vs stage config

The model stores its own config snapshot (`model.config`) set at construction time.
**Updating a stage config object does NOT update `model.config`.**

Parameters the model reads at forward / LU time (e.g. `Z_lr`, `latent_dims`,
`no_of_steps_in_latent_space`, gating flags) must be mirrored explicitly:

```python
stage2_config.Z_lr   = 0.4
model.config.Z_lr    = 0.4   # required — model reads Z_lr from model.config during LU
```

Parameters consumed only by `train_model`'s outer loop (`blocked_phase_length`,
`dataset_name`, `no_of_steps_in_weight_space`, etc.) are read from the stage config
and do **not** need to be mirrored. When in doubt, patch both.
