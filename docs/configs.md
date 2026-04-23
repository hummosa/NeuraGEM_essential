# Configuration Objects

`configs.py` — classes: `Config`, `ContextualSwitchingTaskConfig`, `SeqLearnConfig`

Abbreviations used throughout the codebase: **LU** = Latent Update (Z optimization) | **WU** = Weight Update (standard BPTT)

---

## Class Hierarchy

```
Config  (base — all shared parameters)
├── ContextualSwitchingTaskConfig   (1D Gaussian context switching)
└── SeqLearnConfig                  (Beukers et al. sequence learning)
```

`Config.__init__` sets every parameter that is shared across tasks. Subclasses call `super().__init__()`, set their task-specific fields, then call `self._validate()` at the end to catch misconfigured coupled parameters.

---

## Base Class: `Config`

Parameters are grouped into sections in code order. The most important ones to know are in **Core Training**, **Latent Variable (Z)**, and **Latent Optimizer (LU)**.

### Core Training

| Field | Default | Description |
|---|---|---|
| `env_seed` | 1 | Random seed for data generation and model init |
| `epochs` | 1 | Passes through the dataset per phase |
| `no_of_steps_in_weight_space` | 1 | WU gradient steps per batch |
| `no_of_steps_in_latent_space` | 1 | LU gradient steps per batch |
| `device` | `cuda` if available | All tensors are moved here |

### Latent Variable (Z)

Z has shape `(batch, seq_len, Z_dim)` where `Z_dim = product(latent_dims)`.

| Field | Default | Description |
|---|---|---|
| `latent_dims` | `[2]` | `Z_dim = product(latent_dims)`. Change to `[4]` for a 4-dim Z, or `[2, 2]` for 4-dim split into 2 independently-activated chunks |
| `latent_chunks` | 1 | Number of independently-activated sub-vectors. Must divide `Z_dim`. Each chunk gets its own steepness and multiplier |
| `latent_activation` | `'none'` | Applied to Z before it is used. Options: `'softmax'`, `'sigmoid'`, `'none'` |
| `softmax_temp` | 1 | Temperature for softmax (higher → more uniform) |
| `pass_previous_latent` | `True` | Carry Z across batches (warm-start LU from prior context). Set `False` to reset Z to zeros each batch |
| `what_latent_to_use` | `'self'` | Which latent to use: `'self'` (learn Z), `'taskID'` (oracle one-hot), `'uniform'` (constant), `'zeros'` |

### Latent Optimizer (LU)

| Field | Default | Description |
|---|---|---|
| `LU_lr` | 0.1 | Z learning rate |
| `LU_optimizer` | `'Adam'` | `'Adam'`, `'AdamW'`, or `'SGD'` |
| `LU_Adam_betas` | `(0.9, 0.999)` | Adam beta parameters |
| `LU_momentum` | 0.0 | SGD momentum (only used when `LU_optimizer='SGD'`) |
| `l2_loss` | 0 | L2 regularization weight on Z |
| `loss_reduction_LU` | `'sum'` | How to reduce per-element loss before backward: `'sum'` or `'mean'` |
| `latent_aggregation_op` | `'exponential_increase'` | Gradient aggregation across the time dimension before each LU step. Options: `'exponential_increase'`, `'average'`, `'last'`, `'none'` |
| `exponential_increase_steepness` | `[2] * latent_chunks` | Per-chunk recency bias. `steepness=0` → uniform weight (same as `'average'`); `steepness=2` → mild recency bias; `steepness=40` → focus on last few steps. **Must be one value per chunk.** |
| `exponential_increase_multipliers` | `[1] * latent_chunks` | Per-chunk scale applied after filter normalization |

> **Coupling rule:** `len(exponential_increase_steepness)` and `len(exponential_increase_multipliers)` must both equal `latent_chunks`. `_validate()` raises an `AssertionError` if they don't. Defaults auto-size, so if you set `latent_chunks = 2`, just update these lists to length 2.

### Weight Optimizer (WU)

| Field | Default | Description |
|---|---|---|
| `WU_lr` | 0.001 | Network weight learning rate |
| `WU_optimizer` | `'Adam'` | `'Adam'` or `'SGD'` |
| `WU_momentum` | 0.0 | SGD momentum |
| `loss_reduction_WU` | `'sum'` | `'sum'` or `'mean'` |

### Architecture

| Field | Default | Description |
|---|---|---|
| `rnn_type` | `'lstm'` | `'lstm'`, `'gru'`, or `'rnn'`. LSTM recommended for long sequences |
| `use_mul_gating` | `True` | Multiplicative gating: Z is projected through a sparse mask and element-wise-multiplied into the hidden state |
| `pre_gating` | `True` | Apply gate before each RNN step |
| `post_gating` | `False` | Apply gate after each RNN step. Can be enabled alongside `pre_gating` |
| `use_add_gating` | `False` | Additive gating: concatenate Z to input instead of using multiplicative gating |
| `P_gates_bernoulli_prob` | 0.3 | Fraction of Z→hidden connections enabled in the static random mask |
| `use_input_attention` | `False` | Additional input attention mechanism |

> `post_gating` and `pre_gating` are fully independent booleans. Both can be `True` (gate before and after), both can be `False` (no gating at all).

### Training Phases / Curriculum

| Field | Default | Description |
|---|---|---|
| `add_passive_learning_phase` | `False` | Phase 1: WU only, no LU (pre-trains weights without Z adaptation) |
| `passive_phase_length` | 200 | Timesteps in passive phase |
| `add_interleaved_phase` | `True` | Phase 2: randomly mixed contexts, both WU and LU |
| `interleaved_phase_length` | 500 | Timesteps in interleaved phase |
| `latent_updates_during_shuffle` | `True` | Allow LU during interleaved phase |
| `add_blocked_phase` | `True` | Phase 3: blocked contexts, both WU and LU |
| `blocked_phase_length` | 1000 | Timesteps in blocked phase |
| `shuffle_or_interleave` | `'interleave'` | Task ordering in the interleaved phase |
| `random_transition_shuffle_or_interleave` | `'shuffle'` | Low-level transition ordering |
| `start_always_on_the_same_block` | `False` | Always begin with context A (the minimum mean) |

### Noise Injection

| Field | Default | Description |
|---|---|---|
| `add_noise_to_input` | `False` | Add Gaussian noise to observations before forward pass |
| `noise_std` | 0.0 | Standard deviation of the noise |

### Logging

| Field | Default | Description |
|---|---|---|
| `log_weights` | `False` | Log LSTM gate weight snapshots each batch |
| `log_hidden_states` | `False` | Log RNN hidden states per timestep |
| `log_end_weights` | `False` | Log final weight norms after training |
| `log_initial_burn_in_timesteps` | `False` | Log the first `seq_len` timesteps (normally skipped) |
| `eval_z_space_interval` | 0 | Freeze model and snapshot Z every N batches; `0` to skip |

### I/O

| Field | Default | Description |
|---|---|---|
| `save_model` | `False` | Save model checkpoint after training |
| `load_saved_model` | `False` | Load existing checkpoint instead of training |
| `export_folder` | `'./exports/'` | Root output directory |
| `export_path` | auto | `export_folder/dataset_name/run_name/` — created automatically |

`run_name` is a property: assigning it triggers `update_export_path()` which creates the directory.

### Experimental / Unused in Standard Runs

These are kept for backward compatibility or niche experiments and can be ignored for standard NeuraGEM usage:

| Field | Notes |
|---|---|
| `latent_type` | Only `'1d_latent'` is supported |
| `save_latent_updates` | Not used in standard training |
| `rl_task` | RL experiment flag |
| `use_COIN_channel_experiment` | COIN task variant |
| `add_washout_phase` | COIN memory experiment |

### Methods

`initialize_common_config()` — sets `input_size`, `hidden_size`, `output_size`, `seq_len`, `stride`, `dataset_name` to `None` (subclasses must override these).

`_validate()` — asserts that `exponential_increase_steepness` and `exponential_increase_multipliers` each have length `latent_chunks`, and that `Z_dim % latent_chunks == 0`. Called at end of subclass `__init__`.

`reconfigure_for_prediction(experiment_to_run)` — switches to inference mode: freezes weights (`no_of_steps_in_weight_space = 0`), adjusts `no_of_blocks`. Called by `train_model()` at the start of the test phase.

---

## `ContextualSwitchingTaskConfig(Config)`

**`dataset_name = 'contextual_switching_task'`**

Two Gaussian contexts (A and B) produce scalar observations. The model receives no context label and must infer context from prediction errors, using Z as the context state.

### Task Dimensions

| Field | Default |
|---|---|
| `input_size` | 1 |
| `output_size` | 1 |
| `hidden_size` | 32 |
| `seq_len` | 10 |
| `stride` | 1 |

### Task Data Generation

| Field | Default | Description |
|---|---|---|
| `training_data_means` | `[0.2, 0.8]` | Gaussian means for contexts A and B |
| `default_std` | 0.1 | Observation noise std |
| `task_length` | 1 | Relative difficulty unit (scales `block_size`) |
| `block_duration_distribution` | `'fixed_block_size'` | `'fixed_block_size'` or `'geometric'` |
| `use_high_task_structure` | `False` | Enable a hierarchical second level of context |
| `high_level_latent_change_interval_in_blocks` | 3 | How often the high-level context changes |
| `start_always_on_the_same_block` | `True` | Always start with context A |

### Training Schedule

| Field | Default |
|---|---|
| `epochs` | 3 |
| `batch_size` | 1 |
| `no_of_blocks` | 200 |
| `block_size` | 50 |

### OOD Challenge Block

One block can be swapped for observations from an unseen mean to test generalization:

```python
config.out_of_distribution_challenge = {
    'use_challenge': True,
    'block_no': 15,     # which block index becomes OOD
    'duration': 50,
    'mean': 0.5,        # midpoint between A and B — never seen during training
    'std': 0.2,
}
```

### Experiment Presets

The `experiment_to_run` argument applies a named preset. Presets are implemented as private methods, so each is fully self-contained and readable:

| `experiment_to_run` | Method | Use case |
|---|---|---|
| `'figure'` (default) | `_apply_figure_preset()` | Paper Figure 1 — canonical NeuraGEM |
| `'tweaking'` | `_apply_tweaking_preset()` | Exploration baseline |
| `'weight_grads_comp'` | `_apply_tweaking_preset()` | Gradient comparison |

**`'figure'` preset** (canonical values):

| Parameter | Value |
|---|---|
| `latent_activation` | `'softmax'` |
| `latent_aggregation_op` | `'exponential_increase'` |
| `exponential_increase_steepness` | `[2]` |
| `LU_lr` | `0.8` |
| `WU_lr` | `0.001` |
| `l2_loss` | `0.0001` |
| `LU_Adam_betas` | `(0.6, 0.7)` |
| `blocked_phase_length` | `850` |
| `seq_len` | `4` |
| `block_size` | `25` (= 25 × task_length) |
| `block_duration_distribution` | `'geometric'` |

---

## `SeqLearnConfig(Config)`

**`dataset_name = 'seq_learn'`**

Beukers et al. (2024) blocked sequence-learning task. Two task types, each with two transition patterns, over a 10-state one-hot space.

### Task Dimensions

| Field | Default |
|---|---|
| `input_size` | 10 |
| `output_size` | 10 |
| `hidden_size` | 32 |
| `task_length` | 6 |
| `observation_scale` | 1 |

### Preset

| `experiment_to_run` | Method | Description |
|---|---|---|
| `'few_long_blocks'` (default) | `_apply_few_long_blocks_preset()` | Long blocked training, average gradient aggregation |

**`'few_long_blocks'` key values:** `seq_len=18`, `latent_activation='softmax'`, `latent_aggregation_op='average'`, `pass_previous_latent=False`, `blocked_phase_length=1200`.

---

## Creating and Modifying a Config

```python
# Paper figure preset (canonical NeuraGEM)
config = ContextualSwitchingTaskConfig()

# Override any parameter after construction
config.LU_lr = 0.5
config.hidden_size = 64
config.latent_dims = [4]
config.latent_chunks = 2
config.exponential_increase_steepness = [2, 10]   # one per chunk

# Assigning run_name auto-creates the export directory
config.run_name = 'my_experiment'
# → exports to ./exports/contextual_switching_task/my_experiment/

# Sequence learning
config = SeqLearnConfig()
```

## Export Path

```
export_path = f"{export_folder}/{dataset_name}/{run_name}/"
```

Example: `./exports/contextual_switching_task/my_experiment/`

The directory is created automatically when `run_name` is assigned or `update_export_path()` is called.

---

## Backward-Compatible Aliases

```python
seq_learnConfig = SeqLearnConfig    # old name
CSWConfig       = SeqLearnConfig    # even older name
```
