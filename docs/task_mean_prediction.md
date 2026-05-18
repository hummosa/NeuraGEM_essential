# Mean-Prediction Task

## What the task tests

The model observes noisy 1D samples drawn from one of two Gaussian contexts (means 0.2 and 0.8, default σ = 0.4) and must output the **latent mean** of the current context — not the next observation.  This separates context inference from sequence prediction: a model that can track context should quickly lock onto the correct mean after each switch, while a model that only memorizes local statistics will perseverate on the wrong mean.

## Data format and masking

Each timestep is a 2-element vector `[observation, ground_truth_mean]`.  Two config masks route the dimensions through training without letting the model cheat:

| Mask | Value | Effect |
|---|---|---|
| `input_feed_mask` | `[1, 0]` | Zeros out dim 1 before the forward pass — model never sees the true mean as input |
| `output_loss_mask` | `[0, 1]` | Loss is computed only on dim 1 — model must produce the mean in its second output |

Both masks are applied in `train_and_infer_functions.py`.  Code that reads inputs or outputs for **plotting** must also apply these masks (via `_active_dims` in `functions_and_utils.py`) or it will show the wrong dimensions.

## Config: `MeanPredictionConfig`

Inherits from `ContextualSwitchingTaskConfig`.  Key overrides:

```python
input_size      = 2          # [observation, mean]
output_size     = 2
input_feed_mask  = [1, 0]
output_loss_mask = [0, 1]
dataset_name    = 'mean_prediction'
```

Typical training hyperparameters (from `run_mean_prediction.py` and the sweep):

```python
blocked_phase_length = 1000   # timesteps per training phase
default_std          = 0.4    # observation noise
test_block_size      = 40     # timesteps per context block at test time
seq_len              = 3
```

## Sweep design

`mean_prediction_sweep_config.py` sweeps NeuraGEM over `Z_lr ∈ {0.1, 0.2, …, 0.9}`.  Z_decay is coupled to Z_lr via a power-law approximation:

```
Z_decay = 1e-3 × Z_lr²
```

The plain RNN baseline uses `no_of_steps_in_latent_space = 0` (no latent update step).  All conditions are trained with 10 seeds.

## Metric: side-correct rate

At the end of each context block (last `N` timesteps), the predicted mean is classified as correct if it is on the **same side of the midpoint (0.5)** as the true context mean.  This collapses the continuous prediction error into a binary adaptation signal that is robust to scale.

Two derived metrics (from `mean_prediction_analysis.py`):

- **Perseveration errors** — wrong-side predictions before the model reaches a criterion run of `criterion_n` consecutive correct predictions after each switch.
- **Context slips** — wrong-side predictions after criterion is reached in the same block.  `NaN` when criterion is never reached (model never adapted).
