# Predictive Learning Algorithm

`train_and_infer_functions.py` — functions: `train_model`, `predictive_learning`

---

## Overview

Predictive learning is the core training loop of NeuraGEM. Each batch involves **two sequential gradient-descent steps**:

1. **Weight Update (WU)**: standard BPTT; update all RNN weights to minimize prediction loss.
2. **Latent Update (LU)**: gradient descent on the latent variable Z only; no weight change.

The model learns *how to use* a latent Z (via weights), and simultaneously *what value Z should take* (via latent optimization). The separation of these two optimization steps is the defining architectural choice.

---

## Training Phases

`train_model()` orchestrates three sequential phases. Which phases run is controlled by boolean config flags.

```
Phase 1: Passive Learning       (add_passive_learning_phase=True)
         ├── WU only (LU disabled)
         └── Duration: passive_phase_length timesteps

Phase 2: Active Learning        (add_blocked_phase=True)
         ├── WU + LU
         └── Duration: blocked_phase_length timesteps

Phase 3: Test / Inference       (run_test_phase=True)
         ├── 3a. Inference + LU (weights frozen; reconfigure_for_prediction() called)
         └── 3b. LU disabled    (pure feedforward)
```

Logger records phase transitions via `logger.log_phase(name)` so figures can annotate them.

---

## Pseudocode: `predictive_learning()`

```python
def predictive_learning(logger, config, dataloader, model, criterion, epochs):

    model.init_Z(batch_size, seq_len)      # Z ← zeros(B, seq_len, Z_dim)
    model.init_hidden(batch_size)           # h ← 10/hidden_size, c ← 0

    for epoch in range(epochs):
        for inputs, llcids, hlcids in dataloader:
            # ── inputs: (B, seq_len, input_size) ──────────────────────────

            # 1. WEIGHT UPDATE
            model.W_optimizer.zero_grad()

            if add_gating:
                combined = combine_input_with_latent(inputs, 'self')  # (B, seq, in+Z)
                outputs, _ = model.forward(combined)
            else:
                outputs, _ = model.forward(inputs, taskID=llcids, what_latent='self')

            loss = criterion(outputs, targets)    # (B, seq, out), MSE unreduced
            loss.sum().backward()
            model.W_optimizer.step()

            # 2. LATENT UPDATE
            if pass_previous_latent:
                model.detach_Z()        # keep Z value, disconnect from prev graph
            else:
                model.reset_Z(B, seq_len)  # Z ← zeros

            if no_of_steps_in_latent_space > 0:
                before_loss = model.update_Z(inputs, criterion, logger, llcids,
                                             no_of_steps=no_of_steps_in_latent_space)

            # 3. LOG
            logger.log_input(inputs)
            logger.log_predicted_output(outputs)
            logger.log_latent_value(model.Z)
            logger.log_gradients_corrections(model.Z.grad)
            logger.log_llcid(llcids)
            logger.log_hlcid(hlcids)
```

---

## Latent Update: `model.update_Z()`

This is the inner loop that optimizes Z for the current batch.

```python
def update_Z(input, criterion, logger, taskID, no_of_latent_steps):

    before_optim_loss = None

    for step in range(no_of_latent_steps):

        model.Z_optimizer.zero_grad()

        if add_gating:
            combined = combine_input_with_latent(input, 'self')
            outputs, _ = forward(combined)
        else:
            outputs, _ = forward(input, taskID=taskID, what_latent='self')

        loss = criterion(outputs, targets)    # (B, seq, out)

        if step == 0:
            before_optim_loss = loss.detach()

        loss.sum().backward()              # ∂L/∂Z has shape (B, seq, Z_dim)

        adjust_Z_grads(latent_aggregation_op)   # ← CRITICAL step (see below)

        model.Z_optimizer.step()           # Z ← Z - lr * adjusted_grad

    return before_optim_loss
```

**Key invariant**: `loss.backward()` propagates through the RNN forward pass and deposits gradients into `Z.grad` with shape `(B, seq_len, Z_dim)`. These gradients are then *modified in-place* by `adjust_Z_grads()` before the optimizer sees them.

---

## Gradient Aggregation: `adjust_Z_grads()`

Because Z is defined over the entire sequence `(B, seq_len, Z_dim)`, each timestep `t` contributes a gradient `∂L_t/∂Z_t`. The aggregation scheme controls how much weight each timestep receives.

### Available Schemes (`latent_aggregation_op`)

#### `'average'`
```python
mean_grad = Z.grad.mean(dim=1, keepdim=True)   # (B, 1, Z_dim)
Z.grad = mean_grad.expand_as(Z.grad)            # broadcast to (B, seq, Z_dim)
```
All timesteps have equal influence on Z. Equivalent to summing the gradient and applying a single update.

#### `'last'`
```python
last_grad = Z.grad[:, -1:, :]                   # (B, 1, Z_dim)
Z.grad = torch.zeros_like(Z.grad)
Z.grad[:, -1:, :] = last_grad
```
Only the final timestep's gradient updates Z. Useful when the context label is only observable at the end of a sequence.

#### `'exponential_increase'`
```python
# Per chunk, element-wise multiply by a precomputed filter
for chunk_idx in range(Z_chunks):
    filt = exponential_increase_filter[chunk_idx]    # (seq_len,) or scalar
    Z.grad[:, :, chunk_slice] *= filt.unsqueeze(0).unsqueeze(-1)
# Then average across time
mean_grad = Z.grad.mean(dim=1, keepdim=True)
Z.grad = mean_grad.expand_as(Z.grad)
```
Recent timesteps are weighted exponentially more than earlier ones. The filter is:
```
y[t] = exp(steepness * t / seq_len)
y = y / y.sum()                   # normalized so total LR is preserved
y = y * multiplier
```
`steepness=0` → uniform (same as `'average'`); high steepness → only last few timesteps matter.

#### `'none'`
Gradients are left untouched. Each timestep's Z gets updated independently — rarely used.

#### `'exponential'` (simple, not chunked)
```python
weights = torch.exp(torch.arange(seq_len).float())
weights = weights / weights.sum()
Z.grad = Z.grad * weights.view(1, seq_len, 1)
# then average and broadcast
```

#### `'mask_last'`
Average all timesteps except the last (useful for ablations).

#### `'mask_all'`
Zero all gradients; effectively disables LU for this batch (used in diagnostics).

---

### Chunk-Specific Learning Rates (`_apply_chunk_lr_and_decay()`)

After aggregation, per-chunk scaling is optionally applied:

```python
for chunk_idx, chunk_slice in enumerate(chunk_slices):
    scale = chunk_lr / base_lr                     # relative LR for this chunk
    Z.grad[:, :, chunk_slice] *= scale
    Z.grad[:, :, chunk_slice] += chunk_l2 * Z[:, :, chunk_slice]  # weight decay
```

This allows different latent chunks to learn at different speeds or have different regularization strengths.

---

## Latent Carryover (`pass_previous_latent`)

When `pass_previous_latent=True` (default):
- After each batch, Z is **not reset**.
- Instead, `model.detach_Z()` disconnects Z from the current computation graph while preserving its numerical value.
- The next batch's LU starts from the previous batch's optimized Z.

This is a key mechanism: Z accumulates information across batches, acting as a slowly-evolving context state. The model does not need to re-infer context from scratch at every batch.

When `pass_previous_latent=False`:
- `model.reset_Z()` is called; Z is re-initialized to zeros.
- Context inference restarts from scratch each batch.

---

## Prediction Alignment 
When `False`:
- The model predicts step `t+1` from steps `0..t`.
- The last input timestep has no prediction target.

---

## Noise Injection (`add_noise_to_input`)

If `config.add_noise_to_input=True`, Gaussian noise `N(0, noise_std)` is added to the input batch before both WU and LU. This is applied at the batch-preparation stage, not inside the model, so the target remains the clean signal.

---

## Input Feed Masking (`input_feed_mask`)

`input_feed_mask` is a list of 0/1 with length `input_size` (or `None` to disable). When set, it is applied to the input tensor **before the model forward pass** — both during WU and LU — by element-wise multiplication along the feature dimension.

```python
# In _prepare_batch_inputs():
input_feed_mask = getattr(config, 'input_feed_mask', None)
if input_feed_mask is not None:
    m = torch.tensor(input_feed_mask, dtype=inputs.dtype, device=inputs.device)
    gated_inputs = gated_inputs * m   # applied after noise injection, before model sees input
```

**Primary use case — augmented-input trick:** embed a supervision signal (e.g., the ground-truth mean) as an extra input dimension. Hide it from the model with `input_feed_mask` and select it as the only loss signal with `output_loss_mask`. This lets you train on a target that is not the next observation without modifying the loss pipeline.

```
input  dim 0: observation  → mask = 1  (model sees this)
input  dim 1: target_mean  → mask = 0  (zeroed out; model cannot cheat)
output dim 0: (unused)     → output_loss_mask = 0
output dim 1: mean estimate→ output_loss_mask = 1  (loss computed here)
```

Note that the **full unmasked input** is still used as the loss target. For `predict_first_frame=False`, the target is `inputs[:, 1:, :]` including both dimensions, and `output_loss_mask` then selects which dimension(s) contribute to the gradient.

---

## Output Loss Masking (`output_loss_mask`)

`output_loss_mask` is a list of 0/1 with length `output_size` (or `None` to use all dims). It is applied **after** the per-element loss is computed and **before** `.backward()`, zeroing out the gradient contribution of unwanted output dimensions.

```python
# In _mask_loss():
t = torch.tensor(mask, dtype=loss.dtype, device=loss.device)
return loss * t
```

Examples:
- `RotatingTargetsConfig`: `[0,0,0,0,0,1,1]` — trains only on the `(x, y)` attack coordinates, suppressing loss on the 5-color one-hot prefix.
- `MeanPredictionConfig`: `[0,1]` — trains only on dim 1 (mean estimate), leaving dim 0 (observation prediction) unconstrained.

---

## Training Phase Details

### Passive Phase
- `_allow_latent_updates = False` → `update_Z()` is skipped entirely.
- WU proceeds normally.
- Useful for pre-training the RNN weights before latent adaptation begins.
- Duration controlled by `passive_phase_length`.

### Active Phase (Blocked Context)
- Both WU and LU run each batch.
- Context blocks are contiguous: all `block_size` batches within a block share the same context mean.
- Duration: `blocked_phase_length` timesteps.

### Test Phase
Called when `run_test_phase=True`.
1. `reconfigure_for_prediction()` adjusts config (e.g., sets evaluation-only flags).
2. Weights are frozen (`W_optimizer` is not stepped).
3. Sub-phase 3a: LU still runs → model can still adapt Z.
4. Sub-phase 3b: `no_of_steps_in_latent_space = 0` → pure feedforward, no adaptation.

---

## Hyperparameters Summary

| Parameter | Default | Effect |
|---|---|---|
| `no_of_steps_in_latent_space` | 1 | Inner LU iterations per batch |
| `no_of_steps_in_weight_space` | 1 | WU iterations per batch |
| `LU_lr` | 0.1 | Z learning rate |
| `WU_lr` | 0.001 | Weight learning rate |
| `latent_aggregation_op` | `'exponential_increase'` | Gradient aggregation scheme |
| `exponential_increase_steepness` | `[2]` | Per-chunk steepness; 0 = uniform |
| `pass_previous_latent` | `True` | Carry Z across batches |
| `loss_reduction_LU` | `'sum'` | How to reduce loss before `.backward()` |
| `l2_loss` | 0 | L2 regularization weight on Z |
| `LU_optimizer` | `'Adam'` | Optimizer for Z |
| `LU_Adam_betas` | `(0.9, 0.999)` | Adam betas for LU |

---

## Common Configurations by Experiment

### Standard NeuraGEM (`experiment_to_run='figure'`)
```python
latent_aggregation_op = 'exponential_increase'
exponential_increase_steepness = [2]
latent_activation = 'softmax'
LU_lr = 0.8
WU_lr = 0.001
use_mul_gating = True
pre_gating = True
pass_previous_latent = True
```

### Ablation: No Latent Updates
```python
no_of_steps_in_latent_space = 0
```
Reduces to a standard RNN with no adaptation.

### Ablation: Average Gradient Aggregation
```python
latent_aggregation_op = 'average'
```
All timesteps contribute equally; recent context not emphasized.

### Ablation: Additive Gating
```python
use_add_gating = True
use_mul_gating = False
```
Z is concatenated to input rather than multiplied into hidden state.
