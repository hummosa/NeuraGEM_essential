# RNN_with_latent — Model Documentation

`models.py` — class: `RNN_with_latent(nn.Module)`

---

## Purpose

`RNN_with_latent` wraps a standard recurrent cell (LSTM, GRU, or Elman RNN) with:
- A learnable **latent variable Z** (an `nn.Parameter`) that is optimized independently from the weights.
- **Multiplicative gating**: Z is projected through a sparse mask to modulate hidden states.
- Two separate optimizers: one for weights (`W_optimizer`), one for Z (`Z_optimizer`).
- Built-in gradient aggregation across the time dimension for Z updates.

---

## Instantiation

```python
model = RNN_with_latent(config)
```

The constructor reads every relevant parameter from `config`. No arguments beyond `config` are needed.

### Architecture Built at `__init__`

```
input_layer:     Linear(input_size, hidden_size)         — shared input projection
recurrent_cell:  LSTMCell / GRUCell / RNNCell             — core dynamics
output_layer:    Linear(hidden_size, output_size)         — prediction head

Z (Parameter):   (batch, seq_len, Z_dim)                  — latent variable
gating_mask1:    (Z_dim, hidden_size) Bernoulli buffer    — pre-gating mask
gating_mask2:    (Z_dim, hidden_size) Bernoulli buffer    — post-gating mask
```

`Z_dim = product(config.latent_dims)`.  The masks are registered as buffers (not parameters; they do not change during training).

---

## Latent Variable Z

### Shape
`(batch_size, seq_len, Z_dim)` — stored as `nn.Parameter` so PyTorch tracks it for optimization.

Z is **sequence-length-dependent**: a separate Z value exists at every timestep. Typically `Z_dim=2`, `seq_len=10`.

### Initialization and Reset

| Method | When to Call | Effect |
|---|---|---|
| `init_Z(batch_size, seq_len)` | Start of training | Allocate Z as zeros; rebuild Z_optimizer |
| `reset_Z(batch_size, seq_len)` | Each batch when `pass_previous_latent=False` | Zero Z in-place (or reallocate if shape changed) |
| `detach_Z()` | Each batch when `pass_previous_latent=True` | Disconnect from computation graph; keep value |
| `set_Z(tensor)` | External control experiments | Copy external tensor into Z |

### Latent Activation
Before Z is used in gating or additive combination, it passes through `latent_activation_function()`:

| `latent_activation` | Operation |
|---|---|
| `'softmax'` | `softmax(Z / temperature, dim=-1)` → sums to 1 |
| `'softmax_chunked'` | Softmax applied independently per chunk |
| `'sigmoid'` | Element-wise sigmoid → (0, 1) |
| `'none'` | Identity |

`softmax` is the default for NeuraGEM experiments; it constrains Z to a probability simplex, which regularizes optimization.

### Latent Chunks
Z can be partitioned into `latent_chunks` sub-vectors, each independently softmaxed. Config: `latent_dims=[dim_per_chunk]`, `latent_chunks=N`. Each chunk has its own steepness, LR, and L2 decay settings.

---

## Multiplicative Gating (Core Mechanism)

When `use_mul_gating=True` (default), Z gates the RNN hidden state element-wise rather than being concatenated with the input.

### Gating Masks

Two static Bernoulli matrices are created at init:
```python
gating_mask1 = Bernoulli(p=P_gates_bernoulli_prob).sample((Z_dim, hidden_size))
gating_mask2 = Bernoulli(p=P_gates_bernoulli_prob).sample((Z_dim, hidden_size))
```
`p=0.3` by default — each Z unit connects to ~30% of hidden units. These masks do not change during training.

### Projection
```python
def _project_gates(latent_slice, stage):
    activated_z = latent_activation_function(latent_slice)  # (B, Z_dim)
    gate_mask = gating_mask1 if stage == 'pre' else gating_mask2
    gate = activated_z @ gate_mask                           # (B, hidden_size)
    return gate
```

### Application
```python
def apply_mul_gating(hidden_state, cell_state, seq_step, stage, what_latent, taskID):
    latent_slice = _get_Z_slice(seq_step, ...)   # (B, Z_dim)
    gate = _project_gates(latent_slice, stage)    # (B, hidden_size)
    hidden_state = hidden_state * gate
    if cell_state is not None:                    # LSTM
        cell_state = cell_state * gate
    return hidden_state, cell_state
```

### Pre- vs Post-Gating
- `pre_gating=True` (default): Gate is applied **before** the RNN step. Z modulates which information the RNN can access on this timestep.
- `post_gating=True`: Gate is applied **after** the RNN step. Z modulates the RNN's output.
- Both can be enabled simultaneously.

---

## Additive Gating (Alternative)

When `use_add_gating=True`, Z is concatenated with the input:
```python
combined = torch.cat([input, activated_Z], dim=-1)   # (B, seq, input_size + Z_dim)
```
The `input_layer` must be sized accordingly (`input_size + Z_dim → hidden_size`). This is not the default; multiplicative gating is the core mechanism.

---

## Forward Pass

```python
outputs, (h, c) = model.forward(input, taskID=None, what_latent='self')
```

**Input**: `(B, seq_len, input_size)` — already on `config.device`.

**Internals** (per timestep `t = 0 .. seq_len-1`):
```python
x_t = input_layer(input[:, t, :])          # (B, hidden_size)

if pre_gating:
    h, c = apply_mul_gating(h, c, t, 'pre', what_latent, taskID)

if rnn_type == 'lstm':
    h, c = lstm_cell(x_t, (h, c))
elif rnn_type == 'gru':
    h = gru_cell(x_t, h)
else:
    h = rnn_cell(x_t, h)

if post_gating:
    h, c = apply_mul_gating(h, c, t, 'post', what_latent, taskID)

out_t = output_layer(h)                    # (B, output_size)
outputs.append(out_t)
```

**Output**: list of `(B, output_size)` tensors, one per timestep. Caller typically stacks them: `torch.stack(outputs, dim=1)` → `(B, seq_len, output_size)`.

---

## Latent Update (Z Optimization)

```python
before_loss = model.update_Z(input, criterion, logger, taskID,
                              no_of_latent_steps=config.no_of_steps_in_latent_space)
```

For each of `no_of_latent_steps` iterations:
1. Zero `Z_optimizer`.
2. Forward pass → compute MSE loss.
3. `loss.backward()` → populate `Z.grad` of shape `(B, seq, Z_dim)`.
4. `adjust_Z_grads(latent_aggregation_op)` — aggregate across time.
5. `Z_optimizer.step()` — update Z.

See [algorithm_predictive_learning.md](algorithm_predictive_learning.md) for details on gradient aggregation.

---

## Exponential Increase Filter

Precomputed at `init_Z()` via `init_exponential_increase_filter()`:

```python
for chunk_idx in range(Z_chunks):
    steepness = config.exponential_increase_steepness[chunk_idx]
    multiplier = config.exponential_increase_multipliers[chunk_idx]

    t = torch.linspace(0, seq_len - 1, seq_len)
    rate = steepness / seq_len
    y = torch.exp(rate * t)
    y = y / y.sum() * multiplier   # normalize; scale by multiplier

    filter_per_chunk[chunk_idx] = y   # (seq_len,)
```

`steepness=0` → all weights equal (same as `'average'`).  
`steepness=40` → weight is concentrated on the last few timesteps.

---

## Optimizers

### Weight Optimizer (`W_optimizer`)
Collects all parameters **except** Z:
```python
params = [p for name, p in model.named_parameters() if 'Z' not in name]
```
Supports `'adam'` and `'sgd'`. Adam is default. `weight_decay` is set from `config.Z_decay`.

### Latent Optimizer (`Z_optimizer`)
Optimizes only `model.Z`:
```python
Z_optimizer = Adam([Z], lr=LU_lr, betas=LU_Adam_betas)
# or
Z_optimizer = SGD([Z], lr=LU_lr, momentum=LU_momentum)
```
Rebuilt whenever Z is re-initialized (shape change clears Adam moments).

---

## What Latent to Use (`what_latent` argument)

The `forward()` and gating functions accept a `what_latent` string that controls what is used as the latent:

| Value | Source |
|---|---|
| `'self'` | Model's own `Z` parameter (standard) |
| `'taskID'` | One-hot encoding of `taskID` (oracle; bypasses Z optimization) |
| `'uniform'` | All-ones normalized vector (constant; ablation) |
| `'zeros'` | Zero vector (no context signal) |
| `'init'` | Same as `'uniform'` |

---

## Hidden State Initialization

```python
hidden_state = torch.full((B, hidden_size), fill_value=10.0 / hidden_size)
cell_state   = torch.zeros((B, hidden_size))   # LSTM only
```

The initial hidden state is set to `10/hidden_size` (not zero). This non-zero initialization was chosen for stability; it ensures the multiplicative gates do not immediately collapse the hidden state on the first timestep.

---

## Deep Copy Support

```python
model_copy = copy.deepcopy(model)
```

`__deepcopy__` is overridden to correctly clone all tensors (parameter data, buffer data) into a new model instance with a fresh config copy. This is used in experiments that compare before/after training states.

---

## Legacy Method Aliases

These aliases exist for backward compatibility with older experiment scripts:

| Alias | Current Method |
|---|---|
| `init_latent()` | `init_Z()` |
| `reset_latent()` | `reset_Z()` |
| `detach_latent()` | `detach_Z()` |
| `set_latent()` | `set_Z()` |
| `update_latent()` | `update_Z()` |
| `adjust_latent_grads()` | `adjust_Z_grads()` |
| `get_WU_optimizer()` | returns `W_optimizer` |
| `get_LU_optimizer()` | returns `Z_optimizer` |

---

## Config Parameters Consumed by This Class

| Config Field | Effect on Model |
|---|---|
| `rnn_type` | Which recurrent cell to build |
| `input_size`, `hidden_size`, `output_size` | Layer dimensions |
| `latent_dims`, `latent_chunks` | Z shape and partitioning |
| `latent_activation`, `softmax_temp` | Z activation function |
| `use_mul_gating`, `pre_gating`, `post_gating` | Gating mode |
| `use_add_gating` | Concatenation mode |
| `P_gates_bernoulli_prob` | Sparsity of gating masks |
| `LU_lr`, `LU_optimizer`, `LU_Adam_betas` | Z optimizer |
| `WU_lr`, `WU_optimizer`, `WU_momentum` | Weight optimizer |
| `l2_loss` | L2 on weights; also per-chunk L2 on Z |
| `latent_aggregation_op` | Gradient aggregation scheme |
| `exponential_increase_steepness` | Per-chunk exponential filter steepness |
| `exponential_increase_multipliers` | Per-chunk filter scale |
| `pass_previous_latent` | Carry Z across batches vs reset |
| `what_latent_to_use` | Default `what_latent` argument |
| `device` | All tensors moved here |
