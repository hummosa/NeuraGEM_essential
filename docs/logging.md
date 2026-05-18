# Logging and Visualization

`functions_and_utils.py` — classes: `Logger`; functions: `plot_logger_panels`, `plot_corrects_seq_learn`, analysis helpers

---

## `Logger` Class

The Logger is the central data store for everything that happens during training. It records inputs, outputs, latent values, gradients, phase transitions, and model internals at every timestep. After training, the logger is passed directly to plotting functions.

### Design Invariant

All logged data is stored as `List[np.ndarray]`. Each call to a `log_*` method appends one batch worth of data. After training, these lists are concatenated into `(total_timesteps, dim)` arrays at plot time.

Shape convention per entry: `(batch_size, stride, var_dim)` — *except* the first batch, which may be `(batch_size, seq_len, var_dim)` if `log_initial_burn_in_timesteps=True`.

---

### Logged Quantities

#### Training Input / Output

| Attribute | Shape per Entry | Description |
|---|---|---|
| `inputs` | `(B, stride, input_size)` | Raw observations sent to model |
| `training_batches` | `(B, stride, input_size + Z_dim)` | Combined input+latent (if add_gating) |
| `predicted_outputs` | `(B, stride, output_size)` | Model predictions after WU |
| `prediction_losses` | `(B, stride, output_size)` | Per-element MSE loss |
| `training_losses` | `(B, stride, output_size)` | Loss after both WU and LU |
| `training_losses_before_latent_optimization` | `(B, stride, output_size)` | Loss before LU (= loss after WU only) |
| `testing_batches`, `testing_losses` | same | Equivalent quantities on test set |

#### Latent Variable (Z)

| Attribute | Shape per Entry | Description |
|---|---|---|
| `latent_values` | `(B, stride, Z_dim)` | Z after LU optimization |
| `latent_gradients` | `(B, stride, Z_dim)` | Raw ∂L/∂Z before aggregation |
| `gradients_corrections` | `(B, stride, Z_dim)` | `model.Z.grad` after aggregation |
| `latent_updating_losses` | `(LU_steps,)` per batch | Loss at each LU iteration |
| `latent_updating_latents` | `(LU_steps, B, seq, Z_dim)` | Z trajectory during LU |
| `latent_updating_outputs` | `(LU_steps, B, seq, out)` | Outputs during LU |
| `latent_updating_combined_inputs` | `(LU_steps, ...)` | Combined inputs during LU |

#### Context / Task IDs

| Attribute | Shape per Entry | Description |
|---|---|---|
| `context_ids` | `(B, stride, 1)` | Low-level context IDs from dataset |
| `hlcids` | `(B, stride, 1)` | High-level context IDs |

#### Model Internals (optional)

| Attribute | Shape per Entry | Description |
|---|---|---|
| `hidden_states` | `(B, stride, hidden_size)` | RNN hidden state (if `log_hidden_states=True`) |
| `input_attention_weights` | `(B, stride, input_size)` | If `use_input_attention=True` |

#### Phase Markers

```python
logger.phases: List[Tuple[str, int]]
# Example: [('no inference learning', 0), ('Learning and inference', 200), ('Inference only', 1200)]
```

Each entry is `(phase_name, timestep_index)`. Plotting functions use these to draw vertical dividers and colored background panels.

#### `others` Dict

```python
logger.others: defaultdict(list)
```

A flexible catch-all for experiment-specific data. Common keys:

| Key | Description |
|---|---|
| `'grad_norms'` | Gradient norm of RNN weights per batch |
| `'task_latents'` | Oracle task latent (taskID) for comparison |
| `'timestep_passive_learning_ended'` | Index where passive phase ended |
| `'P'` | Gating mask state at various points |
| `'input_layer_weights'` | Snapshots of input layer weight matrix |
| `'rnn_cell.weight_hh'` | Snapshots of recurrent weight matrix |
| `'latent_effective_lr'` | Effective learning rate for Z (LR × grad magnitude) |

---

### Logger Methods

#### Data Logging

```python
logger.log_training_batch(batch)          # combined input
logger.log_training_loss(loss)            # MSE loss tensor → numpy
logger.log_testing_batch(batch)
logger.log_testing_loss(loss)
logger.log_input(inputs)                  # raw observations
logger.log_predicted_output(output)       # model predictions
logger.log_prediction_loss(loss)
logger.log_latent_value(Z)               # Z after LU
logger.log_latent_gradient(grad)
logger.log_gradients_corrections(grad)   # Z.grad after aggregation
logger.log_latent_updating_loss(loss)    # scalar per LU step
logger.log_latent_updating_latent(Z)
logger.log_latent_updating_output(out)
logger.log_llcid(context_ids)
logger.log_hlcid(hlcids)
logger.log_hidden_states(h)              # handles LSTM tuple
```

#### Phase / Structure

```python
logger.log_phase(phase_name)     # records (name, current_timestep)
logger.log_weights(model)        # LSTM gate weight snapshots
```

---

## `plot_logger_panels()`

```python
fig = plot_logger_panels(
    logger, config,
    panel_order=['behavior', 'latent', 'loss'],
    x2=None,           # optional: only show timesteps up to x2
    dpi=150,
    subplot_height=1.5,
    width=10,
    annotate_phases=True,
    rasterize=False,
)
```

Creates a stacked multi-panel figure. Each string in `panel_order` generates one subplot row.

### Available Panels

| Panel Name | Content |
|---|---|
| `'behavior'` | Observations (colored by context) + model predictions |
| `'latent'` | Z values over time (1D: line; 2D: stacked lines) |
| `'latent_2d'` | 2D Z as a 2-component stacked area chart |
| `'latent_chunk_1'` | Z chunk 1 only |
| `'latent_chunk_2'` | Z chunk 2 only |
| `'loss'` | Prediction MSE over time |
| `'gradients'` | ∂L/∂Z gradient magnitudes |
| `'task_illustration_and_hierarchies'` | Context structure: block boundaries, high-level latent |
| `'weights_grad_norm'` | RNN weight gradient norms over time |
| `'latent_effective_lr'` | Effective Z learning rate |
| `'corrects'` | Accuracy for seq_learn task |

### Phase Annotations

When `annotate_phases=True`:
- Vertical dashed lines drawn at each phase transition.
- Background colored panels mark each phase.
- Labels drawn at top of figure.

### Color Scheme

Colors are drawn from `plot_style.Color_scheme`:
- Context A observations: gold
- Context B observations: red
- Predictions: dark grey
- Different models have distinct colors for comparison figures

---

## Analysis Functions

### `get_corrects_and_trial_starts(logger)`

For the seq_learn task. Extracts state-level correctness:
1. Recovers state sequences from one-hot logger output.
2. Identifies trial start indices (state == 0).
3. Computes `corrects[t] = (predicted_state[t] == true_state[t])`.
4. Maps transitions by position in sequence: T0 (start), T1/2, T3/4, T5/6, T7/8, T9 (end).

Returns: `(corrects, trial_starts, transition_corrects_by_type)`

### `plot_corrects_seq_learn(logger, config)`

4-panel figure:
- **A**: Raw per-timestep accuracy with task structure background.
- **B**: Moving average accuracy (window = 1 block).
- **C**: Switch-aligned accuracy — accuracy as a function of trials since last context switch.
- **D**: Switch-aligned loss — same alignment for prediction loss.

### `get_matching_loggers(parameters_to_match, ...)`

Loads saved loggers from disk and filters by parameter values. Used for cross-seed averaging or hyperparameter sweeps. Loads `.npy` files from `export_folder`.

### `get_accuracy()` / `get_accuracies_averaged_across_time()`

Compute aggregate accuracy:
- Average across seeds and specified transition types.
- Optional: find the first timestep where EMA accuracy crosses a threshold (used to measure adaptation speed).

### `explore_data_container(data)`

Debugging utility that recursively prints the shape of nested structures (lists, arrays, tensors, dicts). Useful for inspecting logger contents without knowing the exact structure.

### `stats(var, var_name)`

Prints mean, variance, min, max, and L2 norm of any array-like object. Quick sanity check for gradient magnitudes, Z values, etc.

---

## `plot_style.py` — Visual Style

### `set_plot_style()`

Call once before any figure generation. Configures matplotlib defaults:
- Font: sans-serif, 7pt (publication-optimized)
- Line width: 0.7pt
- Spine: top and right removed
- Tick size: 2–3px
- Legend font: 6pt
- PDF backend: Type 42 (text as searchable text)

### `Color_scheme`

```python
cs = Color_scheme()

cs.contextA              # gold — context A observations
cs.contextB              # red  — context B observations
cs.neuragems_color       # NeuraGEM model color
cs.rnn_short_color       # RNN^short baseline
cs.rnn_long_color        # RNN^long baseline
cs.baseline_color        # generic baseline

cs.get_model_color(model_name)  # lookup by string name

cs.panel_small_size      # (width, height) for small panels
cs.panel_large_size      # for large panels
```

All colors are chosen to be distinguishable in both color and grayscale, suitable for print publication.

---

## Rasterization Utility

```python
rasterize_and_save(fname, rasterize_list, fig, dpi, savefig_kw)
```

For figures with many data points (time series over thousands of timesteps), the data traces are rasterized at the specified DPI while axes, labels, and text remain vector. This keeps file sizes manageable for PDF submission without degrading text quality.

`rasterize_list`: list of matplotlib artists to rasterize (typically the line collections and filled areas).

---

## Saving and Loading Loggers

Loggers are saved as numpy `.npy` files:
```python
np.save(config.export_path + 'logger.npy', logger)
logger = np.load(config.export_path + 'logger.npy', allow_pickle=True).item()
```

The logger is a plain Python object with numpy arrays, so it loads without any special imports.
