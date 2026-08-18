# Rotation Decoding — Latent Z vs. Hidden Activity

`rotation_decoding_analysis.py` — decode the rotation angle from the latent Z and from RNN
hidden activity, and compare them.

The question: during the generalization phase (Phase 3a, novel rotations), to what degree
is the rotation angle recoverable from Z versus from the hidden state, for each model in a
comparison? This is the representational counterpart to the behavioral analyses in
[rotating_targets_analysis.py](../rotating_targets_analysis.py), and it addresses the
framing of Yu et al. (2025), whose claim is that the neural code *lags* rapid behavioral
transfer.

See [task_rotating_targets.md](task_rotating_targets.md) for the task itself.

---

## Quick start

`run_rotating_targets_comparison.py` already wires this up. The standalone form:

```python
from rotation_decoding_analysis import (
    decode_rotation_across_models, plot_decoding_comparison, summarize_decoding,
)

# results maps model label -> {'logger': ..., 'cfg': ...}
decode_results = decode_rotation_across_models(results, phase='phase3a', frames='all')
summarize_decoding(decode_results)
fig = plot_decoding_comparison(decode_results, cfg)
```

Requires `config.log_hidden_states = True` during training.

---

## What is compared

Default bars are `'Z'` vs `'H'`. Fixed-width reductions of the hidden state are available
as diagnostics but are **not** fairness controls — see below.

| Key | Representation | Dims |
|---|---|---|
| `'Z'` | the latent, as logged | `Z_dim` (2–3) |
| `'H'` | full hidden state | `hidden_size` (64) |
| `'H_pls'` | hidden state reduced to `Z_dim` by PLS (supervised), fit in-fold | `Z_dim` |
| `'H_pca'` | hidden state reduced to `Z_dim` by PCA (variance-ordered), fit in-fold | `Z_dim` |

### Why there is no dimensionality-matched bar

`Z_dim` is 2–3 against `hidden_size` 64, which looks like it should favour the hidden
state. It does not, for two measured reasons:

1. **The readout is already rank-limited.** The target is 2-dimensional
   (`[cos θ, sin θ]`), so the ridge coefficient matrix is `(2, hidden_size)` — rank 2. A
   linear decoder projects the hidden state onto at most two dimensions no matter how many
   inputs it receives. Extra width buys no readout power.
2. **The null already tests what is left.** What width *does* buy is freedom to overfit,
   and that is exactly what the block-permuted null measures. On a 3-rotation run the
   full-`H` null sits at ~91° — chance. If the 64 dimensions were letting the decoder
   cheat under held-out-angle CV, the null would fall below 90°.

So shrinking `H` to `Z_dim` answers a different question — *how distributed is the code* —
and `decode_component_sweep` answers that one properly.

### Why `H_pca` looks catastrophic (and is not a bug)

On a measured run, PC0 carried 41.9% of hidden variance and correlated 0.915 with rotation,
while PC1 and PC2 correlated 0.038 and 0.067. The top PCs capture only **one** of the two
circular coordinates, and `cos θ` alone is two-to-one ambiguous (`cos θ = cos −θ`) — visible
as mirror-folding in `plot_decoded_vs_true`. The second coordinate lives in a low-variance
direction PCA does not reach until ~20 components:

| components | PCA | PLS |
|---|---|---|
| 1 | 46.3 | 46.2 |
| 3 | 45.5 | **28.1** |
| 10 | 34.9 | 22.9 |
| 20 | 23.6 | 21.3 |
| 64 | 19.6 | 19.6 |

(Z reference: 18.7° in 3 dims.)

Variance ranking and decodability are different orderings, which is why `'H_pls'` is the
honest fixed-width reduction. Plotting both is informative: a large PLS/PCA gap *is* the
finding that the rotation code occupies low-variance directions.

Fitting either reduction **inside each CV fold** matters — fitting on the full dataset
would leak held-out-angle structure into the projection.

---

## Method

**Target.** Ridge regression onto `[cos θ, sin θ]`, with `θ̂ = atan2(ŝ, ĉ)`. Rotation is
circular, so a linear regression onto the angle itself would be wrong at the wrap.

**Cross-validation: held-out angles.** Folds are grouped by rotation angle
(`GroupKFold(groups=angle_deg)`), so every fold tests angles absent from its training set.
A low error therefore means the representation *interpolates* to rotations the decoder
never saw — a continuous, metric code — rather than memorising a lookup of trained
contexts. The split is asserted leak-free per fold.

**Ridge penalty.** Selected by an inner group-CV over the training angles. Grouping the
inner split by angle too is deliberate: samples within a state-block are heavily
autocorrelated, so a random or leave-one-out inner split leaks and picks an alpha far too
small — which penalises the 64-dim hidden state most.

**Chance level: block-permuted null.** Held-out folds contain only a handful of angles, so
the analytic 90° is not the right reference. Each representation gets its own null by
reassigning each state-block another block's angle and re-running the identical CV.
Permuting at the *block* level (not the sample level) destroys the representation↔angle
mapping while preserving the within-block temporal structure that makes this data
autocorrelated in the first place. Null runs reuse the alpha selected on real data for the
matching fold, which keeps them paired and the runtime bounded.

**Timesteps.** `frames='all'` by default. Note that at outcome frames the attack `(x, y)`
is present in the input, so part of any hidden-state advantage is read off the current
stimulus rather than held context. Pass `frames='cue'` to measure held context only.

---

## API

### Extraction

```python
flatten_hidden_states(logger, config)          # -> (T, hidden_size), aligned with flatten_logger
extract_decode_samples(logger, config,
                       phase='phase3a',        # 'phase3a' | 'phase2' | 'all'
                       frames='all',           # 'all' | 'cue' | 'outcome'
                       z_lag=0)
verify_hidden_state_alignment(logger, config, model, phase='phase3a')
```

`verify_hidden_state_alignment` pushes each logged `h` through `model.output_layer` and
compares against the logged prediction. **It is only valid over a phase with frozen
weights** — during Phases 1–2 the output layer is updated after every batch, so the final
readout is not the one that produced an early logged prediction. Phase 3a runs with
`test_no_of_steps_in_weight_space=0`, so the check is exact there, and that is the phase
this analysis decodes from. Worth running once per run configuration: the whole pipeline
rests on this alignment.

### One logger or many

```python
as_runs(loggers, configs=None) -> [(logger, config), ...]
```

Accepts a single logger, a list of loggers, or a list of `(logger, config)` pairs;
`configs` may be a single config or a list. Falls back to `logger.config`, which
`train_model` sets. Every public function calls this first, so single-run and multi-seed
callers share one code path.

**Decoders are fit per run and only the scores are aggregated.** Pooling timesteps across
seeds would mix incompatible hidden bases — different networks, different coordinates.

### Decoding

```python
decode_rotation(samples, rep='Z', n_folds=6, n_shuffles=20, seed=0)
decode_rotation_for_runs(loggers, configs=None, phase='phase3a', frames='all',
                         reps=('Z','H'), z_lag=0, **kw)
decode_rotation_across_models(results, **kw)   # -> {label: result}
compare_z_lags(results, lags=(0, -1))          # timing diagnostic
```

### Component sweep — how distributed is the code?

```python
decode_component_sweep(samples, ks=(1,2,3,5,10,20,40,64), method='pls', tol=0.1)
component_sweep_for_runs(loggers, configs=None, method='pls', **kw)
component_sweep_across_models(results, method='pls')   # -> {label: result}
plot_component_sweep(sweep_pls, sweep_pca=None)
summarize_component_sweep(sweep_results)
```

`method='pls'` (supervised) is the headline; pass `method='pca'` as well and hand both to
`plot_component_sweep` to overlay the variance-ordered curve.

The scalar summary is `components_to_match_z`: the smallest number of hidden components
that comes within `tol` (default 10%) of Z's own error. The tolerance matters — the two
ceilings are often within a degree of each other, and a strict `<=` test makes the
statistic undefined on a gap far smaller than the seed-to-seed spread.
`components_to_match_z_strict` keeps the strict version. NaN means the hidden state never
reached Z within the swept range, which is a result rather than a failure.

`decode_rotation_across_models` reads `results[label]['loggers']` / `['cfgs']` when
present and falls back to the single-run `['logger']` / `['cfg']`. Moving a run script to
many seeds is therefore a change in the run script, not here.

Result dict per representation: `mae_deg`, `median_ae_deg`, `circ_r2`, `null_mae_deg`,
`null_circ_r2`, `mae_by_miniblock`, out-of-fold `theta_pred`, and sample/feature counts.
Aggregated results add `per_seed`, `mean`, `sem`.

### Reporting

```python
plot_decoding_comparison(decode_results, config)   # MAE per rep, grouped by model, with nulls
plot_decoded_vs_true(decode_results)               # out-of-fold θ̂ vs θ
summarize_decoding(decode_results)                 # printed table
print_lag_table(compare_z_lags(results))
```

### Figure conventions

All panels here are paper-panel sized — see [figure_style.md](figure_style.md), and call
`FigSize.dev()` while exploring. Sizes at paper scale with four model conditions:

| Figure | Size | Structure |
|---|---|---|
| `plot_decoding_comparison` | 2.5 × 1.2 in | one panel; width grows with bar count only |
| `plot_component_sweep` | 2.9 × 1.4 in | one panel, one line per model — **fixed size regardless of model count** |
| `plot_decoded_vs_true` | 4.7 × 1.7 in | one square per model (scatter clouds cannot overlay) |

Every one of them takes `figsize=(w, h)` to override the default outright:

```python
plot_decoding_comparison(decode_results, cfg, figsize=(2.0, 0.9))
```

Two encodings are used consistently:

- **Colour = model**, via `plot_style.get_model_color`, so a condition keeps its colour
  across every figure in the project.
- **Opacity = representation** (`Z` solid, `H` at 45%), so the legend needs two greyscale
  swatches rather than one entry per model × representation. This is what keeps the
  comparison panel small enough to drop into a multi-panel figure.

In `plot_decoded_vs_true` the hidden-state cloud is grey and drawn first, with the
model-coloured Z points on top.

---

## The Z / hidden timing asymmetry

`latent_values[t]` is Z **after** the latent update of batch `t`, while `hidden_states[t]`
comes from the weight-update forward pass, which ran with the Z carried over from batch
`t−1`. Z is therefore one latent-update step ahead of `h` at the same index.

`z_lag=0` (default) reads both at the logged timestep; `z_lag=-1` pairs each hidden state
with the Z that actually drove it. Measured on a 3-rotation run this changes the decoding
error by under 0.1° — Z evolves slowly within a block, so the asymmetry is immaterial in
practice. `compare_z_lags` re-checks it rather than assuming it stays that way.

---

## Self-test

```bash
.venv/bin/python rotation_decoding_analysis.py
```

Synthetic checks on the decoder, independent of any trained model: a representation that
*is* the rotation must decode held-out angles to near-zero error, a representation
independent of rotation must land at chance, and the block-permuted null must sit near
chance with near-zero circular R².

---

## See Also

- [task_rotating_targets.md](task_rotating_targets.md) — the task, block structure, context IDs
- [logging.md](logging.md) — `Logger` contents and the `hidden_states` shape caveat
- [figure_style.md](figure_style.md) — `FigSize` presets used by the plots
