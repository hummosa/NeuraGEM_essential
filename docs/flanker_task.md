# Flanker Task — Implementation & Analysis Guide

Paper reference: *Cortical β Power Reflects a Neural Implementation of Decision Boundary Collapse*, JNeurosci 2024.
Human dataset: Fischer et al. 2018 / Kirschner et al. 2024 flanker EEG (N ≈ 1300).

---

## File Map

| File | Role |
|---|---|
| `configs.py` | `FlankerTaskConfig` (training), `FlankerRandomTrialsConfig` (test) |
| `datasets.py` | `FlankerTaskDataset`, `FlankerRandomTrialsDataset` + `DATASET_REGISTRY` |
| `run_flanker.py` | Main experiment script (`#%%` cells) — two stages plus all analyses |
| `flanker_analyses.py` | Trial extraction, factor construction, plotting, config/model helpers |
| `flanker_metrics.py` | Every per-seed scalar effect, grouped by question; `SIGNATURES` registry |
| `flanker_sweep.py` / `flanker_sweep_config.py` | Seeds × congruency-proportion sweep |
| `flanker_sweep_analysis.py` | Across-subject statistics: one-sample t-tests on the per-seed effects |
| `flanker_figure_utils.py` | Panel primitives (`bars_with_seeds`, `band`, `series`) and variant-aware loading |
| `flanker_sweep_figures.py` | The standing group figures for one condition |
| `flanker_model_figures.py` | Model card (congruency, distance, RT×outcome, adaptation, scorecard) + manipulation series |
| `train_and_infer_functions.py` | `train_model()` training loop |
| `functions_and_utils.py` | `plot_logger_panels()`, `Logger` class |
| `archive/run_flanker_blocked.py` | Retired blocked Stage 2/3 — see "Why blocks were dropped" |

---

## Two stages

| | Stage 1 — Training | Stage 2 — Test |
|---|---|---|
| Config | `FlankerTaskConfig` | `FlankerRandomTrialsConfig` |
| Dataset | `flanker_pretrain` | `flanker_random` |
| Structure | Blocked: 2 trials/block, target slot rotates `block_idx % 5` | **Randomly interleaved**, i.i.d. per trial |
| Weights | Plastic (BPTT) | **Frozen** (`no_of_steps_in_weight_space = 0`) |
| Z | Oracle — `context_ids` = target slot, fed as a one-hot | `'self'` — inferred by LU, one update per trial |
| Target slot | Rotates across blocks | Always centre (slot 2) |
| Length | `n_pretrain_trials` (2000) | `n_trials` (5000) |
| Analysed? | No — training diagnostic only | **Yes, everything** |

Stage 1 teaches the weights to use a spatial-attention latent. Stage 2 freezes them and
lets that latent become a controller updated by feedback on every trial. Conditions are
obtained by **masking one random session**, never by blocking.

### Why blocks were dropped

Blocked presentation confounds condition with time-in-block and with
adaptation-to-switch, and humans do randomly interleaved trials, so blocked effects are
not comparable to the human data. If a blocked result is ever needed, use asymptotic
within-block behaviour only — not the trials right after a switch. The old blocked
stages are preserved verbatim in `archive/run_flanker_blocked.py`.

---

## Task Structure

Five slots: `[far_left(0), near_left(1), center(2), near_right(3), far_right(4)]`.

**Stage 1**: target slot rotates across blocks. One companion slot is drawn randomly per
trial; its direction matches the target with probability
`p_corr_by_distance[|companion − target|]`. This correlation structure is what teaches
the model to weight near slots more heavily than far ones, and it is delicately balanced
— see the comment in `configs.py`. Raising it makes congruent trials easy but the model
never learns to use the slot identity; lowering it makes the model ignore the companion
entirely and no spatial structure is learned.

**Stage 2**: target is always centre. Flankers are slots 1 & 3 (near) or 0 & 4 (far), all
pointing the same way, either with the target (congruent) or against it (incongruent).

Each timestep: active slots receive `direction × signal_strength + N(0, arrow_noise_std)`,
inactive slots receive `N(0, bg_noise_std)`. Noise is resampled every timestep, so
evidence accumulates within the trial.

---

## Input / Loss Masking

6-dimensional input: `[obs_slot_0 … obs_slot_4 | true_direction]`.

```python
input_feed_mask  = [1, 1, 1, 1, 1, 0]   # model never sees dim 5
output_loss_mask = [0, 0, 0, 0, 0, 1]   # loss only on dim 5 (direction)
```

`logger.inputs[:, -1]` = unmasked ground-truth direction (always ±1.0).

## Speed Pressure

`predict_first_frame=True` → t=0 gets a zero frame; the response window starts at t=1.

```
temporal_loss_weights = [0, 1.0, e^-λ, e^-2λ, e^-3λ]   (λ = 0.7)
                      = [0, 1.0, 0.50, 0.25, 0.12]
```

`temporal_decay_factor = 0` → uniform weights. **If `arrows_duration` is ever changed,
λ must change with it** — at 15 timesteps λ=0.7 drives the tail to ~1e-4 and forces
near-instant responses. Parameterize by the end-of-window weight instead:
`temporal_decay_factor ≈ 2.1 / n_response_steps`.

---

## DataLoader Output & Oracle Z

DataLoader yields `(data, context_ids, hlcids)`:

| Field | Stage 1 | Stage 2 |
|---|---|---|
| `context_ids` | `float(target_slot)` → oracle Z | congruency flag (1.0/0.0), for block shading only |
| `hlcids` | congruency flag (1.0/0.0) | **trial type 0–3** |

Stage 2 trial types: `0` near-cong, `1` near-incong, `2` far-cong, `3` far-incong.

`what_latent_to_use='context_ids'`: the loop passes `context_ids` as oracle Z, and
`_get_Z_slice` converts the slot integer to a one-hot of length `Z_dim=5`.
`what_latent_to_use='self'`: the model infers Z via LU and `context_ids` is ignored.

---

## Key Config Parameters (`FlankerTaskConfig` defaults)

| Parameter | Value | Notes |
|---|---|---|
| `arrows_duration` | 5 | Timesteps per trial = `seq_len` = `stride` |
| `trials_per_context_block` | 2 | Trials per block, Stage 1 only |
| `signal_strength` | 1.0 | Arrow amplitude |
| `arrow_noise_std` | 1.3 | Noise on active slots |
| `bg_noise_std` | 0.1 | Noise on inactive slots |
| `p_corr_by_distance` | `[1.0, 0.65, 0.55, 0.25, 0.1]` | Companion correlation by slot distance |
| `temporal_decay_factor` | 0.7 | Speed pressure; 0 = uniform |
| `latent_dims` | `[5]` | Z_dim = 5, one unit per slot |
| `latent_activation` | `'softmax'` | Applied across slots before Z is used |
| `Z_lr` | 0.3 | LU step size |
| `Z_decay` | 1e-4 | L2 on Z |
| `hidden_size` | 64 | LSTM units |
| `p_congruent` / `p_near` | 0.5 / 0.5 | Stage 2 trial-type proportions |

---

## Three analysis conventions

These are enforced in `flanker_analyses.py`; ignoring them produces wrong Z conclusions.

**1. RT is interpolated inside the window and extrapolated beyond it.**
The decision variable is sampled once per timestep, so an integer RT takes only ~4
values. `rt_interp` linearly interpolates the threshold crossing between the bracketing
samples. Only crossings at or after `response_start_timestep` count; earlier timesteps
carry zero loss weight so their output is unconstrained.

Trials that never cross are **extrapolated, not dropped**. Failing to cross is
condition-dependent (~6% congruent vs ~21% incongruent), so discarding those trials
censored incongruent RTs *downward* and shrank the very effects being measured. The
response window is fitted with a least-squares line and projected forward to the
threshold, clipped to `[ad - 1, rt_cap]` with `rt_cap = 10` timesteps
(`flanker_analyses.RT_CAP`); a flat or falling trace takes the cap.

Two consequences to hold in mind:

- About half of the extrapolated trials are not rising at all and land exactly on the
  cap, so RT distributions have a spike at 10 and RT partly encodes *failure to decide*.
  Every RT cell in `flanker_metrics` is paired with a `dec_*` decided fraction for this
  reason, and `cong_effect_rt_decided` gives the crossed-in-window comparison.
- RT effects are therefore much larger than under the old censored convention (at
  baseline p=0.5, congruency effect 1.55 vs 0.33 timesteps) and one conclusion changes
  sign: PERI goes from null (−0.005, p=0.84) to reliably negative (−0.43, p=0.0005),
  i.e. interference *grows* after an error once slow trials are counted.

`decided` still marks which trials crossed inside the window; `report_extraction()`
prints the extrapolated fraction.

**2. Z is one value per trial, and it is logged *after* the update.**
The LU step aggregates the latent gradient over the trial's timesteps and broadcasts one
update back to all of them, so Z carries no within-trial dynamics. More importantly, the
value logged on trial *t* already contains trial *t*'s own update. Use:

- `z_in` / `focus_in` — the state the trial **inherited**; this is what drove its behaviour.
- `delta_z` / `delta_focus` — the update the trial **produced**.

Reading a trial's own Z as "the state it started from" was the alignment error in the
previous analyses. Note also the causal asymmetry: a Z *level* conditioned on the trial
outcome is circular (a focused Z is what made the trial correct), whereas `delta_focus`
grouped by trial properties describes the update rule and is legitimate.

**3. Z is logged raw; the gate is softmaxed.**
The RNN applies `softmax(Z / softmax_temp)` **across the 5 slots**, so a rise in raw dim 2
means nothing if the other four rose too. `z_act` is the activated version and
`focus = z_act[center] − mean(z_act[flankers])` is the scalar summary to report.

`reset_Z_uniform()` should be used instead of `set_Z(randn_like(Z) * 0.2)` when starting a
new stage: the latter gives every timestep a different random start which LU never
corrects, leaving frozen noise on the gate.

---

## `flanker_analyses.py` API

### `extract_trials(logger, config, rt_threshold=0.5, search_from=None, rt_cap=10.0)`

Returns a dict of per-trial aligned arrays. Key groups:

| Group | Keys |
|---|---|
| Behaviour | `correct` (n,ad), `output_traj`, `signed_output`, `true_dir`, `is_correct`, `response_side`, `correct_at_decision`, `resp_at_decision`, `rt`, `rt_interp` (never NaN), `decided`, `cross_idx` |
| Conditions | `trial_type` (hlcids), `context_id`, `trial_idx` |
| Latent | `z_raw`, `z_act`, `z_in`, `delta_z`, `focus`, `focus_in`, `delta_focus`, `z_traj`, `z_within_trial_spread` |
| Optional | `pe` (pre-LU prediction error), `z_grad` (aggregated dL/dZ) — `None` if not logged |
| Scalars | `rt_threshold`, `rt_cap`, `search_from`, `ad`, `n_trials`, `center_slot` |

`correct_at_decision` / `resp_at_decision` evaluate at the threshold-crossing timestep —
that is the response the model actually emitted. `is_correct` / `response_side` use the
final timestep and are kept for backward compatibility.

### `lagged_factors(trials, n_back=2, coding='distance')`

Current and lagged factors as float arrays with `NaN` in the undefined leading slots, so
`f['cong_1'] == 1` is always safe:

```
cong, near, correct, rt, side           # current trial
cong_k, near_k, correct_k, rt_k, side_k # k = 1..n_back
resp_rep     # this response equals the previous one
alternated   # the previous response differed from the one before it
valid        # trials with a full n_back history
```

### Other helpers

```python
select_trials(trials, trial_type=None, is_correct=None, response_side=None)
decode_trial_types(trials, coding='distance')   # -> is_cong, is_near
build_history_groups(trials, n_back=2, current_mask=None, congruent_types=(0, 2))
print_cell_counts(specs, label='')              # n per cell — thin cells should be visible
report_extraction(trials, label='')             # accuracy, censoring, Z spread
trial_measure(trials, measure)                  # measure spec -> (values, ylabel)
```

### Plotting

All take `specs = [(mask, label, color), ...]` and an optional `linestyles` list.

```python
plot_accuracy_by_timestep(ax, trials, specs, config, linestyles=None)
plot_rt_distribution(ax, trials, specs, config, fit_gaussian=True, undecided='extra_bin')
plot_rt_continuous(ax, trials, specs, config, bin_width=0.25)   # interpolated RT density
plot_z_by_timestep(ax, trials, specs, z_dim, config)            # flat by construction; archive only
plot_scalar_bars(ax, trials, specs, measure, group_spacing=None, baseline=None)
```

`plot_scalar_bars` measures: `'accuracy'`, `'accuracy_final'`, `'rt'`, `'rt_interp'`,
`'focus'`, `'focus_in'`, `'delta_focus'`, `'pe'`, `('z', d)`, `('z_in', d)`,
`('delta_z', d)`, `('z_raw', d)`. Legacy `('z_start', d)`, `('z_end', d)` and bare `int`
resolve to `('z', d)`. NaN-safe: censored RTs and undefined history entries are dropped
per group and the surviving n appears in the tick label.

### Config / model helpers

```python
sync_gating(stage_config, from_config)   # carry pre/post/add/mul gating across configs
mirror_to_model(model, stage_config)     # patch model.config AND the live Z optimizer
reset_Z_uniform(model, scale=0.2, seed=None)   # re-seed Z, shared across timesteps
```

---

## Analyses in `run_flanker.py`

| Figure | Question |
|---|---|
| `flanker_pretrain_results.pdf`, `flanker_pretrain_sanity.pdf` | Did Stage 1 train? Learning curve, accuracy by training third, RT (diagnostic, not a result) |
| `flanker_congruency_distance.pdf` / `_bars.pdf` | Congruency effect and the near > far interaction |
| `flanker_session_and_rt_by_outcome.pdf` | Accuracy across the frozen session; RT for correct vs error within each congruency |
| `flanker_accumulation.pdf` | Within-trial evidence accumulation, sign-normalised — the BPL analogue |
| `flanker_sequential_congruency.pdf` | All four history cells (CC/CI/IC/II) → I **and** → C, post-correct only |
| `flanker_sequential_timecourse.pdf` | The same history cells as within-trial accuracy and RT curves |
| `flanker_post_error_timecourse.pdf` | Post-error cells as within-trial curves |
| `flanker_sequential_by_repetition.pdf` | Does the sequential effect survive response-repetition control? |
| `flanker_post_error.pdf` | Post-error slowing / accuracy / inherited control state, incongruent A only |
| `flanker_post_error_near_far.pdf` | Do near errors drive more adaptation than far errors? |
| `flanker_z_update_drivers.pdf` | What *produces* the control update (`delta_focus`, `pe`) |

### Confound controls applied

- **CI and IC are separated.** They were conflated before; "congruent then incongruent"
  and "incongruent then congruent" are opposite trajectories.
- **History analyses are restricted to post-correct trials** (both t−1 and t−2 correct).
  Incongruent trials fail more often, so an unrestricted history cell partly measures
  post-error adaptation rather than conflict adaptation.
- **Both → I and → C targets are shown.** II→C vs CC→C is where a speed/accuracy
  dissociation would appear.
- **Response repetition is a splitting factor.** Congruency sequences correlate with
  response repetitions, and repetition priming produces a Gratton-shaped pattern with no
  control adjustment (Mayr, Awh & Laurey 2003). A sequential effect that only exists when
  repetitions are pooled in is priming, not control.
- **Post-error trial A is restricted to incongruent trials**, and trial B is split by
  congruency. Pooling A made the "post-correct" baseline a mixture dominated by
  congruent-correct trials; pooling B averaged an effect against its own opposite
  (a target-focused state helps incongruent B and hurts congruent B).
- **Near vs far is only compared within incongruent trials**, where flanker distance can
  matter at all.
- **Cell counts are printed for every panel.**

---

## Sweep: seeds × congruency proportion

`flanker_sweep.py` treats each seed as a synthetic subject — its own Stage-1 pretraining,
cached to disk — and tests that same frozen model at every `p_congruent` level, making
congruency proportion a *within-subject* manipulation as in the human design.

```bash
python flanker_sweep.py                 # sequential, resumable
SLURM_ARRAY_TASK_ID=7 python flanker_sweep.py   # one array element
```

Settings live in `flanker_sweep_config.py` (`SEEDS`, `N_TEST_TRIALS`,
`P_CONGRUENT_LEVELS`, …). Results are pickled per (seed, p_congruent, variant) and
reloaded with:

```python
from flanker_sweep import load_result, load_condition
results = load_condition(p_congruent=0.5)                      # baseline
results = load_condition(p_congruent=0.5, variant='decay5x')   # a variant
```

**Variants** add test-stage config overrides as extra sets of sessions, each with its
own `p_levels`. They reuse the cached pretrained models, so only the test session is
re-run. `'baseline'` keeps the original export paths; other variants get a
`p_congruent-X__name` folder, so adding one never invalidates existing results.

```python
VARIANTS = {
    'baseline': dict(overrides={},                  p_levels=P_CONGRUENT_LEVELS),
    'decay5x':  dict(overrides={'Z_decay': 5e-4},   p_levels=[0.5]),
}
```

`Z_decay` is the knob that sets *where* the control state settles — it pulls raw Z
toward zero, and zero is a uniform softmax, i.e. fully diffuse attention. `Z_lr` would
change how fast Z moves and how much it jitters, but not its operating point: with
Adam the stationary Z is where the task gradient balances the decay term, which is
independent of the learning rate.

Ten seeds turn every bar into a mean ± SEM **across subjects** rather than across trials
within one model — the error bar that corresponds to the human analyses.

`flanker_sweep_analysis.py` computes every effect *within* a seed and then tests it
across seeds with a one-sample t-test, plus a paired within-subject contrast across
`p_congruent` levels (the same pretrained model is tested at each level, so pairing is
valid and much more powerful).

**Single-seed results are not trustworthy here.** Several effects that looked clear in
one 5000-trial session — the near/far interaction, post-error reduction of interference,
a near-vs-far difference in the Z update — did not survive across ten seeds, and one
(which repetition condition carries the accuracy SCE) reversed sign. Always check an
effect against `flanker_sweep_analysis.py` before reporting it.

### Group figures

`flanker_sweep_figures.py` writes six figures next to the sweep results. Every panel is
a mean across seeds with SEM, plus one dot per seed; within-subject contrasts also get
thin lines connecting each seed across conditions.

| Figure | Contents |
|---|---|
| `group_fingerprint.pdf` | Accuracy and RT in the four cells; congruency effect by distance; **distance effect decomposed within each congruency** |
| `group_learning_and_rt.pdf` | Stage-1 learning curve, Stage-2 session stability, RT densities by congruency and by outcome |
| `group_within_trial.pdf` | Accuracy build-up, evidence accumulation, congruency cost over timesteps |
| `group_history.pdf` | Four history cells → I and → C; Gratton effect vs. response repetition |
| `group_post_error.pdf` | Post-error slowing and accuracy, inherited Z focus, what drives the update |
| `group_proportion_congruent.pdf` | Congruency effect vs. P(congruent), one line per subject |

Note the difference between the **interaction** and the **simple effects** of distance.
`interaction_acc` is a difference of differences and collapses the two directional
predictions — that near flankers *help* on congruent trials and *hurt* on incongruent
ones. Report `dist_effect_acc_cong` and `dist_effect_acc_incong` separately when the
question is about direction.

`flanker_sweep_analysis.py` and `flanker_sweep_figures.py` each carry their own default
congruency level; keep them in step when comparing tables to figures.

---

## Critical Gotchas

**1. `Z_lr` is not re-read per batch.** It is baked into the Adam optimizer at model
construction, and `set_Z` only rebuilds that optimizer when Z's *shape* changes — which
never happens across flanker stages. Assigning `config.Z_lr` alone therefore has no
effect, and the LU keeps stepping at whatever rate the model was born with. (This is why
the old Stage 2/3 really ran at `Z_lr = 0.3`, not the `0.4` written in the script.)
`mirror_to_model()` now patches the live param groups; always call it.

**2. `model.config` is a direct reference, not a copy.** When a later stage receives
`pretrained_model=model`, `model.config` still points at the training config.
`no_of_steps_in_latent_space` is read from it every batch. `mirror_to_model()` covers it.

**3. Stage configs don't inherit runtime gating changes.** `FlankerRandomTrialsConfig()`
is freshly constructed with class defaults, so a runtime `gating = 'post'` choice is
silently reverted. Always `sync_gating(stage_config, config)`.

**4. `trial_type` meaning is stage-dependent.** `extract_trials` puts `hlcids` into
`trials['trial_type']`. Stage 1: congruency flag (0/1). Stage 2: trial type (0–3). Pass
the right `coding` / `congruent_types`.

**5. `blocked_phase_length` overrides trial counts, and it is in timesteps.**
`train_model` recomputes `no_of_blocks = blocked_phase_length / block_size` at runtime.
Express session length in trials (`n_pretrain_trials * arrows_duration`,
`config.set_n_trials(n)`) so it stays correct if `arrows_duration` changes.

**6. DataLoader uses `shuffle=False`.** Trial order is exactly as generated — required
for the history analyses. Stage 2 trials are i.i.d. by construction, not by shuffling.

**7. The hidden state resets every trial, so Z is the only cross-trial carrier.**
`seq_len == stride == arrows_duration`, so one batch is one trial, and `forward()`
re-initializes `h`/`c` on every call. Nothing but `Z` survives a trial boundary — which is
precisely why the sequential-congruency, post-error and proportion-congruent results can be
read as Z-mediated control. `config.stateful_hidden = True` (default `False`) carries the
LSTM state across trials instead, giving those effects a second possible source; results
from the two settings are not comparable, and a stateful run needs its own Z-ablated
control before any effect is attributed to Z.

**8. `torch.load` needs `weights_only=False`.** These checkpoints are whole pickled
models, and torch ≥ 2.6 defaults `weights_only` to `True`.

---

## Deferred work

**Longer trials.** `arrows_duration = 5` gives only 4 response timesteps; interpolation
helps but RT variance stays compressed. Lengthening to ~15 requires rescaling
`temporal_decay_factor` (see Speed Pressure), raising `arrow_noise_std` by ≈√3 to avoid
ceiling accuracy, and recalibrating `rt_threshold`. It would also allow modelling the
flanker-before-target onset asynchrony (83 ms in the human task).

**Trial-history regression.** A per-seed trial-level GLM (log-RT via OLS on
correct + decided trials, accuracy via logistic) on effect-coded factors, then one-sample
t-tests on coefficients across seeds — the summary-statistics design used on the human
data, with each seed as a subject. `lagged_factors()`, `pe`, `z_grad`, `z_in` and
`delta_z` exist so this can be built without re-running anything.
