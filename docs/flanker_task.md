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
| `flanker_regression.py` | Trial-level GLM on accuracy and RT, mirroring the human analysis — see `flanker_regression.md` |
| `flanker_sweep.py` / `flanker_sweep_config.py` | Seeds × stimulus-noise sweep; `flanker_sweep_config.py` holds every setting |
| `flanker_sweep_analysis.py` | Across-subject statistics: one-sample t-tests on the per-seed effects |
| `flanker_figure_utils.py` | Panel primitives (`bars_with_seeds`, `band`, `series`) and variant-aware loading |
| `flanker_sweep_figures.py` | The seven group figures, in the order the story is told |
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
| Length | `n_pretrain_trials` | `n_trials` (`config.set_n_trials`) |
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
temporal_loss_weights = [0, 1.0, e^-λ, e^-2λ, ...]       λ = temporal_decay_factor
```

Responding early is worth more than responding late, which is what makes RT a meaningful
measure at all. `temporal_decay_factor = 0` → uniform weights.

**If `arrows_duration` is ever changed, λ must change with it.** λ is a per-timestep
decay, so stretching the window drives the tail toward zero and forces near-instant
responses. Parameterise by the **end-of-window weight**, which is what actually sets the
pressure:

```
end_weight = exp(-λ · (n_response_steps - 1))     ⇒     λ = -ln(end_weight) / (n_response_steps - 1)
```

The shipped setting targets `end_weight ≈ 0.41`. The 5-timestep trial ran λ = 0.3 over 4
response steps (`e^-0.9 = 0.407`); the 10-timestep trial runs **λ = 0.112** over 9
(`e^-0.896 = 0.408`), i.e. the same speed pressure over a longer window.

(An earlier version of this file gave the rule as `λ ≈ 2.1 / n_response_steps`, which
targets `end_weight ≈ 0.12` — a much sharper pressure than any shipped config has used.
The formula above is the one the code follows.)

**λ is held fixed across target-onset delays.** `_set_temporal_weights` rebuilds the
window from λ, so a delay shortens nothing — `response_start_timestep` does not move. Were
λ re-fitted per delay level, the delay conditions would differ in speed pressure as well as
in onset, and the comparison would be confounded.

### Changing the trial length

`arrows_duration` is not a plain attribute. `seq_len`, `stride`, `block_size`,
`blocked_phase_length` and `temporal_loss_weights` are all derived from it at construction,
so **use `config.set_arrows_duration(n)`** — a bare assignment leaves the config internally
inconsistent (10-step trials against a 5-long weight vector and a stride of 5), and
`flanker_sweep` applies its overrides with a plain `setattr`, so this is not hypothetical.
`FlankerRandomTrialsConfig` overrides the setter because one batch is one trial there, so
its `block_size` is `arrows_duration` rather than a multiple of it. `_validate()` checks
the invariant `len(temporal_loss_weights) == arrows_duration == seq_len`.

---

## The target-onset delay ("flankers first")

The human task presents the **flankers before the target**. `config.target_delay` is that
onset asynchrony, in timesteps: the flankers are on from frame 0 as usual, and for the
first `target_delay` frames the centre slot carries background noise only.

**It is the target that is delayed, not the flankers that are previewed.** That framing
decides the whole implementation, because it says what must *not* be compensated for:

| | Setting | Why |
|---|---|---|
| Speed pressure | `response_start_timestep` stays **1**; `temporal_loss_weights` unchanged | The model is asked for the target direction from t=1 whether or not the target has arrived. Zeroing the loss over the pre-target window would remove the very pressure that makes an early, flanker-driven response possible |
| RT | measured from **trial start**, never re-referenced to onset | The question is whether the response is *delayed* when the target is late. Subtracting the delay back out would define the effect away |
| Latent update | descends the same loss, pre-target window included | During the delay the LU has only flanker evidence against a target-direction loss, which is the conflict the task is about. `latent_aggregation_op='exponential_increase'` weights later timesteps more when pooling the Z gradient, so post-onset steps still dominate the update |

So during the delay the read-out has nothing but the flankers. On a **congruent** trial they
already point at the answer, and the model can commit before the target exists. On an
**incongruent** trial the same early commitment is an error. That is the flanker effect,
and the delay amplifies it.

### Where it lives

Only `FlankerRandomTrialsDataset` implements it. Stage 1 trains on simultaneous onset, and
`FlankerTaskDataset` **asserts `target_delay == 0`** rather than ignoring it silently. Two
consequences follow:

- `stage1_fingerprint` is computed on the pretraining config, so a delay never invalidates
  the model cache. Delay variants carry test-stage `overrides` and no `pretrain_overrides`,
  which resolves them all to the `'shared'` pretrain tag — **every delay level reuses one
  pretrained model set per seed** and costs no extra pretraining.
- The read-out was calibrated on simultaneous onset, so its behaviour under a delay is an
  extrapolation. That is the intent — it is a probe — but it means a *null* result is
  ambiguous between "no delay effect" and "never calibrated for this". Training with the
  delay is a one-line variant, since the knob lives on `FlankerTaskConfig`.

The per-timestep target noise is drawn **unconditionally**, even while the target is absent,
and only applied once onset has passed. Skipping the draw would desynchronise the RNG
stream and hand each delay level a different trial sequence and different flanker noise;
drawing it regardless means one seed presents the *same* trials at every delay, so the
ladder is a within-seed comparison.

### Reading the figures

`extract_trials` returns `target_delay`, and every within-trial panel marks the first
timestep at which the target can reach the output — `target_delay + 1`, because
`predict_first_frame=True` means the model at timestep t has been fed frames 0..t-1. The
marker is a reading aid only; no measure is referenced to it.

`group_2_within_trial.pdf` and `run_flanker.py` Result 2 are where the mechanism shows:
everything left of that line is flanker-driven, so congruent traces climbing and
incongruent traces heading the wrong way *before* onset is the effect itself.

### One prediction that may not survive

The obvious prediction is that the congruency effect on RT grows with delay. **Watch for it
to shrink or reverse instead.** `rt_interp` is time-to-threshold regardless of correctness,
and a delay lets incongruent trials cross *early in the wrong direction* — so incongruent RT
can fall even as incongruent accuracy collapses. A pilot at `target_delay = 4` (one short,
deliberately undertrained session — not a result) showed incongruent accuracy falling far
more than congruent, `fasterr_incong_decided` rising sharply, and `cong_effect_rt`
shrinking rather than growing.

This is why `DELAY_PANELS` puts the two RT *levels* on one axis rather than only their
contrast, and pairs them with `cong_effect_acc` and `fasterr_incong_decided`: a pooled RT
contrast alone cannot tell "incongruent got slower" from "incongruent errors got faster".

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

## Key Config Parameters

**Read the values from `configs.py`** (`FlankerTaskConfig`, `FlankerRandomTrialsConfig`) —
they are tuned and re-tuned, and a number copied into this file goes stale silently. What
follows is what each knob *does* and why it matters, which does not drift. For a sweep,
`flanker_sweep.describe_runs()` reports the values a run on disk actually used.

| Parameter | What it controls |
|---|---|
| `arrows_duration` | Timesteps per trial = `seq_len` = `stride`. Sets how many usable response steps exist, so it bounds RT resolution. **10** (9 response steps); change it only through `set_arrows_duration()` |
| `target_delay` | Timesteps the **target's** onset is delayed while the flankers stay on from frame 0 — the human onset asynchrony. Test stage only; `FlankerTaskDataset` asserts it is 0. See "The target-onset delay" below |
| `trials_per_context_block` | Trials per block, Stage 1 only |
| `signal_strength` | Arrow amplitude on active slots |
| `arrow_noise_std` | Per-timestep noise on active slots. **The consequential one**: it decides how often the *target slot itself* misleads, which is what separates a bad-luck error from a control failure. The noise sweep is built on it |
| `bg_noise_std` | Noise on inactive slots |
| `p_corr_by_distance` | Companion correlation by slot distance in Stage 1. The gap between index 1 and 2 is the *only* thing that teaches the model near ≠ far, so it sets the ceiling on any distance effect |
| `temporal_decay_factor` | Speed pressure; 0 = uniform. Must be rescaled with `arrows_duration` |
| `latent_dims` | `[5]` — Z_dim matches the number of slots, one unit each |
| `latent_activation` | `'softmax'`, applied across slots, so Z is a normalised attention gate |
| `oracle_gate_jitter` | `True` (default range), an explicit `(lo, hi)`, or `None`/`False`. Redraws the sharpness of the Stage-1 oracle gate every trial. Off by default; see "The oracle gate is a constant" below |
| `Z_lr`, `Z_optimizer`, `Z_decay`, `Z_decay_mode` | The latent update. `Z_decay` sets *where* the control state settles (it pulls raw Z toward zero, and zero is a uniform softmax); `Z_lr` sets how fast it moves and how much it jitters, not its operating point. `Z_lr` is on a scale set by the optimizer — an SGD value and an Adam value are not comparable |
| `hidden_size` | LSTM units |
| `p_congruent` / `p_near` | Stage 2 trial-type proportions |

---

### The oracle gate is a constant, and Stage 2's is not

Stage 1 hands the model the target slot as a one-hot, which `latent_activation_function`
softmaxes into the gate the RNN applies. At `softmax_temp = 1` with `Z_dim = 5` that is the
same vector on **every** training trial: 0.405 on the target and 0.149 on each of the other
four. Stage 1 varies *which* slot the gate points at — the target rotates — but never *how
sharply*. Stage 2 varies both: the inferred gate's peak spans roughly 0.29 to 0.99 across
trials and is sharper than 0.405 on about three quarters of them.

That gap has a specific consequence. The weights are calibrated to emit ±1 at one gate
sharpness, so at a sharper gate the output overshoots — about 2.1 at peak 0.65 and 3.3 at
peak 0.97, against a target of 1.0. The latent update descends squared error to ±1, so on a
**congruent** trial, whose sign is already correct at every gate, the only remaining
gradient is "make the output smaller", and the update satisfies it by flattening the gate.
Half the list therefore teaches the controller to stop attending. The gate ends up peaking
off the target on a third or more of trials, and because a near display leaves slots 0 and 4
empty while a far display leaves 1 and 3 empty, the two conditions pay different prices for
it — which is where the spurious near-vs-far accuracy difference at low `arrow_noise_std`
comes from. `flanker_near_cong_diagnostic.py` has the full chain of evidence.

`config.oracle_gate_jitter = True` (or an explicit `(lo, hi)`) scales the one-hot by a factor drawn per trial before
the softmax, so the read-out is trained across a range of gate sharpness. Sharpness then
stops doubling as a gain control and the gradient on Z carries only "which slot". The factor
is drawn once per forward pass, shared across the trial's timesteps, and applied only on the
oracle path while weights are plastic — an inference stage using `what_latent_to_use='self'`
is untouched. `(0.5, 3.0)` spans gate peaks of about 0.29 to 0.83.

Measured against `None`, 3 seeds, 5000 test trials:

| | `arrow_noise_std` 0.4 | | `arrow_noise_std` 1.0 | |
|---|---|---|---|---|
| | off | jitter 0.5–3 | off | jitter 0.5–3 |
| overall accuracy | 0.871 | 0.981 | 0.837 | 0.844 |
| gate peaks off target | 0.297 | **0.000** | 0.087 | **0.002** |
| near-cong − far-cong accuracy | −0.028 | **+0.001** | +0.013 | +0.016 |
| distance effect, incongruent | −0.069 | −0.048 | −0.138 | **−0.199** |
| distance interaction | +0.041 | +0.049 | +0.151 | **+0.215** |
| congruency effect, accuracy | 0.195 | 0.037 | 0.227 | **0.252** |
| congruency effect, RT (decided) | 0.361 | 0.473 | 0.428 | **0.557** |
| non-responses | 0.128 | **0.022** | 0.085 | 0.066 |

At 0.4 the accuracy congruency effect collapses only because accuracy hits ceiling (0.98);
the RT effect grows. Three seeds — confirm through `flanker_sweep.py` before reporting any
of it.

### Confirmed at 20 seeds, and it downgrades the case for jitter

Every number in the table above was measured with `bg_noise_std = 0.1`. `run_flanker.py`
sets it to 0, and that turns out to do jitter's main job for it. A 2×2 over
`oracle_gate_jitter` × `p_corr_by_distance[2]`, 20 seeds per cell across the whole noise
ladder (`exports/flanker_random/factorial_corr_jitter`), at `arrow_noise_std` 0.9:

| | no jitter | jitter 0.5–1.5 | |
|---|---|---|---|
| near-cong − far-cong accuracy | **+0.011** | +0.017 | already fixed without jitter |
| PERI | **+0.073** | **−0.188** | jitter reverses it, away from humans |
| distance effect, RT incongruent | **+0.137** | +0.097 | loses significance |
| post-error slowing (`pes_BI`) | 0.108 | **0.375** | jitter's one real gain |
| mean focus (Z) | 0.342 | 0.391 | jitter drives the control state higher |
| human signatures matched | **9 / 11** | 8 / 11 | |

So the spurious near-vs-far difference the section above blames on constant gate sharpness
is at least as well explained by background noise in the empty slots: remove that and the
artifact goes without touching the oracle. Jitter is not as critical as this section reads,
and on balance it costs more than it buys — the higher Z it produces is the thing to suspect
for the reversed PERI and the weakened RT distance effect. `p_corr_by_distance[2] = 0.58`
was tested in the same factorial and is worse than 0.52 on both arms, roughly halving the
incongruent distance effect.

Across the whole ladder jitter never matches **more** signatures than the baseline at any
noise level — it ties at 1.3 and 0.7 (and is cleaner at 0.7: 0 signatures opposite vs 1)
and loses at 1.0, 0.9 and 0.4. Its one consistent effect is raising mean focus by about
0.05 at every level. The PERI reversal is mid-ladder: at `arrow_noise_std` 0.4 jitter
roughly doubles PERI instead (0.397 → 0.936), so the sign of its effect on PERI tracks
stimulus noise rather than being fixed.


---

## Five analysis conventions

The first three are enforced in `flanker_analyses.py`; ignoring them produces wrong Z
conclusions. The last two are enforced in `flanker_metrics.py` and decide whether a
sequential or an outcome-conditioned measure is the thing it claims to be.

**Plotting RT.** `plot_rt(ax, trials, specs, config, interpolate=False)` is the one
entry point. `interpolate=False` (default) draws the empirical PMF over the integer
crossing timestep with a final `und.` point; `interpolate=True` draws the sub-timestep
density over decided trials, scaled so its area is the decided proportion, with the same
`und.` marker. Either way the non-responses are a labelled category rather than a spike
inside the RT axis. `plot_rt_distribution` and `plot_rt_continuous` are kept as thin
aliases for the two modes.

**1. RT is interpolated inside the window; trials that never decide sit at the end.**
The decision variable is sampled once per timestep, so an integer RT takes only ~4 values.
`rt_interp` linearly interpolates the threshold crossing between the bracketing samples.
Only crossings at or after `response_start_timestep` count; earlier timesteps carry zero
loss weight so their output is unconstrained.

A trial that never crosses did not respond, and there is no honest number for when it
would have. It is given `rt = arrows_duration`, so those trials pile into one visible bin
at the end of the axis. Two earlier conventions both distorted the RT effects:

- **Dropping them (NaN).** Failing to cross is ~4x more common on incongruent trials, so
  censoring them biased incongruent RT downward and shrank the congruency effect.
- **Extrapolating to a cap of 10.** About half of those trials are not rising at all, so
  they landed on the cap and dominated every mean — the congruency effect on RT read 1.54
  timesteps that way against 0.33 on decided trials alone.

Failing to decide is itself a condition effect (~5% of congruent trials against ~16% of
incongruent), so it is reported as its own quantity rather than folded into RT:
`undecided_frac` per session, `dec_*` per cell, and the last bin of every RT density.
`decided` marks it per trial and `report_extraction()` prints the rate.

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

**4. A contrast on the previous trial's congruency is restricted to post-correct trials.**
Incongruent trials fail more often, so "after an incongruent trial" is also "after an
error" unless trial A's outcome is held fixed. Without the restriction, post-incongruent
slowing and post-error slowing are the same measure with two names.
`history_effects` and `post_conflict_effects` both apply it (`m['valid'] & m['pc']`); the
post-error block holds the mirror-image factor fixed instead, restricting trial A to
incongruent trials and splitting on outcome.

**5. An RT contrast between correct and error responses gets a `_decided` companion.**
`rt_interp` gives a trial that never crossed the trial end (convention 1), and errors are
disproportionately the trials that fail to cross — so on the uncensored measure a
non-response reads as a very slow error. `fasterr_*_decided` is what `SIGNATURES` scores
and `fasterr_*` is kept beside it, because the gap between the two *is* the censoring.
`cong_effect_rt_decided` is the older instance of the same convention. At
`rt_threshold = 0.2` the two barely differ (98%+ of trials decide); at 0.5 they differ a
lot, which is exactly why the pair is reported rather than one of them chosen.

---

## `flanker_analyses.py` API

### `extract_trials(logger, config, rt_threshold=0.5, search_from=None)`

Returns a dict of per-trial aligned arrays. Key groups:

| Group | Keys |
|---|---|
| Behaviour | `correct` (n,ad), `output_traj`, `output_full` (n,ad,output_size), `signed_output`, `true_dir`, `is_correct`, `response_side`, `correct_at_decision`, `resp_at_decision`, `rt`, `rt_interp` (never NaN), `decided`, `cross_idx` |
| Conditions | `trial_type` (hlcids), `context_id`, `trial_idx` |
| Latent | `z_raw`, `z_act`, `z_in`, `delta_z`, `focus`, `focus_in`, `delta_focus`, `z_traj`, `z_within_trial_spread` |
| Optional | `pe` (pre-LU prediction error), `z_grad` (aggregated dL/dZ) — `None` if not logged |
| Scalars | `rt_threshold`, `search_from`, `ad`, `n_trials`, `center_slot` |

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
plot_accuracy_by_timestep(ax, trials, specs, config, linestyles=None)   # P(target), see below
plot_rt_distribution(ax, trials, specs, config, fit_gaussian=True, undecided='extra_bin')
plot_rt_continuous(ax, trials, specs, config, bin_width=0.25)   # interpolated RT density
plot_z_by_timestep(ax, trials, specs, z_dim, config)            # flat by construction; archive only
plot_scalar_bars(ax, trials, specs, measure, group_spacing=None, baseline=None)
```

**`plot_accuracy_by_timestep` plots P(target), not accuracy.** It shows `trials['correct']`
— the fraction of trials whose decision variable currently has the target's sign — at every
timestep. There is no threshold and no commitment in it: a trial that crossed threshold three
timesteps ago still contributes its *current* sign, and one that never crossed contributes
throughout. The decision variable keeps integrating after the response is emitted and often
flips back toward the target (~4% of decided congruent trials, ~14% of decided incongruent
ones), which this curve counts and the emitted response does not. Its last point equals
`measure='accuracy_final'`; the bar panels use `measure='accuracy'` (`correct_at_decision`),
so the two differ by a few points and by more on incongruent trials.

### Single-trial visualisation

```python
plot_trial(trials, config, trial=0, show_gate=True, show_loss_weights=True,
           show_slot_channels=False)                    # -> fig
export_trial_figure(trials, config, trial, path=None)   # same, straight to PDF
example_trial_indices(trials, config, seed=0)           # [(label, index), ...], one per cell
trial_slot_roles(trials, config, trial)                 # target / flankers / empty
plot_correlation_structure(config)                      # the Stage-1 p_corr_by_distance knob
```

`plot_trial` is the task illustration: one trial as five slot-observation rows (up =
rightward, dashed line = the noiseless arrow level, so a trace on the wrong side of zero
is a slot that misled the model), the true direction, the temporal loss weights, and the
decision variable with its threshold crossing marked. Colour is **role**, not slot
identity — black target, congruency hue and distance shade for the flankers, grey for
slots holding no arrow. The gate column is the softmaxed attention weight, one bar per
slot, against a dotted line at the uniform value `1/n_slots`. Slots holding no arrow are
drawn, not omitted — at `bg_noise_std = 0` their trace is identically zero, so it is drawn
over the dotted zero reference and the row is labelled `empty`.

**Only the response output dim is plotted.** The model is a next-frame predictor with
`output_size == input_size == 6`, but `output_loss_mask = [0,0,0,0,0,1]` trains only the
direction dim, so output dims 0–4 — the slot channels — are never trained. Across a
session their SD is ≈0.1 against observations with SD ≈0.9, and their correlation with the
slot they nominally predict runs 0.04–0.56 with inconsistent sign: drift, not signal.
`show_slot_channels=True` draws them faintly for debugging the read-out.

**It reads from the logger, not from a live model.** This was previously done by
breakpointing inside `predictive_learning` and evaluating a script against its locals,
which is not necessary: `_log_batch` logs `inputs` — the raw **unmasked** batch tensor —
rather than the masked, time-shifted `model_inputs` the RNN is fed, so both the arrow
observations and the hidden true direction survive into `logger.inputs`, and
`predicted_outputs` keeps every output dim. Two alignment facts the panels depend on:

- `predict_first_frame=True` means the RNN at timestep *t* is fed the stimulus from
  *t−1*. The stimulus rows are drawn at the timestep the stimulus was **presented**, so
  the output at *t* responds to the arrows drawn at *t−1* and earlier, never at *t*.
- `update_latent_before_weights=False` means the logged outputs were produced under the Z
  the trial **inherited**. The gate column therefore shows `z_in`, not `z_act` — see
  convention 2 above. It is blank on trial 0, which inherited nothing.

Slot roles are read from the labels, with one exception: in Stage 1 the companion slot is
drawn per trial and never logged, so it is recovered as the non-target slot with the
largest |mean| over the trial. That is exact at `bg_noise_std = 0` (what `run_flanker.py`
and the sweep run) and right on 99.5% / 98.1% of trials at `arrow_noise_std` 0.9 / 1.3
with `bg_noise_std = 0.1`.

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
| `flanker_trial_<cell>.pdf` | What a trial of each condition actually looks like: slot observations, inherited gate, decision variable |
| `flanker_correlation_structure.pdf` | The Stage-1 `p_corr_by_distance` profile that teaches the model near ≠ far |
| `flanker_congruency_distance.pdf` / `_bars.pdf` | Congruency effect and the near > far interaction |
| `flanker_session_and_rt_by_outcome.pdf` | Accuracy across the frozen session; RT for correct vs error within each congruency |
| `flanker_rt_by_outcome.pdf` | Result 1c — the same split per cell, plus the fast-error contrast, decided trials only |
| `flanker_accumulation.pdf` | Within-trial evidence accumulation, sign-normalised — the BPL analogue |
| `flanker_sequential_congruency.pdf` | All four history cells (CC/CI/IC/II) → I **and** → C, post-correct only |
| `flanker_sequential_timecourse.pdf` | The same history cells as within-trial accuracy and RT curves |
| `flanker_post_error_timecourse.pdf` | Post-error cells as within-trial curves |
| `flanker_sequential_by_repetition.pdf` | Does the sequential effect survive response-repetition control? |
| `flanker_post_conflict.pdf` | Result 3c — post-incongruent slowing and accuracy, post-correct trial A |
| `flanker_post_error.pdf` | Post-error slowing / accuracy / inherited control state, incongruent A only |
| `flanker_post_error_near_far.pdf` | Do near errors drive more adaptation than far errors? |
| `flanker_z_update_drivers.pdf` | What *produces* the control update (`delta_focus`, `pe`) |
| `flanker_z_slot_update.pdf` | Result 5b — the same update per slot, geometry and role, plus the raw dL/dZ |

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

## Sweep: seeds × stimulus noise

`flanker_sweep.py` treats each seed as a synthetic subject with its own Stage-1
pretraining, cached to disk, and runs one test session per (seed, variant).

```bash
python flanker_sweep.py pretrain        # populate the model cache first
python flanker_sweep.py                 # sequential, resumable
SLURM_ARRAY_TASK_ID=7 python flanker_sweep.py   # one array element
```

Every setting lives in `flanker_sweep_config.py` — `SEEDS`, `N_TEST_TRIALS`,
`P_CONGRUENT`, `VARIANTS`, and `RUN_NAME`. `RUN_NAME` is the single switch: it decides
where a sweep writes *and* which run every analysis and figure script reads. Never reuse
a run name for different settings; the latent optimizer is baked into the pretrained
model at construction, so a reused cache would silently run the old one.

Results are pickled per (seed, variant) and reloaded with:

```python
from flanker_sweep import load_result, load_condition, describe_runs
describe_runs()                        # every run on disk, read from the stored configs
results = load_condition('noise10')    # every available seed for one variant
```

**Variants** are named sets of config overrides. `overrides` are test-stage only and reuse
the baseline pretrained models; `pretrain_overrides` change Stage 1 and get their own
model cache, and are also applied to the test config, because stimulus parameters must
match across stages. Adding a variant never invalidates existing results.

The current axis is `arrow_noise_std`, because it is the parameter the model's one clear
failure turns on — see `flanker_regression.md` §7 for the argument and the result. It is a
stimulus parameter, so it must match across stages, which makes the comparison across
noise *between*-subject rather than within.

`flanker_sweep_analysis.py` computes every effect within a seed and then one-sample
t-tests it across seeds. `flanker_metrics.SIGNATURES` is the registry of human benchmark
effects with the sign each should take — the single source of truth for the scorecard and
any pass/fail table, so read a verdict from there rather than hard-coding a list.

**Single-seed results are not trustworthy here.** Effects that looked clear in one
5000-trial session have repeatedly failed to survive across seeds, and at least one
reversed sign. A split-half check showed a large share of the across-seed spread in the
sequential and post-error measures is within-session noise at ten seeds, which is why
`SEEDS` was raised. Always check an effect against `flanker_sweep_analysis.py`, and report
the count of seeds with the predicted sign alongside the mean — an effect carried by 9/10
seeds is a different animal from one carried by 5/10 with two outliers doing the work.

### Group figures

`flanker_sweep_figures.py` writes the group figures next to the sweep results, numbered in
the order the story is told. Every panel is a mean across seeds with SEM, plus one dot per
seed; within-subject contrasts also get thin lines connecting each seed across conditions.

| Figure | Contents |
|---|---|
| `group_1_fingerprint.pdf` | Accuracy and RT in the four cells; congruency effect by distance; **distance effect decomposed within each congruency** |
| `group_2_within_trial.pdf` | P(target) build-up, evidence accumulation, congruency cost over timesteps |
| `group_3_rt.pdf` | Row 1: RT densities by congruency and by outcome, and the trials that never decide. Row 2: the same densities per cell for correct and for error, and the fast-error contrast |
| `group_4_history.pdf` | Four history cells → I and → C; Gratton effect vs. response repetition |
| `group_5_post_error.pdf` | Post-error slowing and accuracy, inherited Z focus, what drives the update |
| `group_6_circularity.pdf` | The control deficit precedes the error, so post-error state is circular |
| `group_7_scorecard.pdf` | Every human signature on one axis, matched or not |
| `group_8_noise_series.pdf` | Each signature against `arrow_noise_std` — why the post-error failures happen |
| `group_9_z_update.pdf` | Δ Z focus per cell, split correct vs. error; what a given inherited state buys in accuracy and in RT |
| `group_10_post_conflict.pdf` | Post-incongruent slowing and accuracy — the conflict twin of `group_5`, trial A post-correct |
| `group_11_z_slot_update.pdf` | The update per slot rather than as `focus`: fixed geometry, slot role, and the raw dL/dZ |
| `group_12_delay_series.pdf` | Each RT-relevant signature against `target_delay` — does a later target mean a later response? |

Every per-variant figure lands in that variant's own folder; `group_8_noise_series.pdf` and
`group_12_delay_series.pdf` span a ladder of variants and land one level up, beside the
variant folders. Both are built by `fig_ladder_series`, which takes a ladder of
`(variant, x_value)` pairs and a panel set; `fig_noise_series` and `fig_delay_series` are
thin wrappers that supply `NOISE_PANELS` / `DELAY_PANELS` and their own axis label. A
ladder figure needs at least two levels on disk and prints a skip message otherwise.

`group_10` and `group_11`, and `group_3`'s second row, are built from the `spec_*`
builders in `flanker_figure_utils.py`, which `run_flanker.py` Results 1c, 3c and 5b also
draw from — one panel definition, two callers, so the workbench and the group view of a
measure cannot drift apart. `_as_replicates` is what absorbs the difference between them:
a seed is the replicate at group level, the session is the replicate in the workbench. The
cost is that a single-replicate panel has no error bar, so where the trial-level spread is
the point, `flanker_analyses.plot_scalar_bars` with masks is still the right tool.

```bash
python flanker_sweep_figures.py                    # every variant in the sweep
python flanker_sweep_figures.py --variant noise10
```

Colours follow the shared flanker palette in `plot_style.FLANKER_COLORS` — hue for
congruency, shade for distance, fill for outcome. See `figure_style.md`.

### What the three new measures say (baseline arm, `noise13`, 20 seeds)

All three are 20/20 seeds and p < 0.001, so the directions below are not seed noise.

**Post-incongruent adaptation splits the human prediction in half.** Accuracy behaves:
`pca_BI` = +0.080, more accurate on an incongruent trial after an incongruent one, and
`pca_BC` = −0.039, the cost on a congruent trial that a genuine control adjustment has to
pay. RT does not: `pcs_BI` = −0.238, the model gets *faster* after conflict, and the lag-2
cell contrast agrees (II→I is 0.33 timesteps faster and 0.109 more accurate than CC→I).
That is the interference-reduction (Gratton) pattern, not the slower-and-more-deliberate
pattern — the model buys accuracy without paying in time, so it has no speed/accuracy
trade-off to trade. `pcs_BI` is scored +1 in `SIGNATURES` on the deliberateness reading,
so it currently reads 0/20; that row is a claim about the human data, and if the
interference-reduction reading is the right one for this dataset the sign should flip.

**The model does not show fast errors.** `fasterr_incong_decided` = −0.047: errors are
*slower* than correct responses, and much more so on congruent trials (−0.199). Censoring
is not the explanation — at `rt_threshold = 0.2` about 98% of trials decide in both cells
— so this is a real mismatch with the human data rather than a measurement artifact.

Neither failure is a property of one noise level. Across the whole ladder `pcs_BI` is
negative at 0/20 seeds at every rung (−0.24 at 1.3 through −0.34 at 0.4) and
`fasterr_incong_decided` stays between −0.045 and −0.089, so lowering stimulus noise — the
manipulation that repairs the post-error signatures — does not touch either of these. They
are structural, not a regime the model happens to be in. (`pca_BI` does fade, +0.080 to
+0.002 at `arrow_noise_std` 0.4, which is the accuracy ceiling documented above rather
than a loss of adaptation.)

**Congruent errors are what teach the controller to stop attending.** Per slot, an
incongruent error moves the gate the way an error monitor would want: centre +0.042,
flanker slots −0.040. A congruent error does the opposite and three times harder: centre
−0.138, flanker slots +0.049. Nothing in the incongruent half of the list is wrong with
the update rule; the damage is done by the half of the trials that had nothing to be
misled by. This bears directly on the deferred experiment below — an error-gated learning
rate that does not condition on congruency would amplify the harmful update more than the
helpful one.

Note the difference between the **interaction** and the **simple effects** of distance.
`interaction_acc` is a difference of differences and collapses the two directional
predictions — that near flankers *help* on congruent trials and *hurt* on incongruent
ones. Report `dist_effect_acc_cong` and `dist_effect_acc_incong` separately when the
question is about direction.

`flanker_sweep_analysis.py` and `flanker_sweep_figures.py` each carry their own default
variant; keep them in step when comparing tables to figures.

---

## Critical Gotchas

**1. `Z_lr` is not re-read per batch.** It is baked into the optimizer at model
construction, and `set_Z` only rebuilds that optimizer when Z's *shape* changes — which
never happens across flanker stages. Assigning `config.Z_lr` alone therefore has no
effect, and the LU keeps stepping at whatever rate the model was born with; this has
silently invalidated a stage before. `mirror_to_model()` patches the live param groups;
always call it. The optimizer choice itself cannot be patched after construction, which
is why a sweep run must never be reused across optimizers.

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
precisely why the sequential-congruency, post-error and list-level results can be
read as Z-mediated control. `config.stateful_hidden = True` (default `False`) carries the
LSTM state across trials instead, giving those effects a second possible source; results
from the two settings are not comparable, and a stateful run needs its own Z-ablated
control before any effect is attributed to Z.

**8. `torch.load` needs `weights_only=False`.** These checkpoints are whole pickled
models, and torch ≥ 2.6 defaults `weights_only` to `True`.

---

## Deferred work

**Blocked replication across seeds.** The retired blocked design is what originally
produced the distance-interaction claim, and within a congruent block the local congruency
proportion is effectively 1.0 — the regime where that effect is strongest. Running
`archive/run_flanker_blocked.py` across the same seeds would test whether blocking
manufactured the effect. Read asymptotic within-block behaviour only, and say explicitly
that the result is blocked, because blocked and interleaved numbers are not comparable.

**Longer pretraining.** Congruent accuracy plateaus early in Stage 1 while incongruent
accuracy is often still climbing when the weights are frozen — i.e. the spatial weighting
the distance prediction depends on may not have converged. Check the Stage-1 learning
curve before treating a weak distance effect as a property of the model.

**Error-gated inference learning rate.** Matt Nassar's proposal: raise the latent learning
rate after a perceived error, and ask whether that produces post-error slowing, post-error
accuracy gains, and better overall accuracy — the last being the normative rationale for
making the adjustment at all. Not implemented; this is the costed design.

*Step 0, free, do it first.* `event_locked` already returns `curve_rt` — RT on incongruent
trials against the inherited control state — and `group_9_z_update.pdf` panel 4 draws it
across 20 seeds already on disk. Read its slope. If RT *falls* as inherited focus rises,
then anything that leaves the post-error state better focused predicts post-error
**speeding**, and this mechanism cannot produce PES on its own. The post-incongruent
result above says exactly that is the risk: the model already gets faster and more
accurate together, with no trade-off between them.

*The signal.* `_latent_update_step` in `train_and_infer_functions.py` already computes
`before_optim_loss`, the pre-LU per-element loss; masked to the response dim by
`_mask_loss` that is the model's own prediction error. It is legitimately available, since
it is exactly what LU descends — the model never sees the direction as *input*
(`input_feed_mask` zeroes dim 5), only in the loss. A binary variant is the sign of the
response-window-weighted output against `inputs[..., -1]`.

*The mechanism.* Scale the Z optimizer's learning rate for that trial's LU step, then
restore it. It must patch `model.Z_optimizer.param_groups[*]['lr']` — **not**
`config.Z_lr`, which is baked in at construction and is Gotcha 1 below.
`flanker_analyses.mirror_to_model` is the existing example of doing it right.

*Timing.* `update_latent_before_weights = False`, so trial *t*'s LU runs at the end of
trial *t*. Scaling **that** step is what trial *t+1* inherits, which is the mechanism as
described. Scaling the trial *after* the error is a different model and delays the effect
by one trial; if both are wanted they are two variants, not one knob.

*Config.* `Z_lr_error_mode = None` (`'binary'` | `'pe'`), `Z_lr_error_gain = 1.0`,
`Z_lr_error_threshold` — all default-inert, so every pickled run and every existing figure
is unchanged. Log the per-trial learning rate so `extract_trials` can expose it and the
analysis can confirm the gate fired at roughly `1 - acc_overall`.

*Sweep.* Test-stage `overrides`, not `pretrain_overrides` — the stimulus is unchanged, so
these variants reuse the baseline pretrained models and cost no re-pretraining:
`'errlr2': dict(overrides={'Z_lr_error_mode': 'binary', 'Z_lr_error_gain': 2.0})`.

*The prediction that decides it.* The model's error signal cannot separate a noise-driven
error from a flanker-driven one — that is what `error_diagnosis_effects` and
`frac_err_noisy` / `dfocus_err_noisy` measure — and, per the slot-wise result above, it
also cannot separate a congruent error from an incongruent one. A flat gain amplifies all
of them, including the congruent-error update that is both the largest and the one
pointing away from the target. So the gain has to be crossed with the noise ladder, and a
congruency-conditioned gate is the obvious second variant. Expect help where
`dfocus_err_noisy` has already crossed zero (low `arrow_noise_std`) and harm where it has
not; `group_8_noise_series.pdf` is the figure that would show it. Read-outs: `pes_BI`,
`pia_BI`, `peri`, `dfocus_err_noisy`, and `acc_overall` for the normative question.

(The trial-history regression that used to sit here is built: `flanker_regression.py`,
documented in `flanker_regression.md`.)
