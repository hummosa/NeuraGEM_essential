# Flanker sweep — handoff for the cluster session

You are picking up an in-progress modelling project. This document is the full state:
what is running, what the numbers mean, what we already found, what the queued runs are
supposed to test, and where to go next. Read the onboarding files in §10 first.

---

## 1. The scientific question

CRCNS Aim 1. The hypothesis is that cognitive control is **inference over a small set of
latent embedding units** in an RNN: weights hold slow, generalisable task structure,
while fast gradient updates to a latent vector `Z` reconfigure information flow to meet
the current trial's demand.

The empirical question is *which* latent has to be inferred for the human flanker
fingerprint to emerge. The candidate under test is the simplest one: **a latent encoding
the spatial location of the target** — an attentional controller.

Human comparison data: Fischer et al. 2018 / Kirschner et al. 2024 flanker EEG,
N ~ 1300, 50% congruent, with near/far flanker distance manipulated. Beta-power
lateralisation (BPL) is the within-trial neural proxy.

---

## 2. What the model is and how it runs

Two stages, in `run_flanker.py`:

- **Stage 1 (training).** Oracle Z: the target slot is handed to the model as a one-hot
  latent. Weights are plastic. Target slot rotates across short blocks. 2000 trials.
  **Nothing here is analysed** — it exists to teach the weights to use a spatial latent.
- **Stage 2 (test).** Weights **frozen**. Z is inferred online by gradient descent on
  prediction error, one update per trial, and is the only thing that adapts. Trials are
  drawn i.i.d. from four types: near/far x congruent/incongruent. 5000 trials.

Architecture: LSTM, 64 hidden units, `Z` of dim 5 (one unit per arrow slot), softmaxed
across slots and applied as a multiplicative gate on the hidden state. Trials are 5
timesteps; the response window is timesteps 1-4.

**Blocked designs were retired.** They confound condition with time-in-block and with
adaptation-to-switch, and humans do randomly interleaved trials. Every condition effect
now comes from masking one random session. The old blocked stages are preserved verbatim
in `archive/run_flanker_blocked.py`.

---

## 3. What the numbers mean

Be precise about this — it has caused real confusion.

| Term | Definition |
|---|---|
| **accuracy** | proportion of trials answered correctly (0-1), evaluated at the threshold-crossing timestep, i.e. the response actually emitted |
| **RT** | threshold crossing in timesteps: interpolated inside the response window, and **extrapolated** (least-squares projection of the window, capped at 10) for trials that never cross. No trial is dropped |
| **congruency effect** | congruent accuracy minus incongruent accuracy (or incongruent RT minus congruent RT) |
| **near - far** | computed **separately within congruent and within incongruent trials**. The two have opposite predicted signs |
| **interaction** | (near-far on congruent) minus (near-far on incongruent). A difference of differences — it collapses the two directional predictions, so always report the simple effects too |
| **focus** | `z_act[centre] - mean(z_act[flankers])`. The scalar control state. `z_act` is softmaxed, so it is the gate the RNN actually applied |
| **focus_in** | the focus a trial **inherited** (previous trial's value). This is what drove its behaviour |
| **delta_focus** | the update the trial **produced** |

**Directional predictions for distance.** Near flankers should *help* on congruent trials
(near - far positive) and *hurt* on incongruent trials (near - far negative).

**Two traps.**

1. Z is logged *after* the LU step, so a trial's own Z already contains its own update.
   Use `focus_in` for "what state did this trial start from". Reading a Z *level* after
   conditioning on the trial outcome is circular — a focused Z is what made the trial
   correct. `delta_focus` grouped by trial properties is fine; it describes the update rule.
2. Failure to cross threshold is condition-dependent (~6% congruent vs ~21% incongruent).
   Those trials used to be dropped, which biased observed incongruent RT *downward*; they
   are now extrapolated and capped at 10 timesteps instead. About half of them are not
   rising at all and land on the cap, so RT partly encodes failure-to-decide — read any RT
   cell next to its `dec_*` decided fraction, and use `cong_effect_rt_decided` for the
   crossed-in-window comparison. **Every RT number in §5 predates this change** and is
   roughly 4-5x smaller than the current convention gives; the accuracy numbers are
   unaffected.

---

## 4. Statistical standard — hold this line

**Each seed is a synthetic subject.** Compute every effect *within* a seed, then
one-sample t-test the per-seed values across seeds. That is the summary-statistics design
used on the human data.

**Single-seed results have been wrong repeatedly in this project.** In one 5000-trial
session the near/far interaction, PERI, and a near-vs-far difference in the Z update all
looked clear; none survived ten seeds, and one (which repetition condition carries the
accuracy SCE) reversed sign. Never report an effect from one seed.

**Report sign consistency, not just the group mean.** Several effects here have a
respectable group mean but only 5-6 of 10 seeds agreeing, with the mean dragged by two
outliers. Always print the per-seed values and the count of seeds with the predicted sign.
An effect carried by 8/10 seeds is a different animal from one carried by 5/10.

**State predictions before looking.** Section 6 lists what the queued runs are supposed to
show. Report failures as findings rather than tuning until they come back.

---

## 5. Results so far (10 seeds each; completed and on disk)

### Solid

- **Congruency effect.** At p(congruent)=0.5: accuracy 0.265 +- 0.025, RT 0.333 +- 0.029.
  Both p < 1e-5.
- **Post-error slowing.** +0.126 +- 0.020 (incongruent B), +0.120 +- 0.010 (congruent B).
- **List-wide proportion congruent.** The congruency effect grows as congruent trials get
  common: 0.217 -> 0.265 -> 0.336 across p = 0.2 / 0.5 / 0.8. Paired within-subject
  (0.8 minus 0.2): accuracy +0.119 +- 0.016, t=7.6; RT +0.076 +- 0.020, t=3.8. This is the
  classic direction — control relaxes when conflict is rare.

### Absent, despite earlier claims

- **The near > far distance interaction is not present at p=0.5.** near - far = +0.009 +-
  0.029 (accuracy), p=0.77. Simple effects: congruent +0.004 +- 0.005 (p=0.41),
  incongruent -0.005 +- 0.027 (p=0.86). Both point the predicted way but are ~60x smaller
  than the congruency effect, and only 5/10 seeds show the predicted sign.
- **PERI** (post-error reduction of interference): -0.005 +- 0.025, p=0.84.
- **Post-error accuracy improvement:** not significant in either direction.

`docs/progress_report_NIDA.md` claims the distance interaction and lists post-error
adaptation as in progress. **That claim needs revising or qualifying** — see below for the
condition under which it does hold.

### The distance effect is conditional

At p(congruent)=0.8 it appears: near - far on incongruent trials is **-0.078 +- 0.030**
(p=0.027), 8/10 seeds, and on congruent trials +0.012 +- 0.006. At p=0.2 it **reverses**
(+0.033). Monotonic across the three levels.

The mechanism is visible in the attention profile (mean softmax weight per slot; uniform
would be 0.200 each):

| condition | centre | near | far | near-far |
|---|---|---|---|---|
| p=0.2 | 0.457 | 0.112 | 0.159 | -0.047 |
| p=0.5 | 0.410 | 0.135 | 0.160 | -0.026 |
| p=0.8 | 0.344 | **0.168** | 0.160 | **+0.008** |
| p=0.5, decay 5x | 0.276 | 0.176 | 0.186 | -0.009 |
| p=0.5, decay 100x | 0.204 | 0.199 | 0.200 | -0.001 |

The list manipulation **reallocates**: near-slot weight climbs 0.112 -> 0.168 while
far-slot weight sits unmoved at ~0.160. That makes sense — near slots are the predictive
ones (`p_corr_by_distance[1]=0.65` vs `[2]=0.55`), so when flankers become informative the
useful move is to attend the *near* ones specifically.

Decay, by contrast, **flattens**: everything compresses toward uniform and near stays
below far throughout. This refuted an earlier hypothesis that "diffuse attention lets
distance matter" — 5x decay is *more* diffuse than p=0.8 (focus 0.095 vs 0.180) yet shows
a *smaller* distance effect (-0.037 vs -0.078).

**Current account: the distance effect is a signature of which slots the controller
weights, not of how diffuse it is.** Human prediction: proportion-congruent should
modulate *near*-flanker interference much more than far-flanker interference. The model
shows exactly that asymmetry — from p=0.5 to p=0.8, near-incongruent accuracy drops
0.666 -> 0.569 while far-incongruent barely moves, 0.670 -> 0.648.

### Two findings worth chasing

- **The control state lags by two trials.** At p=0.5, post-correct: the lag-2 congruency
  contrast on accuracy (0.0387 +- 0.0070, p=0.0004) is reliably *larger* than lag-1
  (0.0153 +- 0.0085, p=0.11); paired difference p=0.014. Cell means CC->I 0.645,
  CI->I 0.673, IC->I 0.696, II->I 0.694 group by the *older* trial. Possibly the Adam
  momentum on Z (betas 0.6/0.7). Testable in the human data.
- **The sequential congruency effect is carried by response repetitions** at p=0.5. RT SCE
  is flat (0.012 +- 0.014, p=0.43). Accuracy SCE is significant overall (0.021 +- 0.008,
  p=0.029) but sits entirely on repetition trials (0.036, p=0.005) vs switch trials
  (0.006, p=0.56), difference p=0.021. That is the Mayr, Awh & Laurey (2003) verdict:
  feature-integration priming, not control. **At p=0.8 it flips** — the switch-trial SCE
  becomes significant — but II cells are rare there (0.2^2 = 4% of trials) and the cell
  means are non-monotonic (II->I 0.637 falls *below* IC->I 0.686), so treat that as noisy.

### Decay series at p=0.5 (already run)

| | 1x (1e-4) | 5x | 25x | 100x |
|---|---|---|---|---|
| focus | 0.263 | 0.095 | 0.019 | 0.005 |
| congruency effect (acc) | 0.265 | 0.413 | 0.526 | 0.554 |
| near-far acc, incongruent | -0.005 | -0.037 | -0.059 | -0.070 |
| seeds with predicted sign | 5/10 | 5/10 | 6/10 | 6/10 |
| overall accuracy | 0.800 | 0.749 | 0.705 | 0.694 |

`Z_decay` sets **where** the control state settles — it pulls raw Z toward zero, and zero
is a uniform softmax. `Z_lr` would not: with Adam the stationary Z is where the task
gradient balances the decay term, which is independent of the learning rate.

25x and 100x are degenerate, not merely "less controlled": near-incongruent accuracy is
0.41 and 0.38, i.e. **below chance**, because with two flankers against one target and no
attentional weighting the model follows the flanker majority. Their one use is as the "no
control" limit — with attention flat, the residual near/far gap must come from the trained
weights, which shows the spatial asymmetry *is* baked into them. Usable range is 1x-5x.

---

## 6. What is queued and what it is supposed to show

100 test jobs total; 60 already complete, 40 new. Plus 40 pretraining jobs (4 model sets
x 10 seeds).

### A. Spatial gradient series — the main event

**Why.** In the test stage the target is always the centre slot, so near flankers sit at
distance 1 and far flankers at distance 2. The **only** two numbers that teach the model
anything about near vs far are `p_corr_by_distance[1]` and `[2]`. At the default 0.65 vs
0.55 that gap is **0.10** — a very shallow gradient, which likely explains why the
behavioural distance effect is small and carried by only half the seeds.

| variant | p_corr[1] / [2] | training gap |
|---|---|---|
| `spatial_flat` | 0.60 / 0.60 | 0.00 |
| baseline | 0.65 / 0.55 | 0.10 |
| `spatial_steep` | 0.75 / 0.52 | 0.23 |
| `spatial_steeper` | 0.85 / 0.51 | 0.34 |

All at p(congruent)=0.5. Both values stay above 0.5 so far companions remain weakly
informative rather than flipping to anti-correlated, which would be a different
manipulation.

**Predictions, stated in advance:**

1. The near-far accuracy gap on **incongruent** trials scales with the training gap, and
   becomes reliable (8+/10 seeds) at `spatial_steep` / `spatial_steeper`.
2. `spatial_flat` shows **no** distance effect in either congruency. This is the control
   condition. If it shows one, something other than training statistics is producing it
   and the whole account is wrong.
3. The near-far gap on **congruent** trials goes positive as the gradient steepens.
4. The attention profile at p=0.5 shifts weight onto near slots as the gradient steepens,
   i.e. `spatial_steeper` at p=0.5 should look like baseline at p=0.8.

Prediction 4 is the strongest test: it says the list manipulation and the training
gradient are two routes to the same reallocation.

**If the predictions fail:** the distance effect is not inherited from training statistics
the way we assume, and the next place to look is whether 2000 pretraining trials is simply
too few (see 7.2).

### B. `decay2x` — the matched-focus control

Focus should land around 0.15-0.18, matching what p(congruent)=0.8 produces (0.180). This
is the clean test of whether the *same control level* reached by a different route gives
the *same* behaviour. Based on the 5x result it should **not**: expect a smaller distance
effect than p=0.8 despite matched focus. That would confirm that allocation, not
magnitude, is what matters.

---

## 7. Where to take it next

1. **Blocked replication across seeds.** The retired blocked Stage 3 is what originally
   produced the distance-interaction claim. Within a congruent block the local congruency
   proportion is effectively 1.0 — the regime where the effect is strongest. Running
   `archive/run_flanker_blocked.py` across the same 10 seeds would test whether blocking
   *manufactured* the effect, and would reconcile the old claim with the new null at 50/50.
   Cheap, and directly relevant to what goes in the report.

2. **Longer pretraining.** The Stage-1 learning curve shows congruent accuracy plateaus by
   ~400 trials at 0.93, but **incongruent accuracy climbs from 0.52 to ~0.87 across all
   2000 trials and is still rising at the end**. The weights are frozen before the spatial
   weighting has converged — exactly the structure the distance prediction depends on. Try
   5000-10000 pretraining trials. It is the cheapest possible explanation for the weak
   effect and has not been tested.

3. **Trial-history regression.** Fully specced but not built. A per-seed trial-level GLM
   (log-RT via OLS on correct+decided trials, accuracy via logistic) on effect-coded
   factors, then one-sample t-tests on coefficients across seeds. Model set M1-M7 is
   written out at the bottom of `docs/flanker_task.md`. All the plumbing exists —
   `lagged_factors()`, `pe`, `z_grad`, `z_in`, `delta_z` — so it drops in without
   re-running anything. Use the `statsmodels` formula API. Key controls: restrict the
   sequential-congruency models to post-correct trials, and include `resp_rep` in every
   model, reporting M2 with and without it.

4. **Longer trials (retiming).** `arrows_duration=5` gives only 4 response timesteps, so
   the interpolated RT density is visibly bumpy with peaks between integers. Lengthening
   to ~15 needs three coupled changes: rescale `temporal_decay_factor` to ~2.1/n_response
   (at lambda=0.7 over 14 steps the tail is ~1e-4 and forces instant responses), raise
   `arrow_noise_std` by ~sqrt(3) to avoid ceiling accuracy, and recalibrate `rt_threshold`.
   Would also allow modelling the 83 ms flanker-before-target onset asynchrony.

5. **Aim-3 / TUS analogue.** `Z_decay` is the right knob — it moves the operating point,
   `Z_lr` does not. Usable range ~2x-5x; beyond that the model is degenerate rather than
   merely less controlled. Note decay weakens control *uniformly* while the list
   manipulation *reallocates* it, so the two interventions have distinguishable
   behavioural fingerprints. That is useful, not a problem.

---

## 8. Running it

```bash
# order matters: populate the model cache first, then the test sessions
./submit_job.sh 39 flanker_pretrain   # 40 jobs: 10 seeds x 4 model sets
./submit_job.sh 99 flanker            # 100 jobs: 10 seeds x 8 variants

# locally / sequentially (resumable, skips completed work)
python flanker_sweep.py pretrain
python flanker_sweep.py

# analysis
python flanker_sweep_analysis.py                                  # across-subject tables
python flanker_sweep_figures.py --p 0.5 --variant spatial_steep   # standing group figures
python flanker_model_figures.py --variant spatial_steep --p 0.5   # model card + manipulation series
```

Jobs are ~90 s each. `SKIP_EXISTING` makes everything resumable, so adding a variant only
runs the new work. Submit from the project root; `EXPORT_ROOT` is relative.

**Adding a variant** — edit `VARIANTS` in `flanker_sweep_config.py`. `overrides` are
test-stage config changes and reuse the baseline pretrained models; `pretrain_overrides`
change Stage 1 and get their own model cache. Stage-1 overrides are also applied to the
test config, because stimulus parameters must match across stages. `'baseline'` keeps the
original export paths, so adding variants never invalidates existing results.

**Disk.** Each session pickle is ~8 MB — it stores the full `Logger` (inputs, outputs, Z,
losses at every timestep). 100 sessions is ~800 MB. If that becomes a problem, store
per-trial summaries instead of full loggers (~50x smaller).

---

## 9. Gotchas that have already bitten us

1. **`Z_lr` is not re-read per batch.** It is baked into the Adam optimizer at model
   construction, and `set_Z` only rebuilds that optimizer when Z's *shape* changes, which
   never happens across stages. Assigning `config.Z_lr` alone silently does nothing. This
   is why the old blocked Stage 2/3 really ran at 0.3, not the 0.4 written in the script.
   `mirror_to_model()` now patches the live param groups; always call it.
2. **`model.config` is a reference, not a copy.** A later stage receiving
   `pretrained_model=model` still sees the training config. `mirror_to_model()` covers it.
3. **Stage configs do not inherit runtime gating changes.** Always
   `sync_gating(stage_config, config)`.
4. **`torch.load` needs `weights_only=False`** — these are whole pickled models and torch
   >= 2.6 defaults it to `True`.
5. **Z init must be shared across timesteps.** `set_Z(randn_like(Z) * 0.2)` gives every
   within-trial timestep a different random start that LU never corrects, leaving frozen
   noise on the gate. Use `reset_Z_uniform()`.
6. **Console encoding.** Arrow glyphs in labels raise `UnicodeEncodeError` on a cp1252
   terminal. `flanker_analyses._console_safe()` handles printing; figures are fine.
7. **`sys.argv` in a Jupyter / VS Code interactive window** belongs to the kernel, not to
   the script. `flanker_sweep_figures.p_from_argv()` only accepts a bare numeric argument.
8. **Blocked-design results are not comparable to interleaved ones.** If you use anything
   from `archive/`, say so explicitly and read asymptotic within-block behaviour only.

---

## 10. Onboarding — read these, in this order

### Essential, read fully before doing anything

| File | Why |
|---|---|
| `docs/flanker_handoff.md` | this document |
| `docs/flanker_task.md` | the main guide: task structure, both stages, the three Z conventions, the full `flanker_analyses` API, the confound controls, and the gotchas list |
| `flanker_sweep_config.py` | what is being run — seeds, trial counts, congruency levels, and every variant with its rationale |
| `flanker_sweep.py` | the runner: job matrix, variant-aware pretrain caching, atomic writes, result paths, loaders |
| `flanker_analyses.py` | `extract_trials` (every per-trial field), RT interpolation/extrapolation, `z_act` / `z_in` / `delta_z`, `lagged_factors`, plotting helpers, `sync_gating` / `mirror_to_model` / `reset_Z_uniform` |
| `flanker_metrics.py` | `session_effects` — the precise definition of every metric quoted in §5, grouped by question, plus the `SIGNATURES` benchmark registry |
| `flanker_sweep_analysis.py` | Across-seed statistics and the printed tables |

### Important, skim then consult as needed

| File | Why |
|---|---|
| `flanker_figure_utils.py` | panel primitives and variant-aware loading; `session_curves` builds learning curves, RT densities, within-trial dynamics |
| `flanker_sweep_figures.py` | the standing group figures for one condition |
| `flanker_model_figures.py` | the model card (congruency, distance, RT x outcome, adaptation, scorecard) and the manipulation series |
| `run_flanker.py` | the single-session script — what each figure shows and how the condition masks are built |
| `configs.py` | `FlankerTaskConfig` (Stage 1, including `p_corr_by_distance` and the note on why it is delicately balanced) and `FlankerRandomTrialsConfig` (Stage 2) |
| `datasets.py` | `FlankerTaskDataset` and `FlankerRandomTrialsDataset` — how trials are actually generated |
| `train_and_infer_functions.py` | the WU/LU loop: `predictive_learning`, `_latent_update_step`, and crucially the order (WU -> LU -> log), which is why Z is logged post-update |

### Background, consult when relevant

| File | Why |
|---|---|
| `models.py` | `RNN_with_latent`: `_get_Z_slice` (the softmax gate), `adjust_Z_grads` / `_apply_exponential_increase` (why Z has no within-trial dynamics), `_build_Z_optimizer` (why `Z_lr` is sticky) |
| `docs/progress_report_NIDA.md` | the claims sent to NIDA in July, two of which the ten-seed results contradict |
| `archive/run_flanker_blocked.py` | the retired blocked stages, for the replication in 7.1 |
| `docs/algorithm_predictive_learning.md` | the general NeuraGEM algorithm, if the WU/LU split is unfamiliar |
| `docs/Hans_paper_beta_power.pdf` | the BPL paper the within-trial accumulation analysis is meant to match |
| `submit_job.sh` | SLURM submission, including the two flanker branches |

**Do not start from the figures in `exports/` alone.** Several were generated at different
`p_congruent` levels during exploration and the filenames do not record which. Regenerate
with `flanker_sweep_figures.py <level>` before interpreting anything.

---

## 11. How to report back

Give the human: the effect, its mean +- SEM across seeds, t and p, **and the count of
seeds showing the predicted sign**.

Label every number with what it is — accuracy or RT? congruent or incongruent trials?
Ambiguity there has caused real confusion in this project. Prefer a small labelled table
over prose full of bare numbers.

Say plainly when a prediction failed. Do not tune parameters until a predicted effect
returns. If an effect only appears under one setting, that conditionality *is* the
finding — that is exactly how the proportion-congruent result was discovered.
