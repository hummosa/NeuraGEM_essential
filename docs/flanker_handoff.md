# Flanker sweep — handoff for the cluster session

You are picking up an in-progress modelling project. This document is the full state:
what is running, what the numbers mean, what we already found, what the queued runs are
supposed to test, and where to go next. Read the onboarding files in §9 first.

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
  latent. Weights are plastic. Target slot rotates across short blocks.
  **Nothing here is analysed** — it exists to teach the weights to use a spatial latent.
- **Stage 2 (test).** Weights **frozen**. Z is inferred online by gradient descent on
  prediction error, one update per trial, and is the only thing that adapts. Trials are
  drawn i.i.d. from four types: near/far x congruent/incongruent.

Architecture: an LSTM with `Z` of dim 5 — one unit per arrow slot — softmaxed across slots
and applied as a multiplicative gate on the hidden state. A trial is a handful of
timesteps, the first of which is a zero frame, so the response window is everything after
it. Session lengths, hidden size and trial length are in `flanker_sweep_config.py` and
`configs.py`; note the single-session script and the sweep do not have to agree on them.

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
| **RT** | threshold crossing in timesteps, interpolated inside the response window. A trial that never crosses did not respond and is given `rt = arrows_duration`, so it sits in one visible bin at the end of the axis rather than being dropped or projected to an invented value |
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
2. Failure to cross threshold is **condition-dependent** — incongruent trials fail
   several times more often than congruent ones — so it is a condition effect in its own
   right and is reported as one (`undecided_frac` per session, `dec_*` per cell, the last
   bin of every RT density) rather than folded into RT. Two earlier conventions both
   distorted the RT effects: dropping those trials censored incongruent RT downward, and
   extrapolating them to a cap let non-responses dominate every RT mean. Read any RT cell
   next to its `dec_*` decided fraction, and use `cong_effect_rt_decided` when you want
   the crossed-in-window comparison. RT numbers quoted under either retired convention are
   not comparable to current ones.

---

## 4. Statistical standard — hold this line

**Each seed is a synthetic subject.** Compute every effect *within* a seed, then
one-sample t-test the per-seed values across seeds. That is the summary-statistics design
used on the human data.

**Single-seed results have been wrong repeatedly in this project.** In one long session
the near/far interaction, PERI, and a near-vs-far difference in the Z update all looked
clear; none survived across seeds, and one — which repetition condition carries the
accuracy sequential effect — reversed sign. Never report an effect from one seed.
`run_flanker.py` is a workbench for mechanism and shape; `flanker_sweep_analysis.py` is
the evidence.

**Report sign consistency, not just the group mean.** Effects here can have a respectable
group mean with barely half the seeds agreeing and two outliers doing the work. Always
print the per-seed values and the count of seeds with the predicted sign; an effect
carried by 9 seeds in 10 is a different animal from one carried by 5.

**Seeds are the cheap fix for a noisy effect.** A split-half check showed a large share of
the across-seed spread in the sequential and post-error measures was within-session noise
at ten seeds — which is what made PIA and PERI flip between runs. One job per seed.

**State predictions before looking**, and report failures as findings rather than tuning
until a predicted effect returns. If an effect only appears under one setting, that
conditionality *is* the finding — it is exactly how the proportion-congruent result was
found.

---

## 5. Results — where to get them

**This document does not carry result numbers any more.** They were repeatedly overtaken
by re-runs, and a stale table is worse than no table. Regenerate instead:

```bash
python flanker_sweep_analysis.py                    # across-seed tables
python flanker_sweep_figures.py --variant noise10   # the group figure set
python flanker_regression.py  --variant noise10     # the same signatures as GLM coefficients
```

`flanker_sweep.describe_runs()` reports every run on disk with the parameters read from
the stored configs, so it says what a run actually contains rather than what a note claims.

For the standing qualitative picture — what holds, what fails, and the mechanism behind
each — see `flanker_metrics_status.md`. For the trial-history regression and the noise
series that explains the post-error failures, see `flanker_regression.md`.

Two conclusions from earlier rounds are worth carrying forward because they are about
*method*, not about a number:

- **The distance effect is conditional on the list.** It strengthens as congruent trials
  become common, and the attention profile shows why: the list manipulation *reallocates*
  weight onto the near slots, whereas `Z_decay` merely *flattens* the whole profile. Same
  focus magnitude by two routes gives different behaviour, so allocation is the thing to
  report, not magnitude. That conditionality is the finding, not a caveat.
- **Claims in `docs/progress_report_NIDA.md` predate all of this.** Two of them were
  contradicted by later multi-seed runs. Qualify them before reusing.

---

## 6. Where to take it next

1. **Blocked replication across seeds.** The retired blocked design is what originally
   produced the distance-interaction claim; within a congruent block the local congruency
   proportion is effectively 1.0, the regime where the effect is strongest. Running
   `archive/run_flanker_blocked.py` across the same seeds would test whether blocking
   manufactured it. Read asymptotic within-block behaviour only, and label the result as
   blocked — blocked and interleaved numbers are not comparable.

2. **Longer pretraining.** Congruent accuracy plateaus early in Stage 1 while incongruent
   accuracy is often still climbing when the weights are frozen — i.e. the spatial
   weighting the distance prediction depends on may not have converged. Check the Stage-1
   learning curve before treating a weak distance effect as a property of the model.

3. **Longer trials (retiming).** Only a handful of usable response timesteps means the
   interpolated RT density is visibly bumpy. Lengthening needs three coupled changes:
   rescale `temporal_decay_factor` to ≈2.1/n_response_steps, raise `arrow_noise_std` by
   ≈√3 to avoid ceiling accuracy, and recalibrate `rt_threshold`. Would also allow
   modelling the flanker-before-target onset asynchrony in the human task.

4. **The post-error failure.** The model has a prediction-error minimiser, not an error
   monitor, and no stimulus-noise level gives PES, PIA and PERI at once. The three
   candidate fixes, in increasing order of theory imported: more/faster latent updates; an
   error-gated learning rate; an explicit conflict representation feeding the latent.
   `flanker_metrics_status.md` states the trade-off each one buys.

5. **Aim-3 / TUS analogue.** `Z_decay` is the right knob — it moves the operating point,
   `Z_lr` does not. Beyond a moderate range the model is degenerate rather than merely
   less controlled: with attention flat the model follows the flanker majority and
   near-incongruent accuracy falls below chance. Its one use is as the "no control" limit.
   Decay weakens control *uniformly* while the list manipulation *reallocates* it, so the
   two have distinguishable behavioural fingerprints. That is useful, not a problem.

---

## 7. Running it

```bash
# order matters: populate the model cache first, then the test sessions
python flanker_sweep.py pretrain     # one pretrained model per seed per variant
python flanker_sweep.py              # the test sessions; resumable, skips completed work

# on the cluster
./submit_job.sh <n-1> flanker_pretrain
./submit_job.sh <n-1> flanker

# analysis
python flanker_sweep_analysis.py
python flanker_sweep_figures.py --variant noise10
```

`SKIP_EXISTING` makes everything resumable, so adding a variant only runs the new work.
Submit from the project root; `EXPORT_ROOT` is relative.

**Every setting lives in `flanker_sweep_config.py`** — `SEEDS`, `N_TEST_TRIALS`,
`P_CONGRUENT`, `VARIANTS`, `RUN_NAME`. `RUN_NAME` is the single switch: it decides where a
sweep writes *and* which run every analysis and figure script reads. **Never reuse a run
name for different settings** — the latent optimizer is baked into the pretrained model at
construction and `mirror_to_model` can only patch lr/decay, so a reused cache would
silently run the old optimizer.

**Adding a variant** — edit `VARIANTS`. `overrides` are test-stage only and reuse the
baseline pretrained models; `pretrain_overrides` change Stage 1, get their own model cache,
and are also applied to the test config, because stimulus parameters must match across
stages. Adding a variant never invalidates existing results.

**Disk.** Each session pickle stores the full `Logger` (inputs, outputs, Z, losses at every
timestep) and is large. If that becomes a problem, store per-trial summaries instead
(~50x smaller).


## 8. Gotchas that have already bitten us

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

---

## 9. Onboarding — read these, in this order

### Essential, read fully before doing anything

| File | Why |
|---|---|
| `docs/flanker_handoff.md` | this document — the question, the definitions, the standard |
| `docs/flanker_task.md` | the main guide: task structure, both stages, the three Z conventions, the full `flanker_analyses` API, the confound controls, the gotchas |
| `docs/flanker_metrics_status.md` | what currently holds and what fails, with the mechanism behind each |
| `docs/flanker_regression.md` | the trial-history regression, the human specification it mirrors, and the noise series |
| `flanker_sweep_config.py` | what is being run — seeds, trial counts, variants, and the run name every script follows |
| `flanker_metrics.py` | `session_effects` — the precise definition of every measure, plus the `SIGNATURES` benchmark registry |
| `flanker_analyses.py` | `extract_trials` (every per-trial field), the RT convention, `z_act` / `z_in` / `delta_z`, `lagged_factors`, plotting helpers, `sync_gating` / `mirror_to_model` / `reset_Z_uniform` |

### Important, skim then consult as needed

| File | Why |
|---|---|
| `flanker_sweep.py` | the runner: job matrix, variant-aware pretrain caching, atomic writes, result paths, loaders, `describe_runs` |
| `flanker_sweep_analysis.py` | across-seed statistics and the printed tables |
| `flanker_figure_utils.py` | panel primitives and variant-aware loading; `session_curves` builds learning curves, RT densities, within-trial dynamics |
| `flanker_sweep_figures.py` | the numbered group figures, in the order the story is told |
| `run_flanker.py` | the single-session workbench — what each figure shows and how the condition masks are built |
| `docs/figure_style.md` | figure sizing and the shared flanker palette (hue = congruency, shade = distance, fill = outcome) |
| `configs.py` | `FlankerTaskConfig` (Stage 1, including the note on why `p_corr_by_distance` is delicately balanced) and `FlankerRandomTrialsConfig` (Stage 2) |
| `datasets.py` | how trials are actually generated |
| `train_and_infer_functions.py` | the WU/LU loop, and crucially the order (WU → LU → log), which is why Z is logged post-update |

### Background, consult when relevant

| File | Why |
|---|---|
| `models.py` | `RNN_with_latent`: `_get_Z_slice` (the softmax gate), `adjust_Z_grads` (why Z has no within-trial dynamics), `_build_Z_optimizer` (why `Z_lr` is sticky) |
| `docs/progress_report_NIDA.md` | the claims sent to NIDA in July, some of which later multi-seed results contradict |
| `archive/run_flanker_blocked.py` | the retired blocked stages, for the replication in §6.1 |
| `docs/algorithm_predictive_learning.md` | the general NeuraGEM algorithm, if the WU/LU split is unfamiliar |
| `docs/Hans_paper_beta_power.pdf` | the BPL paper the within-trial accumulation analysis is meant to match |
| `code_shared/STA_BH/Regression_on_Behavior.m` | the human behavioural regression our GLM mirrors |
| `submit_job.sh` | SLURM submission, including the two flanker branches |

**Do not start from the figures in `exports/` alone.** Several were generated under
settings the filename does not record. Regenerate with `flanker_sweep_figures.py` before
interpreting anything.


## 10. How to report back

Give the human: the effect, its mean +- SEM across seeds, t and p, **and the count of
seeds showing the predicted sign**.

Label every number with what it is — accuracy or RT? congruent or incongruent trials?
Ambiguity there has caused real confusion in this project. Prefer a small labelled table
over prose full of bare numbers.

Say plainly when a prediction failed. Do not tune parameters until a predicted effect
returns. If an effect only appears under one setting, that conditionality *is* the
finding — that is exactly how the proportion-congruent result was discovered.
