# hier_switch — the cue-noise sweep

The running record for running the stack at more than one `pulse_noise_std`. What the
question is, what an optimal reader can possibly buy, how the levels are kept apart on
disk, and what each level actually produced.

Definitions of every derived quantity are in `docs/hier_switch_analyses.md`; the narrative
the results feed into is `docs/hier_switch_methods.md`.

---

## The question

The model loses less accuracy to cue conflict than the animals do, so the low- against
high-conflict split in story panel d is weaker than the paper's
(`docs/hier_switch_methods.md`, "Why the low/high gap in panel d is smaller than the
paper's"). `pulse_noise_std = 0.5` was never tuned — it was set once to make conflict an
actual sensory uncertainty rather than a label, since with noiseless pulses counting
recovers the cue exactly at every conflict level. Raising it is the faithful fix: it makes
the paper's own conflict levels bite, rather than substituting more extreme ones.

It also changes the task and needs every model retrained, which is what this sweep costs.

## What noise can buy: the ideal-observer calibration

Measured with `hier_switch_observer.ceiling(sigma, n_trials=20000)` — a fresh test stream,
no model, no training, `env_seed` 0 and `data_stream` 1. Accuracies are how often the ideal
observer reads the cue correctly; the paper's two forced levels are conflict 0.29 (7:2) and
0.5 (6:3).

| pulse noise | @ conflict 0.29 | @ 0.5 | **gap** | ceiling (all levels) |
|---|---|---|---|---|
| **0.5 (the existing tree)** | 0.988 | 0.927 | 0.061 | **0.9231** |
| **0.6 (this run)** | 0.962 | 0.867 | **0.095** | **0.8944** |
| 0.7 | 0.930 | 0.819 | 0.110 | 0.8696 |
| 0.8 | 0.897 | 0.779 | 0.118 | 0.8461 |
| 0.9 | 0.866 | 0.746 | **0.120 (max)** | 0.8235 |

Two things follow, and the second is why the sweep is small.

**No noise level reaches the paper's ~0.20 gap** with 9 informative pulses. The gap
saturates near 0.12 around σ 0.8–0.9 and then shrinks, because past that the low-conflict
end collapses too. This is a property of an optimal reader, not of the model, so it bounds
what any amount of retraining can produce.

**σ 0.6 already captures ~80 % of what is achievable**, for 0.029 of ceiling. So: run 0.6,
then reassess. 0.8–0.9 is deferred pending the 0.6 result.

The model's own gap at σ 0.5 is larger than the observer's (0.084 against 0.061), so if it
scales, expect roughly 0.12–0.15 at σ 0.6. That projection is the thing this run measures.

**The ceilings are fixed numbers, not recomputed per run.** They live in
`hier_switch_observer.CEILING` and the estimate moves by ~0.007 between sample sizes (σ 0.5
gives 0.930 / 0.927 / 0.923 at 2000 / 4000 / 20000 trials). Selection is expressed as a
fraction of them, and a criterion that drifts between runs is worse than one that is
slightly off. `hier_switch_observer.py` run as a script checks every entry.

## The levels are paired, trial for trial

The dataset RNG is keyed on `(env_seed, data_stream)` and the noise draw is the last
consumer per trial (`hier_switch_dataset.py`, `rng.normal(0, sigma, ...)`), so the same
seed at a different sigma sees an **identical** cue / conflict / context / vis sequence with
the same standard-normal draws scaled by sigma. Verified bit-identical between 0.5 and 0.6,
with the pulse-channel difference having s.d. exactly 0.100.

Worth stating in any write-up, and worth stating with its caveat: **this pairs the data,
not the models.** Training at 0.6 sees different gradients from trial 1, so "seed 7 at 0.6"
is a different network, not a perturbed one, and re-selection makes the seed sets differ
anyway. Across levels the comparison is between independent replications that happen to
share a stimulus stream. The one genuinely within-network comparison is the probe below.

---

## How the levels are kept apart

`exports/hier_switch/` is one tree, shared by every worktree through a symlink. Nothing in
it is namespaced by noise level, and re-running as-is would not error — it would silently
mix, because a condition whose `session.npz` exists is skipped and reported as done.

**The tune tag does the isolating, and it does it for free.** Both session writers derive
their tag from the model's own path (`hier_switch_test_inference.main`,
`hier_switch_perturb.clamp_grid`), so `tune_v18` models write to
`inference_tests/tune_v18_NG_s*/` and `clamp/tune_v18_NG_s*/` without any writer knowing a
level exists. There is no ambient state to forget, and the skip-if-exists becomes safe
rather than dangerous, because a 0.6 tag can never find a 0.5 session.

A namespaced `EXPORTS` root was considered and rejected: `EXPORTS` is read-side only —
sessions are written through `cfg.export_path`, built from `export_folder` + `run_name` —
so a 0.6 run under it would have written into the 0.5 tree, been skipped, and had its 0.5
sessions re-analysed and filed under 0.6.

`HIER_SWITCH_LEVEL` (`hier_switch_group.LEVELS`) therefore only chooses **which tags to
read** and **where the group table and figures land**. Getting it wrong costs a few minutes
of regeneration and is obvious on sight.

| level | noise | NG tag | RNN tag | seeds | outputs |
|---|---|---|---|---|---|
| `n05` (default) | 0.5 | `v15` | `v17` | 10 | `group/` |
| `n06` | 0.6 | `v18` | `v19` | 20 | `group/n06/` |

Two guards, because a tag is a naming convention and not a check:

- `hier_switch_group.check_level` refuses a model whose training `pulse_noise_std` is not
  the level's, before it spends a session on it. Every `task` and every clamp task runs it.
- `hier_switch_perturb.level_tags` replaces the bare `os.listdir` over `clamp/` in both its
  callers, so a dropped seed's leftover cells — or a second noise level — cannot be pooled
  into one panel.

And one record: `session_report` carries each session's `pulse_noise_std`, and `aggregate`
refuses to build one table out of two levels rather than averaging them.

## Selection is relative to the ceiling

A fixed `steady >= 0.8` is a stricter test at a lower ceiling (0.896 of ceiling at σ 0.6
against 0.863 at σ 0.5), so the two levels would not be selected by the same rule.
`hier_switch_tune.arms` now uses `SELECTION_FRAC x CEILING[sigma]` with
`SELECTION_FRAC = 0.79`, and writes `selection.json`, which `hier_switch_group` reads — the
seed list is no longer transcribed by hand into two places.

At σ 0.5 the distribution is strongly bimodal and 0.79 is simply the midpoint of the empty
band:

| | steady accuracy | as a fraction of ceiling | Z d′ |
|---|---|---|---|
| the 6 discoverers | 0.816 – 0.930 | 0.884 – 1.008 | 2.02 – 3.12 |
| the 4 failures | 0.465 – 0.651 | 0.503 – 0.705 | 0.15 – 0.40 |

The gap is 0.179, so **any** coefficient in (0.705, 0.884) selects the same six. The
threshold is not discriminating finely; it is telling "discovered the contexts" from "sat
at chance". `arms v13` under the new rule selects exactly `(0, 1, 3, 5, 6, 9)`, which is
the regression check.

- **Z d′ stays absolute** at 1.5. It is a separation in Z units with no ceiling to scale
  by, and its margin is a factor of five. Scaling it would be inventing a normalisation.
- **The passive bar** is `0.81 x ceiling` = 0.748 at σ 0.5, i.e. the 0.75 it replaces.
- **The RNN uses the same fraction** (`hier_switch_figures.LEARNED`), so the two groups
  stop being selected by rules that drift apart as the noise rises. At σ 0.5 that is 0.729
  and it reclassifies none of the ten v17 seeds — but its margin there is asymmetric: the
  RNN's empty band is 0.507–0.760, so the bar clears the lowest learner by only 0.031. A
  σ 0.6 RNN seed landing between about 0.72 and 0.78 of ceiling is a case to look at rather
  than to let the rule decide.
- **If the largest gap ever falls below 0.10**, `arms --save` refuses and prints the
  distribution instead. If the separation goes away at a higher noise, no coefficient is
  defensible and that is the result, not something to pick a number through.

## One stage was dropped

`tune_v13` screened and `tune_v15` retrained the discoverers to save `model.pt`. Their
summaries are **bit-identical** on all six seeds (steady and Z d′ to four decimals): the
config defaults *are* v15's explicit 4000 passive / 5000 active, and `save_model` is popped
before the config is built so it consumes no RNG. `v18` therefore screens and saves in one
grid, and the retrain stage — 20 × ~6 min, and a step where the seed list was re-entered —
is gone. `v13` and `v15` are left frozen as the σ 0.5 record.

---

## Runbook

`run_tune.sh` takes the tag. The other three read `HIER_SWITCH_LEVEL` twice: once to size
the array, and once written into the submitted script, so the job cannot inherit a
different value than the array was sized from.

| # | stage | command |
|---|---|---|
| 0 | the paired probe (below) | `HIER_SWITCH_PROBE=1 ./hier_switch/run_sessions.sh` |
| 1 | screen **and save** NG, 20 seeds | `./hier_switch/run_tune.sh v18` |
| 2 | select the discoverers | `hier_switch_tune.py arms v18 --save` |
| 3 | train the RNN baseline, 20 seeds | `./hier_switch/run_tune.sh v19` |
| 4 | core sessions | `HIER_SWITCH_LEVEL=n06 ./hier_switch/run_sessions.sh` |
| 5 | clamp grid | `HIER_SWITCH_LEVEL=n06 ./hier_switch/run_clamp.sh` |
| 6 | aggregate and figures | `HIER_SWITCH_LEVEL=n06 …group.py aggregate`, `…figures.py`, `…figures.py story` |
| 7 | manipulations | `HIER_SWITCH_LEVEL=n06 ./hier_switch/run_manipulations.sh` |

Stages 4–6 answer the panel-d question; stage 7 is most of the cost, so it goes last.
Per-task times measured from `sacct` on the σ 0.5 jobs: NG screening 5–6 min, RNN training
4–5 min, core sessions 7–8 min, softmax clamp 3 min, sigmoid clamp 8–10 min, manipulations
**27–30 min** (not the 40–55 the script comment claims). Peak RSS 1.4 GB against a 2.8 GB
default, so no `--mem`. About 2.2 GB of disk for the level, on the lab data quota.

---

## The paired probe (stage 0)

`run_test` sets arbitrary keyword arguments onto the config and the dataset reads
`cfg.pulse_noise_std` live, so the forced triplet can be re-run on the **already trained**
σ 0.5 networks at a noisier cue with no code changes beyond naming the conditions.
Recorded as `softmax_n06_rc_{none,low,high}` under the existing `tune_v15_*` tags, read
back with `hier_switch_group.py probe`.

This is the one genuinely within-network comparison available: same weights, same seeds,
same stream. It asks whether the **inference mechanism** survives higher sensory noise at
fixed weights, which is *not* the question `tune_v18` asks — whether the phenomenon
re-emerges when a network is grown under noise. Kept out of `all_conditions`, so it can
never reach the group table or the figures.

### Result

_Pending: array 6592751._

---

## Results at noise 0.6

_Pending: v18 screening submitted as array 6592764._

What to look at, and what would count:

1. **The psychometric.** Does the curve drop to roughly the paper's 0.85–0.90 at conflict
   0.29 and 0.65–0.70 at 0.5? Projected 0.92 / 0.80 at σ 0.6.
2. **Panel d**, the reason for the run. The low- against high-conflict reversal split is
   0.084 at σ 0.5; expect 0.12–0.15.
3. **Yield.** How many of 20 NG seeds discover and 20 RNN seeds learn. Below ~6 discoverers
   every group statistic is in question and the run should stop and say so.
4. **What else moved.** The encoding table, the clamp dissociation and the manipulation
   dose–response were all measured at σ 0.5. Report differences rather than assuming they
   carry over.
5. **The RNN is the fragile one.** `RNN300`'s `WU_lr = 3e-3` and 250–350-trial blocks were
   tuned at σ 0.5 as the shortest blocks on which a plastic RNN commits instead of hedging.
   If it hedges at σ 0.6 that is a finding — report the undecided rate. A three-point
   `WU_lr` diagnostic on one seed says whether the learning rate is the cause; do not adopt
   a retuned block length or learning rate without checking in, because changing a second
   knob alongside the noise would confound the comparison.

---

## Deferred

Noise 0.8–0.9 (the gap optimum) pending the 0.6 result; any cross-noise comparison figure;
re-tuning `Z_lr`, the passive-phase length or the RNN's block length. All were fixed at
σ 0.5 and may be suboptimal there too, but changing them alongside the noise would confound
the comparison. Change one thing.

A third level is two `GRIDS` entries, two `COMMON` entries and one `LEVELS` row.
