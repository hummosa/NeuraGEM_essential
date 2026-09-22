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

### The answer, up front

σ 0.6 was run in full. **It does what the calibration said it would, and that is not
enough.** The gap between the paper's two forced levels went 0.084 → 0.101, and training
under the noise bought nothing over merely testing at it. The reversal split that motivated
the run did not appear on the behavioural criterion; the latent-side criterion, already
solid, tightened onto the normative value. The yield halved and the RNN baseline stopped
working.

The useful part is the bound. An optimal reader's gap peaks near 0.12 at σ 0.8–0.9 and then
shrinks, and the model now tracks the observer rather than exceeding it, so **no noise level
reaches the paper's ~0.20 gap with 9 informative pulses.** The distance from Fig 1e is not
sensory noise alone, and the knob that would move the bound is the number of informative
pulses, not their variance — see the last section.

Read "Results at noise 0.6" for the numbers. Nothing in the σ 0.5 tree changed.

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

Six seeds, all paired. `hier_switch_group.py probe`:

| measure | σ 0.5 | σ 0.6 | change | sem | seeds w/ sign |
|---|---|---|---|---|---|
| accuracy, conflict 0.29 | 0.938 | 0.901 | −0.036 | 0.005 | 6 / 6 |
| accuracy, conflict 0.50 | 0.854 | 0.799 | −0.056 | 0.008 | 6 / 6 |
| **conflict gap (0.29 − 0.50)** | **0.084** | **0.103** | **+0.019** | 0.011 | 4 / 6 |
| steady accuracy | 0.878 | 0.848 | −0.030 | 0.003 | 6 / 6 |
| trials to switch, low early conflict | 4.09 | 4.45 | +0.36 | 0.144 | 5 / 6 |
| trials to switch, high early conflict | 4.30 | 4.68 | +0.39 | 0.246 | 5 / 6 |
| **switch cost (high − low)** | **0.212** | **0.235** | **+0.023** | 0.250 | 3 / 6 |

**The psychometric moves, the reversal split does not.** Accuracy falls at both
conflict levels in every seed, and falls more at the harder one, so the conflict gap
widens from 0.084 to 0.103. But the switch cost — the reversal latency after a
high-conflict start minus a low-conflict one — is unchanged: +0.023 trials against a
standard error of 0.250, with 3 of 6 seeds going each way. Both latencies rise by about
0.37 trials; their difference does not.

Two things worth keeping straight. The conflict gap's +0.019 is smaller than the
observer's own +0.034 over the same change, so the model's gap goes from *exceeding* the
observer's at σ 0.5 (0.084 against 0.061) to roughly matching it at σ 0.6 (0.103 against
0.095). And 0.103 undershoots the 0.12–0.15 projected from scaling the σ 0.5 excess.

**What this does and does not license.** It is the network's inference mechanism meeting a
harder cue at weights that were shaped by an easier one. It says the psychometric
responds to noise immediately while the reversal split does not, at fixed weights. It does
not say what a network *grown* under the noise does — the representation it learns is the
thing that might change, and that is the question `v18` / `v19` exist to answer.

---

## Results at noise 0.6

### Yield and selection (stage 1–2)

**6 of 20 seeds discovered the contexts, against 6 of 10 at σ 0.5** — the rate halves,
from 60 % to 30 %. Selected: `(1, 6, 9, 17, 18, 19)`, written to
`exports/hier_switch/tune_v18/selection.json`, which also carries the ceiling, both
coefficients, the separation below and the full 20-seed table.

The group is therefore the same size as σ 0.5's, and every group statistic rests on the
same n = 6. That was the stated floor, and the run sits exactly on it.

Splitting the yield by where a seed failed says more than the headline rate, because the
two phases fail for different reasons:

| | σ 0.5 | σ 0.6 |
|---|---|---|
| learned the task at all (passive phase) | 7 / 10 = 70 % | 12 / 20 = 60 % |
| **discovered the contexts, given it learned the task** | **6 / 7 = 86 %** | **6 / 12 = 50 %** |
| discovered the contexts overall | 6 / 10 = 60 % | 6 / 20 = 30 % |

Learning the task at all is only slightly harder. What falls is the **conditional** rate:
of the networks that learned the task, 86 % went on to discover the context structure at
σ 0.5 and half do at σ 0.6. The noise is not mainly making the task too hard to learn; it
is making the latent structure harder to find in a network that has learned the task. That
is the same decoupling the separation table below shows from the other direction.

**Steady accuracy has stopped separating discoverers from failures.** This is the more
interesting half, and it is what the boundary-separation check is for:

| criterion | lowest kept | highest cut | margin | / range |
|---|---|---|---|---|
| σ 0.5, steady / ceiling | 0.884 | 0.705 | +0.179 | 0.355 |
| σ 0.5, Z d′ | 2.019 | 0.398 | +1.621 | 0.545 |
| σ 0.6, steady / ceiling | 0.925 | 0.920 | **+0.005** | **0.011** |
| σ 0.6, Z d′ | 1.873 | 1.253 | +0.619 | 0.224 |

At σ 0.5 both criteria separate the two groups. At σ 0.6 only the latent one does. Seeds
12 and 0 reach 0.92 and 0.80 of ceiling with Z d′ of 1.25 and 0.76 — they do the task
well **without the latent separating the contexts**. The selection is still defensible,
but it is being made by Z, and should be described that way rather than by quoting an
accuracy threshold.

Worth saying plainly: at σ 0.5 "discovered the contexts" and "got good at the task" were
the same seeds, and that coincidence is what made the accuracy rule look sufficient. It
was never the criterion doing the work — Z d′ separated more cleanly at σ 0.5 too (0.545
against 0.355). Raising the noise pulled the two apart and made that visible.

### The headline: the conflict gap widened, the reversal split did not

Group table: `exports/hier_switch/group/n06/group.json`, figures in
`exports/hier_switch/group/n06/figures/`. n = 6 NeuraGEM seeds at each level.

**The accuracy gap between the paper's two forced levels went 0.084 → 0.101**, and that is
essentially all of what the higher noise bought:

| | @ conflict 0.29 | @ 0.50 | gap | steady / ceiling |
|---|---|---|---|---|
| σ 0.5, trained (v15) | 0.938 | 0.854 | 0.084 ± 0.019 | 0.951 |
| σ 0.6, **probe** (v15 weights) | 0.901 | 0.799 | 0.103 | — |
| σ 0.6, trained (v18) | 0.919 | 0.818 | **0.101 ± 0.009** | 0.958 |
| ideal observer, σ 0.5 → 0.6 | | | 0.061 → 0.095 | |

**Training under the noise bought nothing over merely testing at it** — 0.101 against the
probe's 0.103. The gap is set by the sensory noise, not by what the network learned under
it. And 0.101 undershoots the 0.12–0.15 projected from scaling the σ 0.5 excess, for a
reason the numbers make plain: at σ 0.5 the model's gap exceeded the observer's by 0.023,
and at σ 0.6 it exceeds it by 0.006. The model tracks the optimal gap and has lost the
margin above it. Since the observer's own gap peaks near 0.12 around σ 0.8–0.9, **the
paper's ~0.20 gap is not reachable by raising the noise in this task design.** Whatever
else separates the animals from the model here, it is not sensory noise alone.

Worth being clear that this is a negative result for the intervention, not for the model.
NeuraGEM held up: steady accuracy fell 0.878 → 0.856 but *rose* as a fraction of ceiling,
0.951 → 0.958, and its undecided rate is unchanged (0.151 → 0.149).

**The reversal split — the reason for the run — did not appear.** The behavioural
criterion went the wrong way, and the latent-side one, which was already solid, tightened
onto the normative value:

| P2, switch latency high − low early conflict | σ 0.5 | σ 0.6 |
|---|---|---|
| behaviour (first correct trial) | +0.212, 4/6 | **−0.049, 2/6** |
| Z side (when Z crossed to the true context) | +0.606, 6/6 | **+0.903, 6/6** |
| decided (output committed) | +0.659, 6/6 | +0.829, 5/6 |
| the ideal observer's own size | +1.074 | +0.854 |

At σ 0.5 the model's latent split undershot the observer's (0.606 against 1.074); at σ 0.6
it matches it (0.903 against 0.854). So the higher noise made the latent switching *more*
normative while leaving the behavioural read-out as flat as it was. That is consistent
with the account already in `docs/hier_switch_methods.md` — the behavioural criterion is
contaminated by chance sign flips when the output sits near zero, and it is the measure to
distrust — and it is now supported at two noise levels rather than one.

Everything else carried over. Every 6/6 row at σ 0.5 is still 6/6 (the conflict weighting
B3/B5, the gradient tracking ε_CW, the rule-decoding drop, the RT rows, the encoding
table, Z demixing). Two rows improved: `B5 P(tipped | error) falls with conflict` went
2/6 → 5/6 and the integration index 4/6 → 6/6.

### The RNN baseline collapsed

This is the one place where σ 0.6 broke something outright.

| | σ 0.5 (v17, n=10) | σ 0.6 (v19, n=20) |
|---|---|---|
| seeds clearing the learner bar | 6–7 of 10 | **2 of 20** |
| steady accuracy, pooled | 0.719 | 0.562 |
| **undecided rate, pooled** | 0.653 | **0.923** |
| undecided rate, the seeds that pass | — | 0.699 |
| switch latency (decided) | 75 trials | 148 trials |
| rule decoding at t = 16 | 0.603 | 0.556 |

It hedges on 92 % of trials, and the two seeds that clear the bar still hedge on 70 %. A
baseline of n = 2 hedging networks is too thin to put in a figure, and the NG-vs-RNN rows
at σ 0.6 should be read as "the baseline does not do this task at this noise" rather than
as a comparison of two working models.

Before treating that as a fact about the task: `RNN300`'s test `WU_lr` of 3e-3 and its
250–350-trial blocks were tuned at σ 0.5, as the shortest blocks on which a plastic RNN
commits. `hier_switch/rnn_wu_probe.py` runs one trained seed at several rates to say
whether the rate is the cause.

**It is not.** `rnn_wu_probe.py v19 6`, against the σ 0.5 sweep in
`docs/hier_switch_handoff.md` §5d (same seed index, same 300-trial-trained starting
weights):

| test `WU_lr` | σ 0.5: acc / undecided | σ 0.6: acc / steady / undecided / \|dec\| |
|---|---|---|
| 1e-3 | 0.68 / 0.83 | 0.539 / 0.545 / 0.959 / 0.117 |
| **3e-3** (the setting in use) | **0.89 / 0.36** | **0.633 / 0.646 / 0.760 / 0.308** |
| 1e-2 | 0.51 / 1.00 | 0.494 / 0.492 / 0.999 / 0.053 |
| 3e-2 | 0.48–0.52 | 0.494 / 0.494 / 0.772 / 0.339 |

**The optimum is in the same place at both noise levels, and the whole curve has dropped.**
3e-3 is the unique best on accuracy at σ 0.6 as it was at σ 0.5, with 1e-3 and 1e-2 worse
on either side; peak accuracy fell 0.89 → 0.63 and hedging roughly doubled, 0.36 → 0.76.
3e-2 reproduces the thrashing signature the σ 0.5 log recorded at 1e-1 — undecided drops
to 0.77 and |decision| rises to 0.34, but accuracy sits at chance, so it commits and
commits wrongly, which is worse than hedging.

So the baseline's failure at σ 0.6 is a fact about the task at this noise, not a stale
hyperparameter, and **the rate should not be retuned**: there is nothing better to move it
to. The honest statement is that a plastic-weight RNN on 250–350-trial blocks does not do
this task at σ 0.6. Whether some *other* baseline could — a longer block, or the stateful
feedback-RNN in `docs/hier_switch_handoff.md` §7.4 — is a separate question and a model
change.

_Stages 3–6 done; stage 7 (manipulations) pending._

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

## What was checked

- **The noise-0.5 tree is untouched.** With the whole noise-0.6 tree on disk — its
  sessions, its 120 clamp cells per activation, its group table — `group.json` and all 24
  figures regenerate **byte-identically** to a snapshot taken before any of this was
  written, with `SOURCE_DATE_EPOCH` pinned so the comparison is exact. That is the check
  worth repeating whenever a level is added; the two figure runs are about a minute each.
- **The levels cannot pool.** With both sets of clamp tags present, the n05 reader returns
  120 cells from six `tune_v15_*` tags and the n06 reader 120 from six `tune_v18_*`. Before
  `level_tags`, both callers took everything `os.listdir` returned.
- **A mismatched model is refused.** `check_level` on a `tune_v18` model under `n05` exits
  with the noise it found against the noise the level wants, before recording anything.
- **Sessions carry their noise.** `load_session(...)['meta']['pulse_noise_std']` is 0.6 for
  the v18 sessions and still 0.5 for the v15 ones, so the ideal observer recalibrates per
  session rather than assuming a ceiling.
- **The selection rule reproduces the old one.** `arms v13` under the relative rule selects
  exactly `(0, 1, 3, 5, 6, 9)`, and the relative `LEARNED` reclassifies none of the ten v17
  seeds.
- Self-tests pass: `hier_switch_analyses.py`, `hier_switch_hidden.py`,
  `hier_switch_dataset.py`, `hier_switch_observer.py` (all five ceilings),
  `hier_switch_hooks.py`.

## What the 0.6 result says about going further

**Noise 0.8–0.9 is not worth running for the gap, and the σ 0.6 numbers are why.** A third
level costs two `GRIDS` entries, two `COMMON` entries and one `LEVELS` row, so the
machinery is not the obstacle — the argument is:

- The observer's gap at σ 0.8–0.9 is 0.118–0.120 against 0.095 at σ 0.6. At most another
  0.025, and the model now tracks the observer rather than exceeding it, so expect roughly
  0.12 where the paper shows ~0.20.
- Yield already halved at σ 0.6 (conditional discovery 86 % → 50 %). At σ 0.8 the screen
  would likely have to run 40+ seeds to land six discoverers, and the group would be
  selected from a thinner and more atypical tail.
- Steady accuracy already stopped separating discoverers at σ 0.6. Further out the
  selection rests on Z d′ alone, which is a weaker basis for a group than two agreeing
  criteria.
- The RNN baseline is already gone at σ 0.6, so there would be nothing to compare against.

**The more promising direction is the one the calibration points at.** The gap is bounded
by the *number of informative pulses*, not by their noise: with 9 informative pulses an
optimal reader cannot lose more than ~0.12 between conflict 0.29 and 0.5 at any noise.
Fewer informative pulses moves that bound directly, and is the knob to calibrate next if
closing the distance to Fig 1e matters.

It is not a one-line change, though. `n_informative` (`hier_switch_config.py:71`) is
coupled to `conflict_counts` by an assertion that every pair sums to it with a strict
majority (`:313`), so a smaller value needs a new ladder — and the paper's 7:2 and 6:3 do
not exist below 9, so the forced levels would stop being the paper's and would have to be
labelled as ours. `hier_switch_observer.ceiling` takes the config's ladder as given, so
calibrate the candidate ladders first, exactly as this level was calibrated, and decide on
the observer's gap before training anything.

## Deferred

Re-tuning `Z_lr`, the passive-phase length or the RNN's block length. All were fixed at
σ 0.5 and may be suboptimal there too, but changing them alongside the noise would confound
the comparison. Change one thing.

Retuning the RNN's test `WU_lr` is **settled, not deferred**: the diagnostic above shows
3e-3 is still the optimum at σ 0.6, so there is nothing to move it to.

A cross-noise comparison figure: nothing here needs one yet, because the interesting
numbers are single values per level rather than curves. The tables in this document are
the comparison.
