# Three-Stage Cued-Context Curriculum on the Rotating-Targets Task

`rotation_curriculum_config.py` · `_sweep.py` · `_analysis.py`

A curriculum in three stages: the standard task with unsignalled context switches, then a stage
where the context is **cued**, then the cue is withdrawn again. The question is what the cued
stage leaves behind — and whether the answer depends on how fast the model's latent adapts.

This ports the Stage-1 / Stage-A / Stage-B protocol from
`../NeuraGEM_rl_objective/run_probabilistic_reversal.py` onto the
[rotating-targets task](task_rotating_targets.md), where context is a rotation angle rather than
a reversal of reward contingencies.

---

## The three stages

| | | `what_latent_to_use` | LU | WU |
|---|---|---|---|---|
| **S1** | standard task, uncued switches | `'self'` | on, at `Z_lr_train` | on |
| **S2** | **cued context** | `'context_ids'` | **off** | on |
| **S3** | uncued again, **weights frozen** | `'self'` | on, at `Z_lr_test` | **off** |

**The cue is not in the observation.** It is the ground-truth rotation injected as a one-hot into
Z, which multiplicatively gates the hidden state (`use_mul_gating`, `post_gating`). Nothing about
the input changes between stages, which is what lets the same model be carried through all three.

The oracle latent is built from `latent_inputs = context_ids[:, 1:]` — the context of the frame
being *predicted* — so at a block boundary the cue announces the **new** rotation before the
model has to predict under it. S2 therefore has no detection delay at all, not even the one-trial
floor the ideal observer pays. That is the intended reading of "cued": the context is given, not
merely hinted at.

Each stage exists for a reason, and the reasons are sequential:

- **S2 forces a Z→behaviour mapping into the weights.** With `no_of_steps_in_latent_space=0` the
  latent is *given*, never fitted, so the weights cannot treat Z as one more free parameter —
  the only way to profit from it is to read it.
- **S3 freezes the weights so Z is the only adaptive variable left.** Recovery speed after a
  switch is then a function of `Z_lr` and of nothing else. This is the whole point of the design,
  and it is what makes `Z_lr` a clean behavioural dial rather than one knob among many.

> **Why the cue rides Z rather than the input.** An input cue is available too — the rotation
> already sits in every observation when `enable_context_output` is on, hidden by
> `input_feed_mask`, so unmasking it during S2 would be a one-line change. It was not used,
> because a network given the answer on its input can route it through the recurrent weights and
> never engage the latent at all, which would make S3 a test of nothing in particular. Injecting
> the cue *as* the latent is what guarantees the weights learn the mapping S3 goes on to probe.
> The input-cue arm remains an obvious follow-up.

---

## No belief head

The [slips/perseveration experiment](rotation_slips_perseveration.md) reads context off an
explicit output head (`enable_context_output`), where it *is* the measurement. Here it would be
an intervention on the thing being studied: supervising the network to report the rotation
actively encourages exactly the context representation the curriculum is supposed to build or
fail to build. Measured on that experiment's own pilot, the head changed xy behaviour for both
non-oracle models — it is a supervised gradient, not an inert readout.

So `CONTEXT_OUTPUT_ENCODING = None`, `input_size = output_size = n_colors + 2 = 7`, and the
context belief is inferred from behaviour instead. The attack for colour `c` under rotation θ
sits at polar angle `2πc/n_colors + θ`, so the rotation the model is *acting on* is

```python
belief_rad = arctan2(pred_y, pred_x) - 2*pi*colour/n_colors
```

read off the predicted attack. `oi[t]` is the prediction *of* frame `ii[t]`, so the predicted
attack lands on the outcome frame and the colour that cued it on the preceding cue frame — one
judgement per trial, on the outcome frame, with no off-by-one.

This is not a new estimator: it is `trial_norm_errors` from
`rotating_targets_analysis._analyze_adaptation`, computed on the predicted coordinates against
the two candidate targets for the cued colour. The whole analysis is built on it — see
[the measurement](#the-measurement-switch-aligned-normalized-state-error).

`‖pred_xy‖ / target_radius` remains available as a commitment measure: a model hedging between
the two candidate targets predicts their midpoint, at `cos(sep/2) = 0.866` of the radius at
sep 60°.

---

## Prerequisite fix: `Z_decay` was applied twice

`Z_decay` went through **both** available paths on every LU step: the Z optimizer's own
`weight_decay` (`models._build_Z_optimizer`) *and* a gradient term in
`models._apply_chunk_lr_and_decay`, which runs unconditionally from the end of `adjust_Z_grads`
despite an older docstring claiming it was a no-op in standard runs. Under Adam both add
`decay * Z` to the gradient, so the realised decay was exactly `2 × Z_decay` and every tuned
value on disk meant half what it said.

`Config.Z_decay_mode` now selects the path:

| value | optimizer `weight_decay` | manual grad term | |
|---|---|---|---|
| `'grad'` | `0.0` | applied | **what this experiment uses.** Honours `chunk_l2_losses`, and means the same thing under Adam / AdamW / SGD |
| `'optimizer'` | `Z_decay` | skipped | coupled for Adam, decoupled for AdamW — the meaning changes with `Z_optimizer` |
| `'both'` | `Z_decay` | applied | the historical behaviour, and still the default so existing results stay reproducible. Marked deprecated |

The default was left at `'both'` deliberately: flipping it globally would silently change every
other experiment in the repo. Flipping it is a one-line follow-up.

**The Z_lr grid is calibrated on the total, not the field value.** The slips sweep's
`1e-3 * lr²` went through both paths, so its realised decay was `2e-3 * lr²`. This module
therefore states the calibrated total and divides by the mode:

```python
_effective_z_decay = lambda lr: 2e-3 * lr ** 2
def z_decay_for(lr, mode=Z_DECAY_MODE):
    return _effective_z_decay(lr) / (2.0 if mode == 'both' else 1.0)
```

which is why S1 at `Z_lr = 0.4` reproduces the existing head-off run **exactly** (see
[Verification](#verification)).

---

## Design

Task design — rotations, block structure, noise, gating — is imported from
`rotation_slips_perseveration_config.make_base_config` rather than restated, so the two
experiments cannot drift: `train_rotations = [0, 60]`, `block_size = 140`,
`block_duration_distribution = 'geometric'`, `rotation_block_order = 'random_no_repeat'`,
`latent_dims = [2]`, `latent_activation = 'softmax'`, `WU_lr = 0.001`.

### The stage tree

One job = one `(Z_lr_train, noise_std, seed)` cell, run entirely in memory:

```
S1  uncued, self-Z, WU+LU @ Z_lr_train                  1 run
 |__ for cue_mode in {'none', 'oracle_z'}:
       S2  from deepcopy(model_S1)                      1 run each
        |__ for z_lr_test in Z_LR_TEST:
              S3  from deepcopy(model_S2), W frozen     1 short run each
```

S1 does not depend on `cue_mode`, so the cued and control arms fork from **identical S1 weights**
— a paired control rather than two independent runs. Likewise every S3 fork within one arm shares
its S2 weights, so differences along the `Z_lr_test` axis are purely `Z_lr`.

### Grids

| Axis | Values |
|---|---|
| `Z_LR_TRAIN` (trait) | `'RNN'`, `0.05`, `0.2`, `0.6` |
| `Z_LR_TEST` (S3 fork) | `None`, `0.01`, `0.05`, `0.1`, `0.2`, `0.4`, `0.6`, `0.9` |
| `CUE_MODES` | `'none'` (control), `'oracle_z'` (cued) |
| `noise_std` | `0.20` |
| seeds | 1 (pilot) / 10 (full) |

The three numeric trait values are one from each region of the dose-response **already measured
on this task** at 15 values × 10 seeds
(`rotation_slips_perseveration_config.py`, the table above `_Z_LRS`): the transition (0.05), the
optimum (0.2), and the degradation limb (0.6). That is the main reason to expect this port to be
less finicky than the original — the grid is chosen from data rather than guessed.

Every trait value is also a fork value (asserted in the config), so the **diagonal** of the
`Z_lr_train × Z_lr_test` grid is the matched "trait" condition — each individual keeps its own
`Z_lr` throughout — while everything off the diagonal isolates `Z_lr` from the weights it
produced. Both readings come out of one set of runs.

Two arms carry most of the interpretive weight:

- **`Z_lr_train = 'RNN'`** — `no_of_steps_in_latent_space = 0` through S1 and S2, LU switched on
  in S3. A model that never developed latent inference and is then asked to use one.
- **`Z_lr_test = None`** — `no_of_steps_in_latent_space = 0` in S3: Z pinned. This is the arm the
  reference implementation had disabled (`add_control = False`) and it is load-bearing. If a
  model with Z frozen still tracks context, then something other than Z carries context across
  blocks and nothing downstream is interpretable.

### Stage lengths

`S1_LENGTH`, `S2_LENGTH`, `S3_LENGTH` and `PASSIVE_LENGTH` in the config are the source of truth
— they also feed `RUN_NAME`, so changing them writes to a new export directory and leaves the
previous run intact. At `block_size = 140` a stage of length *N* gives roughly `N / 140` blocks,
and the geometric duration draw makes the realised length ~1.16× nominal.

Two things to keep an eye on when shortening them:

- **S1 must be long enough to learn the task.** Sanity check (a) is exactly this; if S1's
  pre-switch error is not near 0 there is no flexibility to measure yet.
- **The last third of S1/S2 must still hold enough switches to average.** The figures print the
  switch count per curve beneath the axes. At `S1_LENGTH = 2000` the last third is only ~5
  switches, which is thin for one seed.

S3 is the cheapest stage to lengthen — its weights are frozen — and it is the one every fork is
read from, so it is the natural place to spend blocks.

`S2_BLOCK_SCALE = 1.0`. The reference shortened its cued stage's blocks (200 → 10 trials) "so
frequent reversals keep context/Z relevant" — so that within-block inference could not substitute
for the gate. Keeping block structure identical across stages instead keeps the trial axis
directly comparable between stages; if sanity check (b) fails, drop this to ~0.3 and rerun.

### Where Z starts in S3

`S3_Z_INIT` picks where the latent starts when S3 hands control back to self-inference:

- `'uniform'` — `1/Z_dim` everywhere, which is what the reference used and what the config
  currently sets.
- `'last_cued'` — the pre-softmax one-hot of the last S2 block's rotation. The oracle one-hot
  passes through `latent_activation_function`, so S2 only ever showed the weights gates at
  `softmax([1,0]) = [0.731, 0.269]`; a fresh uniform `[0.5, 0.5]` is strictly out of that
  distribution. Under `'random_no_repeat'` the last S2 rotation is uncorrelated with S3's first
  block, so about half the runs start in the wrong context.
- `'zeros'` — leave Z at zeros; softmax makes that uniform anyway, but skips the rebuild.

The uncued control gets whichever rule is set, identically, so the two arms enter S3 matched.

This only affects the **first** block of S3. The analysis restricts S1/S2 to their last third and
takes all of S3, so the initial transient is averaged over every later switch either way — which
is why `'uniform'` is a reasonable simplification rather than a confound.

---

## Why this should be less brittle than the original

The reference's own comments record the trouble: *"`Z_lr = 0.9` too high / `0.1` too low"*,
*"Takes 3e3. Waaay too long"*, *"for NG Z_lr=0.1, it is hard to train! fails just the same as a
fresh model"*, *"this extended training does weird things to converged NG"*. Several of the
causes are structurally absent here:

- **Dense supervised error, not sparse stochastic reward.** Z is driven by the same MSE that
  trains the weights. The reference used `z_loss_source='policy'`, an advantage-weighted log-prob
  whose gradient scale shifts with the policy and needed retuning whenever the loss source changed.
- **No hidden-state carry.** `config.stateful_hidden` defaults to `False` and the rotation
  configs leave it off, so `forward` re-initialises the hidden state every batch
  (`models.py`) and no carry channel leaks context across blocks. Turning it on would add
  one; don't, without re-running the curriculum comparisons.
- **No input augmentation.** The reference fed previous action and reward into the input, which
  competed with Z as a context route — and left them on despite its header saying otherwise.
- **The field names are the ones the model reads.** `Z_lr` / `Z_decay` here; in the RL repo those
  two attributes were silently ignored in favour of `Z_lr` / `l2_loss`.
- **The dose-response is already measured** on this exact task (see Grids).

What is *not* solved by any of this, and is what the pilot exists to check, is whether the cued
stage actually gets used — criterion (a) below.

---

## The measurement: switch-aligned normalized state error

The metric is the paper's, and the repo's — `rotating_targets_analysis._analyze_adaptation`
already computes it. For a cued colour, take the model's predicted attack and ask where it sits
between the two candidate targets:

```
norm_err = ||pred - target(current_rotation, colour)||
         / (||pred - target(current_rotation, colour)|| + ||pred - target(other, colour)||)

  0.0  predicting the correct target for this trial's rotation
  0.5  exactly between the two candidates — chance
  1.0  predicting the target the *other* rotation would put there
```

No belief head, no criterion runs, no perseveration or slip counting. Just where the behaviour
sits between the two rotations, trial by trial.

### The trial axis

A trial is a cue timestep plus its outcome timestep. Trials are numbered relative to the switch:

| x | what it is | what it tells you |
|---|---|---|
| `< 0` | earlier trials of the previous block | the asymptotic, pre-switch state — the model should be near 0 |
| `0` | the **last trial of the previous block** | the switch falls between 0 and 1 |
| `1` | the **first trial of the new block** | no evidence from the new block has been seen yet, so this is the model's *prior*. ~1.0 means fully committed to the old rotation |
| `2` | the second trial — a colour **not yet seen** under the new rotation | low here means the model generalised the rotation from ONE observation. This is the zero-shot reuse the task exists to test |
| `3+` | the rest | recovery rate, and where it settles |

Defaults are `n_pre=2` (so x runs from −1) and `n_post=14`, which is 2.8 mini-blocks at
`n_colors=5` — every colour appears at least twice. Mini-block ends are marked, because by x=5
every colour has been observed exactly once under the new rotation: anything below chance
*before* that line is transfer rather than experience.

`AnalysisParams.anchor` chooses what counts as "current" on a trial:

- `'trial'` (default) — the rotation actually in force. Pre-switch trials then read as the
  asymptote (near 0), the curve spikes at x=1 and decays. This is what makes x ≤ 0 informative.
- `'new'` — always the post-switch rotation, giving a single monotone "distance from the new
  context" curve with pre-switch near 1.

### Windows

Flexibility only means something once the task has been learned, so **S1 and S2 use the last
third of their blocks** (`AnalysisParams.block_frac`). S3 freezes the weights, so nothing is
learned during it and all of its blocks are used.

### Averaging, and adding seeds

Two levels, so more seeds do the right thing without any change:

1. within a run, mean over the switches in the window → one curve per seed;
2. across seeds, mean ± SEM.

With a single seed there is no across-seed spread, so the band falls back to SEM across switches
and `AdaptationCurve.sem_over` records which was used — the figures print it beneath the axes
rather than passing a switch-level band off as a seed-level one.

### Scalars read off a curve

`curve_summary()` returns `pre`, `t1`, `t2`, `reuse` (= t1 − t2), `mb1`, `mb2` and `asym`.
`reuse` is the headline one: how much of the map a single observation bought back.

`z_separation()` is the one latent diagnostic kept — `|mean softmax(Z)[:,0] under rotation A −
under rotation B|`, measured post-activation because the gate is what Z actually does to the
hidden state. 0 means the latent carries no context at all.

---


## Running it

```bash
# Stage 0 — pilot: PILOT = True in the config (the default), then
./submit_job.sh 3 curriculum          # 4 array tasks, ~25 min each
python rotation_curriculum_analysis.py

# Stage 1 — full sweep: PILOT = False
./submit_job.sh 39 curriculum         # 40 array tasks

# Stage 2 — analysis
python rotation_curriculum_analysis.py
```

One job runs 19 training runs, so `submit_job.sh` gives the `curriculum` experiment a 2-hour wall
clock while leaving the other sweeps on their 20-minute limit.

Running the analysis module directly first executes a synthetic self-test — a perfect prediction
must decode to the true rotation, a stale one must score as fully perseverative — with no trained
model involved.

### Eyeballing a single tree

`inspect_curriculum_run.py` draws one branch stage by stage with `plot_logger_panels`, which is
the fastest way to see whether a run did what it was supposed to:

```bash
python inspect_curriculum_run.py --list-runs                 # run directories on disk
python inspect_curriculum_run.py --train 0.2 --cue oracle_z --z-lr-test 0.2 --x2 6000
python inspect_curriculum_run.py --train RNN --list          # every stage in the tree
python inspect_curriculum_run.py --train 0.2 --run-name sep60_16800-8400-8400_head-off_decay-grad
```

`RUN_NAME` is built from the stage lengths, the head setting and the decay mode, so changing
`S1/S2/S3_LENGTH` repoints every reader at a new directory automatically — and the old run stays
on disk rather than being overwritten. Two consequences:

- **In a long-running IPython session, `%autoreload 2` will not pick this up.** It patches
  functions and classes but does not reliably re-execute module-level assignments, so `RUN_NAME`
  and `EXPORT_ROOT` keep their old values and `load_tree()` quietly reads the *previous* run.
  Call `refresh()` after editing the config. `load_tree()` prints the run directory and the stage
  lengths the tree was actually built with, so a mismatch shows up immediately.
- `run_name=` / `--run-name` reads a specific run regardless of the config — the way to compare a
  short pilot against an earlier long one. `list_runs()` shows what is on disk and marks the one
  the config currently points at.

```python
from inspect_curriculum_run import *
tree = load_tree(0.2)                                        # 'RNN' also works
show_stitched(tree, 'oracle_z', 0.2)                         # all three stages, one axis
show_curriculum(tree, cue_mode='oracle_z', z_lr_test=0.2, x2=6000)   # one figure per stage
show_stage(tree, ('S3', 'oracle_z', None))                   # any single stage
```

> **The curriculum stages are not phases.** `logger.phases` records only the phases *inside one
> `train_model` call*, and each stage is a separate call with its own logger — so nothing shows
> S1/S2/S3 together by default. S1 carries `['no inference learning', 'Learning and inference']`
> (the passive warm-up is 386 of 19722 timesteps, a sliver at the far left); S2 and S3 carry
> `['Learning and inference']` alone, because they warm-start from a trained model and configure
> no passive phase, and every stage runs `run_test_phase=False` so the Phase-3 names never
> appear. `show_stitched()` glues the three loggers end to end and rewrites `phases` so the
> **stage** boundaries are what `annotate_phases` draws. It is only valid because every stage
> shares `seq_len`, `stride` and `input_size`.

Usable panels: `rotating_targets_behavior`, `latent_2d` / `latent`, `loss`, `weights_grad_norm`.
`gradients` and `latent_effective_lr` read fields `compact_logger` drops and come out empty;
`context_belief` needs a belief head this experiment does not have.

> **The Z panel is flat and meaningless during a cued stage.** `logger.latent_values` records
> `model.Z`, the raw latent *parameter*. Under `what_latent_to_use='context_ids'` the gate is
> built from the ground-truth context inside the forward pass and never written back to
> `model.Z`, so the panel shows whatever S1 left there. A flat Z in S2 is **not** evidence that
> the cued stage lacks latent dynamics — the gate driving the network simply is not plotted.
> `show_stage()` prints a warning. (A flat Z in the `z_lr_test=None` S3 fork *is* meaningful:
> there the latent really is pinned.)

### Sanity checks

`check_acceptance()` prints these. Each is a way the headline comparison could be
*uninterpretable* rather than merely disappointing, so they gate reading anything else.

| | Criterion | Failure means |
|---|---|---|
| **(a)** | S1 pre-switch error ≤ 0.25 | the models never learned the task, so there is no flexibility to measure |
| **(b)** | S2 cued trial-1 error ≤ 0.25 | the cue is not being used. It names the *next* block's rotation before the model predicts, so a model reading the gate should show **no switch cost at all**; if it does, the weights are inferring context internally. Drop `S2_BLOCK_SCALE` and rerun |
| **(c)** | S3 frozen-Z asymptote ≥ 0.4 | something other than Z carries context across blocks, and nothing downstream is interpretable |
| **(d)** | S3 first-mini-block error spans ≥ 0.1 across `Z_LR_TEST` | the `Z_lr` dial does nothing — the thing that was finicky in the original |

### What the pilot showed (1 seed, noise 0.20)

Shape only — one seed, and the numbers move with the stage lengths. Three results are worth
recording because they are what the design was built to detect:

**Only one trait arm learns context inference on its own at these stage lengths.** In S1, where
each arm is on its own, `α_z^train = 0.2` is the only one that returns to the new rotation inside
the window; the RNN and `0.6` never leave the old one, and `0.05` is halfway:

| `α_z^train` | RNN | 0.05 | **0.2** | 0.6 |
|---|---|---|---|---|
| S1, second mini-block | 0.79 | 0.62 | **0.22** | 0.75 |
| S1, asymptote (last mini-block) | 0.71 | 0.47 | **0.22** | 0.72 |

This is why `HEADLINE_Z_LR_TRAIN = 0.2` and not `Z_LR_TRAIN[-1]`: the single-arm figures (F1, F2)
have to show an arm that actually works uncued, or the cued stage gets credit for fixing a
baseline that was broken for an unrelated reason. **Re-check this after changing the stage
lengths** — S1 is short here, and the slower arms may simply not have finished learning.

**The cue removes the switch cost entirely.** In S2 cued the curve is flat: trial-1 normalized
state error 0.14 against 0.68–0.86 for the matched uncued control, and no spike at all. That is
the intended reading of "cued" — the rotation is given before the prediction, so there is nothing
to infer.

**The cued stage installs a Z→behaviour map in a network that never had one.** The RNN arm's
*uncued* S3 sits at 0.49–0.51 — chance — at **every** forked `α_z` including `off`: with the
weights frozen and a latent the weights were never trained to read, there is no context channel
at all and turning LU on changes nothing. After the cued stage the same architecture shows a full
dose-response, and the zero-shot reuse index (trial 1 − trial 2) climbs with `α_z`:

| S3 `α_z` | .01 | .05 | .1 | .2 | .4 | .6 | .9 |
|---|---|---|---|---|---|---|---|
| trial 2 (reuse trial) | 0.85 | 0.83 | 0.74 | 0.64 | 0.38 | 0.30 | 0.30 |
| reuse (t1 − t2) | −0.01 | 0.02 | 0.08 | 0.15 | 0.41 | 0.44 | 0.38 |
| first mini-block | 0.82 | 0.77 | 0.62 | 0.46 | 0.37 | 0.40 | 0.42 |

Recovery speed is monotone in `α_z` up to ~0.4 and then flattens while the asymptote worsens —
the speed/stability tradeoff, with the weights frozen so it cannot be anything but `Z_lr`.

**The benefit is concentrated in the arms that need it, and the cued stage erases the
differences between arms.** In S3 cued at a matched `α_z` all four trait arms lie on top of one
another (first mini-block 0.461–0.464 across RNN / 0.05 / 0.2 / 0.6) — recovery depends only on
the `α_z` S3 is *running at*, not on what the model developed under. The corollary is that for
`α_z^train = 0.2`, which already had a working latent, S3 ≈ S1 and the cued stage adds close to
nothing; the whole effect lives in the impaired arms. F1 shows one arm and F3 shows the
contrast, so read them together.

> With one seed the bands are SEM **across switches**, and S1's last third holds only ~5 of them
> at these stage lengths. The figures say so beneath the axes. Adding seeds switches the band to
> across-seed automatically; nothing else changes.


## Figures

All paper-panel sized; see [figure_style.md](figure_style.md). Every curve figure carries the
same furniture: a chance line at 0.5, a solid line at x = 0.5 for the switch, dashed lines at the
mini-block ends, and the averaging note (`over seeds` vs `over switches`, and the counts) printed
beneath the axes rather than in a caption that can drift.

Colour convention: the *stage* is grey / cue-orange / NeuraGEM-blue for S1 / S2 / S3; the *dial*
(`Z_lr_test`) takes the plasma ramp; the *model class* (`Z_lr_train`) keeps the RNN's registered
green and gives the NeuraGEM arms a Blues ramp.

| | Content |
|---|---|
| **F1** | **The curriculum.** Switch-aligned adaptation at S1, S2 and S3, cued branch and control branch side by side. S1 is drawn in both panels — it is the shared origin both branches fork from, so any difference between the panels is what the cued stage did. Shows **one** trait arm (`HEADLINE_Z_LR_TRAIN`), so read it with F3 |
| **F2** | **S3 by the forked `Z_lr`.** Weights are frozen and every fork sees identical data, so the only thing separating these curves is the latent learning rate. The dose-response, in adaptation-curve form |
| **F3** | Every trait arm at S1 and at S3, so the curriculum's effect on each is visible against its own starting point |
| **F4** | The two scalars the curve is read for, against the forked `Z_lr`: **trial 1** (the prior — commitment to the old rotation, scored before any evidence) and **trial 2** (zero-shot reuse — a colour never seen under the new rotation) |
| **D** | Gate separation between the two rotations in S3. Does the latent actually flip? |

`summarize()` prints `pre / t1 / t2 / reuse / mb1 / mb2 / asym` for every (trait arm, stage),
with the switch count and which level the SEM came from.


## Implementation traps

All verified in this repo; each one fails silently rather than loudly.

1. **`train_model` never touches `model.config`**, which is a live reference read by the LU path
   and by `_build_Z_optimizer`. Set `model.config = cfg_stage` at every stage boundary.
2. **`_rebuild_Z_optimizer` only fires when Z's *shape* changes.** A new `Z_lr` is otherwise
   ignored and the old Adam moments keep driving the step size. `_hand_over()` does both halves.
3. **`no_of_blocks` is recomputed** from `blocked_phase_length / block_size` inside `train_model`.
   Set the length; setting the block count has no effect.
4. **`oracle_context_encoding` and `oracle_context_values` are cached on the model at `__init__`.**
   They must be right on the config that *builds* the model, even though S1 runs with
   `what_latent_to_use='self'` — nothing fails until S2. `make_base_config` asserts them.
5. **Every phase rebuilds its dataset from `default_rng(config.env_seed)`.** Without
   `STAGE_SEED_OFFSETS`, S2 and S3 replay S1's exact block and noise stream.
6. **`RNN_with_latent.__deepcopy__` used to fail on any model that had run LU.** It rebuilds from
   `config`, allocating Z at `config.seq_len`, while a model that has run LU had Z resized by
   `_ensure_Z_shape` to `seq_len - 1` (whenever `predict_first_frame=False`), so
   `load_state_dict(strict=True)` raised on `Z` and its `latent` alias. Fixed by matching the live
   shape first — the whole stage tree depends on `deepcopy`.
7. **Weight freezing is `no_of_steps_in_weight_space = 0`**, which still runs forward and
   backward and only skips `.step()`. S3 therefore still logs `predicted_outputs`, which is the
   entire behavioural readout.
8. `compact_logger` asserts the logger has no Phase-3 phase names; every stage runs with
   `run_test_phase=False`, so it passes. Keep it that way.
9. `save_models=False` everywhere — the model filename omits the condition and arms would
   overwrite one another.

---

## Verification

Run with `.venv/bin/python` (system python has no torch).

- **`Z_decay_mode` unit check** — with the prediction gradient zeroed and SGD, the realised decay
  is exactly `1×`, `1×`, `2×` `Z_decay` for `'grad'`, `'optimizer'`, `'both'`.
- **Self-test** (`python rotation_curriculum_analysis.py`) — synthetic, no trained model. Two
  reference models are built analytically from `get_target_positions`:
  - *perfect* (predicts the correct target for the rotation in force) must score **0 everywhere**;
  - *lag-one* (uses the rotation of the **previous trial**, so it is wrong only on the first trial
    of a block) must spike to **exactly 1.000 at x=1 and sit at 0 at every other position**. That
    is the alignment claim in full — the switch falls between x=0 and x=1, and x=2 is already
    inside the new block. It also pins `curve_summary`: `reuse = 1.0` and `mb1 = 1/n_colors`.
  - the same lag-one model under `anchor='new'` must read 1 for x ≤ 1 and 0 from x = 2.

  > An earlier version of this test used a globally-stale model (always predicting the previous
  > *block*'s target) and asserted it would be correct before the switch. It is not: with two
  > rotations alternating, block *b−2* has the same rotation as block *b*, so that model is wrong
  > on both sides of the switch. The lag-one model is the one that isolates the alignment.
- **Stage stitching** — tree shape, per-stage phase names, S3 weights frozen and `Z_lr`/`Z_decay`
  forked, stages seeing different data while S3 forks share theirs, the frozen-Z fork not moving
  Z while a live fork does.
- **Regression** — S1 at `Z_lr = 0.4`, `noise = 0.20`, seed 0 against the existing head-off cell
  `exports/rotation_slips/sep60_blocked_16800_geometric/NG$\alpha_z=0.4$/context_encoding-None_noise_std-0.2/results_seed-0.pkl`,
  scored with `analyze_block_switch_adaptation`. **Max difference 0.0000** across trial positions
  — the task design is genuinely shared, and the halved `Z_decay` field under `'grad'` mode
  reproduces the old effective decay exactly.


## Caveats

- Stage lengths, `S2_BLOCK_SCALE`, `S3_Z_INIT` and the grids are **not** all encoded in the export
  path (only the stage lengths, head setting and decay mode are, via `RUN_NAME`). Set
  `SKIP_EXISTING = False` after changing any of the others.
- Geometric blocks average ~1.16× nominal, so the realised stage lengths exceed the nominal
  figures. The analysis works per block, so this only matters if you quote a length.
- The `'RNN'` trait arm trains under a constant uniform gate, so its S1 and S2 are genuinely
  latent-free. Its S3 asks the weights to use a channel they were never trained to read; a
  failure there is a result, not a bug.
- Only one noise level is run by default. The slips-vs-noise causal figure from the
  [slips experiment](rotation_slips_perseveration.md) has no counterpart here until
  `DEFAULT_NOISE` is widened.

---

## See Also

- [task_rotating_targets.md](task_rotating_targets.md) — the task, block structure, context IDs
- [rotation_slips_perseveration.md](rotation_slips_perseveration.md) — the experiment this builds
  on; the belief head, the noise axis, and the measured `Z_lr` dose-response
- [algorithm_predictive_learning.md](algorithm_predictive_learning.md) — training loop, Z
  optimization, latent carryover
- [configs.md](configs.md) — base `Config` fields, including `Z_decay_mode`
- [figure_style.md](figure_style.md) — `FigSize` presets and colour conventions
