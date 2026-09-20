# Hierarchical cue→rule reversal task (Lam et al., Nature 2024)

Paper: Lam, Mukherjee, Wimmer, Nassar, Chen & Halassa, *Prefrontal transthalamic uncertainty
processing drives flexible switching*, Nature 637:127 (2025), doi 10.1038/s41586-024-08180-8.

**Picking this up? Start with `docs/hier_switch_handoff.md`.** It has the roadmap, the current
state and the next steps. This file is the reference: task, model, full tuning log.

Code lives in `hier_switch/`. Every module there puts the repo root on `sys.path` and uses a
`hier_switch_` prefix, so nothing shadows a root module (`datasets`, `configs`).

| File | Role |
|---|---|
| `hier_switch_config.py` | `HierSwitchConfig`: task knobs, timing, channel map, latent settings |
| `hier_switch_dataset.py` | `HierSwitchDataset`, registered as `'hier_switch'`. Run it directly for a generator self-check |
| `hier_switch_train.py` | Train NG, RNN or Oracle; print summary numbers; draw `plot_logger_panels` |
| `hier_switch_analyses.py` | `extract_trials`, `summarize`: per-trial arrays and the numbers used to judge a run |
| `hier_switch_tune.py` | Tuning grids, one entry per SLURM array task, plus `collect` for a table |
| `run_tune.sh` | `./hier_switch/run_tune.sh <tag>` submits a grid as a SLURM array sized from the grid |
| `show_blocks.py` | Per-block table for finished tuning runs: did the model converge within each block? |
| `hier_switch_test_inference.py` | Load a saved model (`run(save_model=True)` → `model.pt`) and test its inference under named latent conditions, each on a copy, on the same test trials. It is also the session recorder: every condition writes a `session.npz` with the per-timestep outputs, the pulse frames, the hidden states and the recovered dL/dZ |
| `hier_switch_observer.py`, `hier_switch_hidden.py`, `hier_switch_perturb.py`, `hier_switch_group.py`, `hier_switch_figures.py` | The phase-2 analyses: ideal observer, hidden state (including the encoding table: what each signal carries), Z clamp, the group pipeline and the figure panels. See `docs/hier_switch_analyses.md` |
| `hier_switch_hooks.py` | Per-trial perturbations of the latent update — the paper's optogenetics. Default-off; run it directly for a self-test. See `docs/hier_switch_analyses.md` §6c |

`functions_and_utils.plot_logger_panels` has `hier_switch` branches for `behavior` and
`corrects`.

---

## The task

A cue of 16 sound pulses. 9 are informative, high-pass (HP) or low-pass (LP); 7 are white
noise (WN). The dominant type is the **cue**. The cue sets the **rule**, i.e. which modality
to attend. A visual and an auditory target appear on opposite sides, and the response is the
side of the attended one. The cue→rule mapping reverses without warning every 30–60 trials.

- **Cueing uncertainty.** Conflict = non-dominant / dominant count: 9:0 → 0, 8:1 → .125,
  7:2 → .29, 6:3 → .5, 5:4 → .8. It is drawn uniformly per trial.
- **Rule uncertainty.** Has the context reversed?

In the paper, switching is slower when the first reversal trials are high-conflict: an error
on an ambiguous cue is blamed on the cue rather than on the world.

## How it is modelled

One trial is one batch (`seq_len = stride = trial_len = 25`). The hidden state resets every
trial, so **Z is the only thing carried across trials**.

| frames | content |
|---|---|
| 0–15 | pulses: one-hot `[HP, LP, WN]` + Gaussian `pulse_noise_std` on the three channels |
| 16–17 | delay |
| 18–24 | targets `[vis, aud]`, ±1 with aud = −vis; response window |

`correct_side = vis × cue_sign × context_sign`. It is a three-way parity, so a reversal flips
the correct answer on every trial. Context 0 means HP → vision.

Channels (augmented-input trick). The labels travel through `logger.inputs` unmasked:

```
idx  0   1   2   3    4    5    6         7        8
     HP  LP  WN  vis  aud  cue  conflict  context  correct
in   1   1   1   1    1    0    0         0        0
out  0   0   0   0    0    0    0         0        1
```

- **Supervision.** The target is the correct side. In a two-choice task that is the same
  information as binary reward: an error says the other side was right. There is no context
  output and no context label in the loss, so NG has to discover the contexts.
- **Pulse noise.** It is what makes conflict a real sensory uncertainty. Without it, counting
  recovers the cue exactly at every conflict level.
- **Speed pressure.** `temporal_loss_weights` are 0 until the first output that has seen the
  targets (`response_start_timestep = target_onset + 1 = 19`). After that they are
  `exp(-0.178 i)`, an end-of-window weight of 0.41 as in the flanker task.
- **Decision.** The response output averaged over the response window. Correct = its sign
  matches `correct`.
- **Blocks.** Built inside the dataset: strictly alternating contexts, lengths uniform in
  `block_len_range` (30–60, the paper's). In the training stream `train_block_schedule`
  lengthens them (see Models). Each base-class "block" is one trial, the same pattern as
  `FlankerRandomTrialsConfig`.
- **Test stream.** The test phase draws from its own RNG stream (`data_stream = 1`, keyed on
  `(env_seed, data_stream)`), so it does not replay the training prefix.

## Models

All three models share one architecture.

| | Training | Test |
|---|---|---|
| **NG** | WU + LU from scratch, Z self-inferred | weights frozen, LU on |
| **RNN** | LU off; Z stays at `Z_init` | weights plastic (it has no other route to adaptation) |
| **Oracle** | Z = true context, one-hot | weights frozen, LU on (must infer Z) |

**Latent settings, and why:**
- **Softmax over 2 Z units at temperature 0.5, `Z_init = 0` (the uniform gate).**
  - The softmax's one degree of freedom is *which* gate. There is no overall gain for the
    latent update to escape into.
  - At temperature 1 the gate is too soft to use (v4).
  - `'none'` collapses to a dead gate, and a sigmoid turns Z into a gain (v2, v3).
- **Z starts at `Z_init`.** `hier_switch_train.build_model` sets it. The RNN keeps that value
  for good.
- **SGD on Z, `Z_lr = 1e4`, plain L2 weight decay `Z_decay = 3e-6`** (`Z_decay_mode = 'grad'`,
  identical under SGD to the optimizer's `weight_decay`).
  - The strength that matters is Z_lr·Z_decay = 0.03: each step is
    `Z ← Z − Z_lr·(grad + Z_decay·Z)`.
  - Adam would divide the gradient by its own running magnitude, and that magnitude is what
    should carry how much an error says about the context.
- **Steepness 0, pre-gating** (v7).
- **Training.**
  - Passive phase: 2 × 2000 trials, weights only, Z held at `Z_init`, one block per
    context.
  - Then 5000 active trials on 200-trial blocks, strictly alternating.
  - The test phase always uses the paper's 30–60.
  - This is the verified configuration (v13: 6/10 seeds discover; seed 0 reproduces
    exactly). A longer passive phase (2 × 3000) with 4000 active trials is an untested
    alternative.
- **Test-only knobs added in phase 2** (all default-off, so no earlier run changes):
`record_hidden` (per-timestep hidden states into `logger.hidden_trace`), `rt_threshold`
(0.5, the |decision| that counts as a response) and `reversal_conflict` /
`reversal_conflict_n` — the paper's controlled reversals, which force the first 5 trials of
every *test* block to 7:2 or 6:3. The forced level is drawn as usual and then overridden, so
the RNG stream is untouched: the forced and unforced sessions are paired trial by trial, and
the training stream never sees it.

**`WU_lr` = 1e-3 (Adam). The weight learning rate is not a useful lever here.** Lower
  (5e-4, v11) changed nothing. Higher (3e-3, v14) stopped every seed from learning the task at
  all. Leave it.

**Oracle → Z inference.** The Oracle trains with the true context as Z. `run_test()` then
takes a copy of the trained model, freezes the weights and optimises Z from the uniform gate
at any `Z_lr`, with the oracle gone. Because those weights are known to use a context gate,
the Z_lr sweep brackets the latent learning rate that can track the reversals.

`run_test()` patches the live Z optimizer as well as the config, because the optimizer never
re-reads `config.Z_lr` (flanker Gotcha 1).

## Running

```bash
.venv/bin/python hier_switch/hier_switch_dataset.py        # generator self-check
.venv/bin/python hier_switch/hier_switch_train.py NG 0     # model, seed
.venv/bin/python hier_switch/hier_switch_tune.py list v1   # tuning grid
```

Figures land in `exports/hier_switch/<run_name>/`:
- `panels_full.pdf` covers the whole run;
- `panels_train.pdf` shows the last 8 training blocks;
- `panels_test.pdf` shows the first 8 test blocks.

The panels, top to bottom:
- behaviour: one dot per trial, the response signed toward context 0's rule, grey = error;
- P(correct), MA(10);
- Z;
- dL/dZ.

## Tuning log

Seed 0 unless noted. Grids in `hier_switch_tune.py`, results under `exports/hier_switch/tune_<tag>/`.
`show_blocks.py <tag>` prints the per-block table (accuracy over the first 50 trials of each
block and over its last 20%, plus |decision| and Z).

**v1: paper blocks (30–60) from trial 1, `'none'`, Z_lr 1e3–1e5, pre/post gating.** Nothing
learns. NG, RNN and every Z_lr sit at chance. Oracle Z learns (0.89 overall; psychometric
1.00 → 0.62 from conflict 0 to 0.8), and it needs ~2000 trials to do so even with the context
handed to it.

Why: the answer is vis × cue × context. Averaged over blocks, the weights see no gradient for
the vis × cue product, and a 30–60-trial block is far too short to learn it inside one.

**v2: one 3000-trial block, then 200–300, then 30–60.** The RNN learns the first block (0.89)
and re-learns every 200–300 block through its weights (first 50 trials 0.04–0.30, the
perseveration the parity structure predicts; last 20% 0.84–0.96). But its output shrinks with
every reversal (|decision| 0.80 → 0.14), and at 30–60 it hedges at chance.

NG learns nothing, not even the 3000-trial block. The latent update drives raw Z to ~0
within a few hundred trials. While the output cannot predict the sign, shrinking the gate is
the cheapest way to cut squared error, and a zero gate silences the network and blocks the
weights' gradient. It is the flanker "flatten the gate" failure, taken all the way to a dead
gate.

**v3: long blocks only (3000, then 1000s).**

| run | learns? | what Z does |
|---|---|---|
| RNN | yes; re-learns each block, last 20% 0.86–0.94 | fixed |
| NG `'none'` Z_lr 10 | yes, same as the RNN | barely moves (inert) |
| NG `'none'` Z_lr 100 | no | → 0, dead gate |
| NG `'none'` + 3000 passive trials, Z_lr 100 / 1000 | learns in the passive phase (0.93) | collapses to ~0 at the first reversal once LU is on |
| NG sigmoid Z_lr 100 / 1000 | **yes**; re-learns each block, last 20% 0.85–0.94 | both dims rise together (1000: raw 0 → 2): a gain, not a context code |
| NG sigmoid Z_lr 1e4 | no | disrupts learning from the first block |

In every run that learns, the re-learning after a reversal is done by the weights, over
hundreds of trials. Z does not separate the contexts (d′ ≈ 0.2). The test phase (30–60
blocks, weights frozen) is at chance throughout.

**Where this leaves it.** The task is learnable, and a sigmoid gate keeps it learnable with
LU on. What does not yet happen is Z carrying the context. Every latent update so far has
moved along the gain direction (both dims together), because after a reversal "turn the
output down" always lowers squared error and "switch the mapping" only helps once the
weights already implement two mappings that Z can pick between.

A 2-way softmax has no gain direction: its one degree of freedom is *which* gate. That is
likely why context discovery worked with it in earlier tasks.

**v4: softmax (temp 1), blocks cut to 20% (600, then 200s), 8000 trials.** Nothing learns,
the Oracle included (0.51). Two separate problems:
- **A 600-trial first block is too short to learn in.** The one arm that kept a 3000-trial
  first block learned it (0.90), then re-learned 200-trial blocks with decaying gain (late
  0.93 → 0.6).
- **A softmax at temp 1 is too soft a gate to use.** It turns the oracle one-hot into
  [0.73, 0.27] vs [0.27, 0.73]. The Oracle's learning curve is identical to the RNN's, which
  has no context at all. v1's Oracle, with no activation (gate [1, 0] vs [0, 1]), learned in
  ~2000 trials.

**v5: sharper softmax.** The Oracle learns at temp 0.5, 0.25 and 0.1 (0.90 after ~2500
trials, with 600/200 blocks: it does not need a long block, since it is handed the context).

**Oracle → Z inference works.** After training, the oracle is removed, weights frozen, and
Z is optimised from uniform on the paper's 30–60 blocks:

| temp | best test Z_lr | steady (11+) | trials 1 / 2 / 3 / 4–5 / 6–10 after reversal |
|---|---|---|---|
| 0.5 | 1000 | 0.91 | 0.13 / 0.17 / 0.04 / 0.20 / 0.50 |
| 0.25 | 1000 | 0.81 | 0.17 / 0.17 / 0.13 / 0.17 / 0.43 |
| 0.1 | 300 | 0.89 | 0.22 / 0.17 / 0.09 / 0.20 / 0.44 |

- **Perseveration, then recovery.** There is clear perseveration, then recovery by 6–10
  trials; the animals switch after ~5 errors.
- **Z_lr range.** Too low (≤ 30–100) is too slow. Too high (≥ 3000 at 0.1, ≥ 1e4 at 0.5)
  thrashes back to chance. The usable Z_lr scales as 1/temp.

**Discovery (NG from scratch) still fails.** RNN and NG learn a 2000-trial first block
(~0.75) and lose it after the first reversal. With a 600-trial first block nothing learns.

The block table shows the mechanism. Z settles in one softmax corner (raw ±0.34, gate
[0.94, 0.06]) and stays in the *same* corner in both contexts, while the weights do the
re-learning until they hedge. The other corner's gate was never trained, so a reversal gives
Z no reason to go there. The contexts only differ through a sign flip of a three-way
interaction; nothing about the other gate is better until it has been trained for the other
mapping. The Oracle gets that training from the labels; discovery has to break the symmetry
itself.

Defaults now: `softmax_temp = 0.5`, `Z_lr = 1000` (the Oracle bracket).

**v6: discovery at the Oracle-bracketed latent (temp 0.5), a 3000-trial first block, then 200s.**
Still no discovery.
- **Z_dim 2 (Z_lr 300 / 1000 / 3000).** It learns the first block (0.90) and re-learns a few
  200-trial blocks, then hedges (|decision| → 0.05), exactly like the RNN. Z stays in *one*
  softmax corner for both contexts; at Z_lr 3000 it drives deeper into it (±1.07) and stays.
- **Z_dim 4.** It never learns even the first block (|decision| ≈ 0.02); not yet diagnosed.

**Why discovery fails here.** The contexts differ only by the sign of a three-way
interaction, so nothing about the *other* gate is better until it has been trained on the
other mapping. And with a soft gate, the units the other corner would favour are still
partly on (0.12 at temp 0.5), so they are trained on the current mapping too. The two
gate states never separate, and the weights settle into a hedge.

In the earlier tasks, the contexts differ in something the untrained network already
predicts badly (a mean, a location). That gives Z a gradient toward *some* other state
from the first reversal. This task offers none.

**v7: Oracle only. Steepness, gating site, and weight decay on Z.** One Oracle was trained
per gating site (pre / post), then the latent grid was tested on copies of it (30–60 blocks,
weights frozen):
- steepness 0 vs 2;
- Z_lr 300–1e4;
- weight decay on raw Z (toward 0, the uniform softmax gate).

**How weight decay is applied.** It is plain L2 weight decay on Z: `Z_decay` with
`Z_decay_mode = 'grad'`, which under SGD is identical to the optimizer's own `weight_decay`.
Each step is `Z ← Z − Z_lr·(grad + Z_decay·Z)`, so its strength is set by the product
`Z_lr·Z_decay`. A literal 0.3 at Z_lr 1000 diverges.

**Steepness barely matters.** 0 and 2 are within noise at matched settings.

**Weight decay sets the switch speed.** Pre-gating, steepness 0:

| Z_lr | Z_decay | Z_lr·Z_decay | overall | steady (11+) | crosses chance at trial |
|---|---|---|---|---|---|
| 1000 | 0 | 0 | 0.73 | 0.87 | 10 |
| 1000 | 3e-5 | 0.03 | 0.80 | 0.90 | 5 |
| 3000 | 3.3e-5 | 0.1 | 0.84 | 0.89 | 3 |
| 3000 | 1e-4 | 0.3 | 0.82 | 0.85 | 1 |

At the strongest decay (0.3), Z switches almost at once but cannot hold the context through
noisy trials.

**Pre-gating beats post.** Oracle training 0.895 vs 0.880; best overall at test 0.84 vs 0.82.
Post needs Z_lr 3000–1e4 to get there.

**Defaults now:** pre-gating, steepness 0, Z_lr 3000, `Z_decay = 3.3e-5`.

**v8: those settings in NG.**
- **The short start fails again.** On the 1200-trials-of-600-blocks schedule nothing learns,
  the RNN included, and every NG run matches the RNN block for block. The 3000-trial
  first-block arm learns block 1 (0.90), then re-learns and hedges as in v6. Three schedules
  now agree (v4, v5, v8): a model without the context needs ~1800 trials in one context to
  learn the task.
- **Z did not move.** In that arm (Z_decay 3.3e-5), raw |Z| stayed below 0.04. Early on the
  pooled gradient is ≈ mean(dL/dZ)/25 ≈ 4e-6 with a random sign, and at this decay that is
  not enough to move Z off the uniform gate. With Oracle weights the errors after a reversal
  are systematic, which is why the same decay works for inference.

  This does not show that decay is what blocks discovery: v6, with no decay, did not
  discover the contexts either.

**v9–v11: first discovery.** Block schedule: a 3000-trial first block, then 200-trial blocks.
Five seeds per setting; exact per-run settings are in `hier_switch_tune.py` (`GRIDS`).

**Discovery appears at Z_lr·Z_decay ≈ 0.03** (Z_lr 3000 with Z_decay 1e-5, or Z_lr 1e4 with
3e-6), in 2–3 of 5 seeds.
- **When it works:** Z flips with every block, and the output stays committed.
- **Test** (paper's blocks, weights frozen): 0.83–0.86 overall, 0.88–0.91 steady, crossing
  chance at trial 2–3. That is the same as Oracle → inference.
- **Other decay values:** 5e-6 discovered in 1 of 3 seeds, 2e-5 in 0 of 3, and no decay
  never.

**The outcome is binary.**
- **Discovery:** Z splits within the first 1–4 reversals after block 1, and the output stays
  committed (|decision| ≈ 0.8).
- **Failure:** Z never splits, and the weights hedge (|decision| ≈ 0.05).

Other levers:
- **Z_lr 1e4 splits earlier than 3000** (block 1–2 vs 3–4).
- **WU_lr 5e-4 changed nothing** (2 of 5).
- **A 4000-trial first block made the split harder** (1 of 5).
- **One seed never learned the first block in any setting.**

**v12: a passive phase instead of the long first block.** 3000 trials, weights only, Z held,
one 1500-trial block of each context; then 5000 active trials on 200-trial blocks.
- **The passive phase mostly fails to teach the task.** Z is frozen in the passive phase, so
  it is identical across the Z_lr / decay arms and its outcome is a seed property. In 1500
  trials of the first context most seeds reach 0.50–0.60, and the flip to the second
  context erases that partial progress. Only one or two of six seeds learn it; with the
  single 3000-trial first block (v10/v11), 4 of 5 seeds learned the task.
- **When the task is learned before Z moves, discovery is reliable at Z_lr·Z_decay = 0.03.**
  4 of 4 learners discovered. At 0.1, 1 of 2 did; at 0.01, 0 of 2. Discovering runs: test
  0.79–0.85, steady 0.86–0.90, crossing chance at trial 2–3.
- **The bottleneck is now learning the task in the passive phase.**

**v13: passive blocks lengthened to 2000** (2 × 2000, one per context), Z_lr 1e4 with
Z_decay 3e-6, uniform Z_init, 10 seeds.

**Result: 6 of 10 seeds discover the contexts.**
- **Passive phase: 7 of 10 learn the task.**
  - Six learn it within the first 2000-trial block.
  - One (s3) only learns it in the second.
  - Three (s2, s7, s8) stay at chance through both blocks. s2 has never learned the task in
    any grid.
- **Of the 7 passive learners, 6 discover the contexts.**
  - Z splits at the 1st–2nd active reversal.
  - The output stays committed (|decision| 0.71–0.87).
  - Test (paper's blocks, weights frozen): 0.75–0.85 overall, 0.82–0.93 steady, crossing
    chance at trial 3–4.
- **The seventh (s4) is partial.** Z splits at block 2 but the output half-hedges
  (|decision| 0.43): test steady 0.65, crossing at trial 8.
- **The remaining failures are all in learning the task itself, before Z is involved.**

**v14: v13 with `WU_lr` 3e-3 in both phases**, 10 seeds. Result: 0 of 10 learn the task, even in
the passive phase. Every seed sits at chance through both passive blocks and hedges in the
active phase (|decision| ≈ 0.02), including the six that discover at 1e-3. Faster weight
learning breaks the initial learning outright. `WU_lr` stays at 1e-3; 5e-4 (v11) made no
difference.

**Defaults after v14:** 2 × 3000 passive and 4000 active trials were tried as defaults, then
reverted before being run. The defaults are v13's verified configuration (2 × 2000 passive,
5000 active, 200-trial blocks, `WU_lr` 1e-3, softmax), so running them gives a working
model.

**Inference test: softmax off at test** (`hier_switch_test_inference.py`, v15 seed 0). The model
was trained with the softmax; at test, raw Z is the gate. Z started at [0.5, 0.5], the
softmax of the uniform start, and weight decay was off (it pulls raw Z toward the silent
gate).

| condition | test acc | steady | gate sum, 2nd half | outcome |
|---|---|---|---|---|
| softmax, as trained | 0.832 | 0.908 | 1.00 | chance crossed at trial 4 |
| no softmax, Z_lr 300 | 0.51 | 0.54 | 0.07 | first reversal handled; then the gain ratchets down and it hedges |
| no softmax, Z_lr 1000 | 0.51 | 0.52 | −0.42 | the gate collapses at the first reversal |
| no softmax, Z_lr 3000 | 0.29 | 0.29 | 19 | collapses, then blows up (outputs NaN) |
| no softmax, Z_lr 1e4 | — | — | NaN | diverges |

**The "which context" direction is available without the softmax.** At Z_lr 300 the model is
at 0.83 in block 1 and flips its gate at the first reversal: back to 0.80 within ~30 trials,
gate [0.28, 0.68].

**But every reversal leaks gain.** After a confident error, shrinking both gate units is the
fastest way to cut squared error, and the correct trials that follow never fully restore
it. The gate sum ratchets from 1.0 to 0.3 or below and the model hedges. What the softmax
contributes is the sum-to-one constraint, which removes the gain direction.

**Inference test: sigmoid in place of the softmax at test** (same model). Z started at 0, since
sigmoid(0) = 0.5 is the same starting gate. Weight decay was on: under the sigmoid it pulls
toward that middle gate, not toward silence.

| condition | Z_lr·Z_decay | test acc | steady | Z d′ | crosses chance at trial |
|---|---|---|---|---|---|
| softmax, as trained | 0.03 | 0.832 | 0.908 | 2.68 | 4 |
| sigmoid, Z_lr 1e3 | 0.003 | 0.53 | 0.57 | 0.35 | 7 |
| sigmoid, Z_lr 3e3 | 0.009 | 0.59 | 0.66 | 0.90 | 8 |
| sigmoid, Z_lr 1e4 | 0.03 | 0.61 | 0.65 | 1.30 | 4 |
| **sigmoid, Z_lr 3e4** | 0.09 | **0.67** | **0.71** | 2.02 | 3 |
| sigmoid, Z_lr 1e5 | 0.3 | 0.62 | 0.62 | 0.85 | 3 |
| sigmoid, Z_lr 3e4, Z_decay 1e-6 | 0.03 | 0.56 | 0.59 | 0.27 | 9 |

- **It never collapses.** Weight decay keeps the gate sum near 1.
- **At its best (Z_lr 3e4) it switches as fast as the softmax** (chance crossed at trial 3).
- **But it holds the context less well: 0.71 steady against 0.91.** The two Z units drift
  together as well as apart, so there are slips within blocks.
- **The gain comes from the stronger decay, not the higher Z_lr.** The same Z_lr at the
  trained decay strength (0.03) hedges (gate sum 0.61), and a higher Z_lr (1e5) thrashes.
- **Figures.** The comparison figure is at
  `exports/hier_switch/inference_tests/tune_v15_NG_s0/inference_summary.pdf`; each
  condition's `panels_full.pdf` and `panels_test.pdf` are in its own folder.
