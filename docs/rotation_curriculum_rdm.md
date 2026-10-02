# How a Context Representation Develops Across Stage 1 — RDMs on the Curriculum

`rotation_curriculum_rdm.py`

> **Start with [rotation_curriculum_rdm_summary.md](rotation_curriculum_rdm_summary.md)** —
> one page on what was run, what is plotted and what it shows. This file is the detail.

The model-side companion to the planned human fMRI study. One pattern per trial, a
representational dissimilarity matrix (RDM) over trials sampled across the whole of S1, and a
summary index read off that matrix as a function of training. Every step after "one pattern per
trial" is metric-only, so the same analysis runs on voxels unchanged — which is the point of
running it on the model first.

This differs from [rotation_geometry.md](rotation_geometry.md), which asks *what shape* the
rotation code has across many angles in a frozen test phase. Here there are only two contexts and
the axis of interest is **training time**.

---

## Why a checkerboard is the prediction

`train_rotations = [0, 60]` with `rotation_block_order = 'random_no_repeat'`: with two values
those are the same thing, so blocks **strictly alternate** and every block boundary is a real
switch. Order the sampled trials chronologically and a model that holds context has low
dissimilarity within a block and high dissimilarity across adjacent blocks — a checkerboard whose
contrast should grow as the representation develops. A model that does not hold context is flat.

---

## What is measured

### Cue frames, not outcome frames

At the outcome frame the attack `(x, y)` is in the input, so the rotation is readable from the
*current stimulus*; a checkerboard there is partly stimulus-driven. At the cue frame only the
colour one-hot is present, so the rotation has to come from held context — Z, or the recurrence.
`frames='cue'` is the default and the claim; `frames='outcome'` is the contrast.

**This is a design requirement for the human study**: the cue and outcome periods have to be
separable, or the measured RDM carries the stimulus.

### Windows, mini-block aligned

A mini-block is `n_colors = 5` trials and holds every colour exactly once. Hidden activity depends
on the current colour, so a window that is not a whole mini-block carries a colour imbalance into
the matrix. Two windows per block, 10 trials total:

| | |
|---|---|
| `early` | the 2nd mini-block — past the switch transient, context still fresh |
| `late` | the last complete mini-block — settled |

Trials are ordered **by colour within each window**, so residual colour structure appears as a
fine 5-row pattern, visually distinct from the block-scale checkerboard.

Mini-blocks are recovered from the colour sequence itself (`_miniblock_phase`), not by arithmetic
off the block start. That matters: every block boundary from an `llcid` change in the logged stream
is exact, but the **first** one is the phase start read from `logger.phases`, which is counted
differently and lands three trials into a mini-block (measured: S1 seed 0 needs offset 3 for block
0, offset 0 for every later block). Detecting the offset makes every window colour-balanced and
fails loudly instead of silently returning a window with one colour twice and another missing.
The first block is dropped anyway — its start is the phase boundary, not a context switch, so
"early in the block" does not mean the same thing there.

At `S1_LENGTH = 2000` this gives **13 blocks × 10 trials = a 130 × 130 matrix**, every block
present, no subsampling. The diagonal is training time.

### The index, and the drift problem

`context_modulation` returns two numbers per sliding window of 4 consecutive blocks (4 gives two
blocks of each context and block lags 1, 2, 3). Both exclude same-block pairs — trials inside one
block are temporally adjacent and would be close for reasons that have nothing to do with context.

```
cmi     (mean between-context − mean within-context) / mean overall, over cross-block pairs.
        What an fMRI paper reports, and what transfers directly.

cmi_dc  ((d1 + d3)/2 − d2) / overall, the parity contrast at block lags 1, 2, 3.
```

`cmi_dc` exists because under strict alternation **same context is exactly even block lag**, so
any context contrast is also a lag-parity contrast — and a slow drift in activity across training
is a smooth function of lag that lands in the same place. Because `(1 + 3)/2 = 2`, the parity
contrast cancels any *linear* trend in lag by construction and suppresses smooth ones.

Read `cmi` as the headline and `cmi_dc` as the control. Where they disagree, believe `cmi_dc`. On
Z they agree to within 0.005 at every arm, so drift is not contaminating that measurement — but
`Z_lr = 0.01`'s matrix is visibly a smooth ramp rather than a checkerboard, which is the failure
mode the control is there to catch.

> **Sign of the drift bias.** Within-context pairs sit at larger block lags on average (2.0 vs
> 1.5), so pure drift pushes `cmi` *negative*. Confirmed in the self-test: drift-only synthetic
> data gives `cmi = −0.029`, `cmi_dc = 0.03`. The bias is conservative, not inflationary.

### What `cmi` cannot say, and where that is fatal

**An RDM is invariant to a global A<->B relabeling of the contexts.** Under strict alternation
"still holding the previous block's context" *is* a global relabeling — so a reliably
perseverative model and an instantly correct one produce the **same** `cmi`. Verified on
synthetic data with a known held context:

| | `cmi` | directional index |
|---|---|---|
| instant correct inference | 0.714 | 0.092 |
| perfect perseveration | 0.699 | **0.910** |

A third reading is open too. The switch surprise is context-specific — A->B and B->A move the
targets in opposite directions, and the transition is determined by the new context — so a
checkerboard at mini-block 0 can also be the error signal alone. Three incompatible
explanations, one number.

This is **harmless in the late window** (a model that held the wrong context for a whole block
would show a pre-switch `norm_err` near 1, and every arm measures 0.145-0.253) and **fatal at
mini-block 0**. So `cmi` is reported as *clustering* — is the representation organised by block
at all — and never as context identity. Identity needs the directional index below.

### Guardrails

`build_rdm` z-scores features across the **whole sampled trial set, pooled**. Keep it that way:
z-scoring within block or within window would subtract out the context offset itself. Nothing
mean-centres per block and nothing regresses time out before building the RDM — drift is handled
by the local window and by `cmi_dc`, not by pre-whitening.

---

## What the existing runs show (Z, 8 arms × 10 seeds, `zgrid1`, cue frames)

Z is already saved in every curriculum export (`latent_values` survives `compact_logger`), so this
needed no new compute. Z is 2-D softmax — it is the model's *explicit* context variable, so treat
it as a strong positive control rather than the fMRI analogue.

| `Z_lr` | RNN | 0.01 | 0.05 | 0.1 | **0.2** | 0.4 | 0.6 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| S1 `mb2` (behaviour, lower better) | 0.772 | 0.771 | 0.725 | 0.372 | **0.202** | 0.477 | 0.689 | 0.764 |
| `cmi` early | 0.000 | 0.157 | 0.648 | 0.646 | **1.067** | 0.767 | 0.173 | 0.011 |
| `cmi` late | 0.000 | 0.286 | **1.163** | 0.978 | 0.761 | 0.266 | 0.020 | −0.009 |

**A clear inverted-U, peaking where the behaviour peaks.** The `early` index peaks at
`Z_lr = 0.2`, the same arm the behavioural dose-response optimises at, and across all 80
(`Z_lr`, seed) cells it correlates with behavioural recovery at **ρ = −0.66** (negative = more
context modulation, less error).

**The early and late windows do not agree.** The `late` index peaks instead at `Z_lr = 0.05`
(1.163, the highest in the grid) — an arm whose behaviour is near chance (`mb2` 0.725). Its
correlation with behaviour is weaker (ρ = −0.44). The behavioural scalar is read a few trials
*after* a switch, so `early` is the matched comparison.

> **An earlier version of this document read that dissociation as "a slow latent converges within
> the block but too late to be used".** That was a hypothesis, and the directional index has since
> shown the mechanism is the *opposite* one — see
> [the settled answer](#what-the-directional-index-shows-and-the-005-case-settled). The relabeling
> invariance above is exactly why the wrong reading was available: a high early `cmi` is equally
> consistent with prompt commitment to the **wrong** context.

**For the fMRI, the design implication is to sample early in the block *and* use a directional
measure there.** An index built from settled, late-in-block trials would have ranked
`Z_lr = 0.05` best; a label-invariant index read early would have called its perseveration
"fast context inference".

### Controls

| | |
|---|---|
| `'RNN'` arm | `no_of_steps_in_latent_space = 0`, so Z is exactly constant and its RDM is identically zero. Verified. The empirical floor. |
| Permutation null | the block→context map is shuffled, keeping block structure intact — trials within a block are not exchangeable, so a trial-level shuffle would give a null far tighter than the data's own noise and every arm would clear it. |
| `Z_lr = 0.9` | `cmi = 0.007`, inside its null `[−0.024, 0.024]` — the analysis returns "no" when there is nothing there. |
| Colour balance | a colour-only synthetic effect scores `cmi = 0.000` (self-test case 4). |
| `cmi` vs `cmi_dc` | agree to 0.005 on Z, so the measured effect is not drift. |

`reliability` in `summarize_rdm` is the split-half reliability of the curve's **shape**, not of the
effect. A flat-but-large index has no shape to recover and lands near 0 or below; read it beside
`cmi`, never instead of it.

---

## The separation front: mini-block since switch x training block

`sweep_miniblocks` / `plot_separation_front`. Every mini-block is already in the sampled data and
every one is a full colour permutation, so this axis costs no new compute — `EARLY_MINIBLOCK` was
one module constant and the window spec now takes any selector.

The motivation: late-block separation saturates early for the well-tuned arms, so what keeps
developing is how *fast* separation is reached after a switch. Measured on Z (`cmi`, mean over 10
seeds, training block 2.5 -> 11.5):

| arm | what the map shows |
|---|---|
| `0.05` | **non-monotone in mini-block**: brightest at mb0 (0.67 -> 1.10, the highest value in its own map), a dark band at mb2 (~0.1), then ~1.1 from mb4 on. A trajectory *passing between* two attractors, not slow monotone convergence |
| `0.1` | mb0 rises across training, 0.06 -> 0.65, while mb1 falls 0.85 -> 0.56 — the front does move toward the switch |
| `0.2` | mb0 flat at ~0.00, mb1 already 0.91 at block 2.5. Nothing left to develop |
| `0.4` | mb1 rises 0.12 -> 0.85 while `late` stays low (0.00 -> 0.30) — separates early in a block and degrades by the end of it |
| `0.6`, `0.9` | flat near zero everywhere |

**The front does advance toward the switch — but `cmi` cannot say what is advancing.** For `0.1`
the growing mb0 clustering reads naturally as "faster context inference"; the directional index
shows it is the opposite. This is the single strongest reason not to report `cmi` alone on this
axis.

---

## The directional template index

`template_index` / `analyze_templates` / `plot_template_index`.

```
d_new = dist(x, settled(context of this block))
d_old = dist(x, settled(the other context))
index = d_new / (d_new + d_old)          0 = switched, 0.5 = ambiguous, 1 = still the old context
```

It is the representational twin of `norm_err` on the same 0 / 0.5 / 1 scale, which is what lets
the two be plotted on one axis. It is **not** label-invariant, so it resolves the perseveration
ambiguity, and the ratio form partly cancels a perturbation that pushes activity away from *both*
templates — which a difference-of-means contrast like `cmi` cannot do.

### Estimating `settled(context)` — causal and adjacent

`settled_old` is the reference (`late`) window of block **b-1**, `settled_new` the reference window
of block **b-2**, which carries the same context as b under strict alternation. Scoring a block's
early trials against its *own* late window would be circular: they share weights, drift point and
noise realisation, so `d_new` would come out low whether or not the context was ever inferred.

Three reasons this particular choice, over leave-one-block-out or an average of several blocks:

1. **Symmetric.** The index is a ratio of distances to two templates, so both must be estimated
   the same way. Averaging n blocks shrinks template noise from s^2 to s^2/n and systematically
   reduces distances to that template, so an asymmetric pair biases the ratio. One block each is
   symmetric by construction.
2. **Causal.** The index is read against training block, so a template drawn from *later* blocks
   would leak the very axis under study.
3. **Exact, not a proxy.** Perseveration means "still in the state block b-1 left me in", and
   late(b-1) *is* that state rather than an estimate of a population template.

The cost is the two leading blocks (they score NaN) and a template built from 5 trials.

### `zscore=False`, unlike the RDM

Per-feature z-scoring equalises informative and uninformative dimensions, so a feature carrying no
context signal has its *noise* scaled up to unit variance — which compresses the index toward 0.5
and costs exactly the scale that makes it comparable to `norm_err`. On synthetic data with a known
held context: raw gives 0.910 / 0.092 (perseverative / correct), z-scored gives 0.746 / 0.249. On
the 2-D softmax Z the two agree to three decimals; the difference is expected to matter on the
64-unit hidden state. Both compress rather than flip, so the reading against 0.5 is robust either
way, and `zscore=True` is available.

### What the directional index shows, and the 0.05 case settled

Z, 10 seeds, cue frames, all blocks. 0 = at the new context, 1 = still in the old one.

| `Z_lr` | mb0 | mb1 | mb2 | mb3 | mb4 | late |
|---|---|---|---|---|---|---|
| `RNN` | *not scoreable — Z is constant, so both templates coincide* ||||||
| 0.01 | 0.824 | 0.722 | 0.635 | 0.570 | 0.529 | 0.409 |
| **0.05** | **0.888** | 0.646 | 0.406 | 0.249 | 0.178 | 0.156 |
| 0.1 | 0.729 | 0.271 | 0.185 | 0.204 | 0.194 | 0.209 |
| 0.2 | 0.486 | 0.266 | 0.267 | 0.266 | 0.255 | 0.299 |
| 0.4 | 0.459 | 0.399 | 0.414 | 0.406 | 0.409 | 0.443 |
| 0.6 | 0.490 | 0.493 | 0.497 | 0.493 | 0.478 | 0.499 |
| 0.9 | 0.497 | 0.508 | 0.499 | 0.510 | 0.498 | 0.492 |

**`Z_lr = 0.05` reads 0.888 at mini-block 0 — near 1, not near 0.5.** It commits promptly and
confidently to the *previous* block's context, then corrects over the following four mini-blocks
(0.888 -> 0.646 -> 0.406 -> 0.249 -> 0.178). That is perseveration followed by correction, not
slow ambiguous convergence: the two stories predicted ~1.0 and ~0.5 at mb0 and the measurement is
unambiguous.

Across training its mb0 index *rises* (0.863 -> 0.902), as does `0.1`'s (0.692 -> 0.751). **What
advances toward the switch as training proceeds is perseveration, not inference** — the
representation becomes more decisively committed to the old context at the moment of the switch,
presumably as the context code itself sharpens. Read the separation-front map for `0.1` again with
that in hand.

Six regimes fall out, and they order the way behaviour does:

| | |
|---|---|
| `RNN` | no representation at all (index undefined) |
| `0.01` | perseverates and never fully corrects (0.824 -> 0.409) |
| `0.05` | perseverates, then corrects within the block (0.888 -> 0.156) |
| `0.1` | brief perseveration, corrects inside one mini-block (0.729 -> 0.271) |
| `0.2` | does not perseverate — already ambiguous at mb0 (0.486), then moves (0.266) |
| `0.4`-`0.9` | never settles at either template (0.40-0.51 throughout) |

### The scales really do match, and where they part

Because the index shares `norm_err`'s scale, the two can be compared number for number. The
alignment is off by one name and it matters: behavioural `curve_summary` calls trials
`1..n_colors` **`mb1`**, which is mini-block index **0** here (`behavioural_by_miniblock` handles
this).

| `Z_lr` | repr mb0 | behav mb0 | repr mb1 | behav mb1 |
|---|---|---|---|---|
| 0.01 | 0.824 | 0.818 | 0.722 | 0.771 |
| 0.05 | 0.888 | 0.846 | 0.646 | 0.725 |
| 0.1 | 0.729 | 0.762 | 0.271 | 0.372 |
| 0.2 | 0.486 | 0.524 | 0.266 | 0.202 |
| 0.4 | 0.459 | 0.575 | 0.399 | 0.477 |
| **0.6** | **0.490** | **0.742** | **0.493** | **0.689** |
| **0.9** | **0.497** | **0.800** | **0.508** | **0.764** |

For `Z_lr <= 0.2` the two track each other closely — the representational index is genuinely the
twin of the behavioural one. For `0.6` and `0.9` they part hard: Z carries *no* context
information (0.49, i.e. exactly ambiguous) while behaviour is strongly perseverative (0.74-0.80).
Whatever holds the old context in those arms is therefore **not Z** — it has to be the weights or
the recurrence. That is a question only the hidden state can answer, and it is now the strongest
reason to run the hidden-activity sweep below.

---

## Hidden activity needs a re-run

Hidden activity is absent from **every** rotation export: `compact_logger` listed `hidden_states`
in `_DROP_FIELDS`, and neither `log_hidden_states` nor `record_hidden` was ever set (the slips
config turns it off explicitly, "behavioural analysis only; keeps the pickles small").

At `stride = 1, batch_size = 1, seq_len = 5` the two available channels carry *identical numbers*:
the final `h` that `forward` returns is the post-gate `h` of the logged timestep, which is also
`_hidden_trace[-1]`. `record_hidden`/`hidden_trace` only earns its keep if `stride` grows, so
`log_hidden_states` is used — `flatten_hidden_states` and `extract_decode_samples` already read it
with no new plumbing.

To start the run:

```python
# rotation_curriculum_config.py
RECORD_HIDDEN = True        # repoints RUN_NAME at ..._zgrid1_hid
PILOT         = False       # 10 seeds, not 1 -- the array size is derived from this
```

```bash
./submit_job.sh 79 curriculum      # 80 array tasks (8 Z_lr x 10 seeds), ~10 min each
.venv/bin/python rotation_curriculum_rdm.py
```

> **Both flags, or the array is the wrong size.** `PILOT = True` (the current value) makes
> `generate_jobs()` return 8, so `submit_job.sh 79` would submit 80 tasks of which 72 raise
> "Task id out of range". Check with
> `.venv/bin/python -c "import rotation_curriculum_sweep as s; print(len(s.generate_jobs()))"`
> and pass that count minus one.

`RECORD_HIDDEN` is in `RUN_NAME`, so the hidden-logging run writes to its own directory and the
existing 253 MB of behavioural results stay intact. It also means flipping the flag repoints every
reader, including the behavioural figures — keep it `False` to work against the existing run, and
pass `run_name=` to read a specific one regardless.

Cost: ~5.7 MB per tree (22k logged timesteps × 64 units × float32), so ~720 MB for the full sweep.
If that is unwelcome, cast to float16 (the `hier_switch_analyses.save_session` precedent) or leave
`log_hidden_states` off for the two `S3_pinned` branches.

**The re-run must reproduce the behaviour exactly.** Logging appends draw no randomness, so
`rotation_curriculum_analysis.summarize()` and `check_acceptance()` on the `_hid` run must match
the documented un-tagged tables. A divergence means a config edit leaked into the run.

### What the full tree buys, beyond S1

Running every stage with hidden logging (rather than S1 alone) gives S2 and S3 RDMs with no second
re-run, and one exact positive control: in S2 with `cue_mode='oracle_z'` the gate *is* the
ground-truth one-hot, so the hidden RDM must be a near-perfect checkerboard. If the pipeline
cannot recover that, nothing else is readable.

> The *Z* panel is not the control to use there. `logger.latent_values` records the raw `model.Z`
> parameter, and under `what_latent_to_use='context_ids'` the gate is built inside the forward pass
> and never written back — so S2's Z is whatever S1 left behind. Same trap
> `inspect_curriculum_run.show_stage()` warns about.

---

## What the hidden activity shows (run `..._hid`, 80 trees, 10 seeds)

Behaviour reproduces the un-tagged run to within 0.0005 on S1 `mb2` at every arm, so this run is
a clean superset and everything below is comparable.

Two hidden channels are recorded, because under `post_gating` they are different quantities:
`H` is the post-gate state the readout consumes, `H_pre` the same timesteps before the gate is
applied. The recurrence never sees Z, so only `H` can carry the gate.

### `cmi` is not comparable across representations — read d' instead

For unit-variance features with per-unit separation d', between-context squared distance is
`2 + d'^2` per feature against `2` within, so

```
cmi ~= sqrt(1 + d'^2 / 2) - 1
```

A 64-unit population at d' = 0.92 gives `cmi ~= 0.20`, which is exactly what post-gate `H`
measures. Against the 2-dimensional Z's `cmi ~= 1.1` that looks like a weak effect. **It is not**
— it is the same effect on a scale that shrinks as the effect spreads over more dimensions.
`unit_discriminability` is the measure to compare across representations; `cmi` is only
comparable *within* one.

### Context is in the gate, not in the recurrence

Per-unit |d'| between contexts, matched on colour so the stimulus is held constant. The `'RNN'`
arm is the no-context floor.

| rep | `Z_lr` | mean \|d'\| | max \|d'\| | units > 0.5 | dead units |
|---|---|---|---|---|---|
| `H` (post-gate) | `RNN` | 0.341 | 0.498 | 0 / 64 | 34 |
| `H` | 0.05 | 0.471 | 0.653 | 24 / 64 | 34 |
| `H` | **0.1** | 0.882 | 1.701 | **58 / 64** | 34 |
| `H` | **0.2** | 0.924 | 1.985 | **61 / 64** | 34 |
| `H` | 0.4 | 0.491 | 0.783 | 29 / 64 | 34 |
| `H` | 0.9 | 0.340 | 0.467 | 0 / 64 | 34 |
| `H_pre` (pre-gate) | `RNN` | 0.366 | 0.470 | 0 / 64 | 0 |
| `H_pre` | 0.1 | 0.413 | 0.513 | 2 / 64 | 0 |
| `H_pre` | **0.2** | 0.425 | 0.539 | **7 / 64** | 0 |
| `H_pre` | 0.9 | 0.367 | 0.482 | 0 / 64 | 0 |

**Post-gate hidden activity represents context strongly** — 61 of 64 units at d' > 0.5 for
`Z_lr = 0.2`, and the checkerboard is plainly visible in the `H` row of `rdm_grid_ZvsH`.

**Pre-gate hidden activity does not.** Every arm sits at 0.36-0.43 against an `'RNN'` floor of
0.37, with at most 7 units above 0.5, and its RDM row is featureless. So the context signal in the
hidden state is **entirely imposed by the multiplicative gate**; the recurrent state the network
maintains on its own does not distinguish the two contexts. That is what `pre_gating=False` buys,
and it was worth recording both channels to see it rather than inferring it.

> **34 of 64 units are identically zero in the post-gate state.** The gate is
> `softmax(Z) @ mask` with `mask ~ Bernoulli(P_gates_bernoulli_prob=0.3)` over `Z_dim = 2`, so
> ~49% of units draw `[0, 0]` and are silenced outright, ~9% draw `[1, 1]` and are ungated, and
> the remaining ~42% carry the context-dependent gain. The effective post-gate population is
> therefore ~30 units, not 64. Worth knowing before quoting a population size.

### The directional index needs the `'RNN'` arm as its floor, not 0.5

Directional index by mini-block (10 seeds). Read the hidden rows **against the `'RNN'` arm**, not
against 0.5:

| rep | `Z_lr` | mb0 | mb1 | mb2 | mb3 | mb4 | late |
|---|---|---|---|---|---|---|---|
| `Z` | 0.05 | 0.888 | 0.646 | 0.406 | 0.249 | 0.178 | 0.156 |
| `Z` | 0.2 | 0.486 | 0.266 | 0.267 | 0.266 | 0.255 | 0.299 |
| `H` | **`RNN`** | **0.546** | **0.540** | **0.526** | **0.517** | **0.509** | **0.495** |
| `H` | 0.05 | 0.647 | 0.579 | 0.485 | 0.420 | 0.393 | 0.390 |
| `H` | 0.1 | 0.596 | 0.451 | 0.401 | 0.400 | 0.402 | 0.422 |
| `H` | 0.2 | 0.522 | 0.431 | 0.437 | 0.438 | 0.438 | 0.458 |
| `H_pre` | `RNN` | 0.533 | 0.531 | 0.520 | 0.511 | 0.506 | 0.497 |
| `H_pre` | 0.05 | 0.549 | 0.541 | 0.523 | 0.507 | 0.495 | 0.487 |
| `H_pre` | 0.2 | 0.537 | 0.520 | 0.511 | 0.507 | 0.505 | 0.507 |

**Why the floor is not 0.5.** The two templates are estimated symmetrically — one block each —
but they sit at *different temporal lags*: `settled_old` comes from block b-1 and `settled_new`
from b-2. If activity drifts across training, trials in block b are systematically nearer b-1
than b-2, so `d_old < d_new` and the index rises above 0.5 for reasons that have nothing to do
with context. The decline from mb0 to `late` has the same origin, and it mimics the shape of a
real effect.

This cannot be symmetrised away: under strict alternation same-context is *always* an even block
lag, so the same-context template is always further back in time than the other-context one — the
same structural problem as the lag-parity confound in `cmi`. The fix is the repo's standard
control: **read every arm against the `'RNN'` arm at the same mini-block.** Relative to that floor:

| `Z_lr` | mb0 | mb1 | mb2 | mb4 | reading |
|---|---|---|---|---|---|
| `H` 0.05 | **+0.10** | +0.04 | −0.04 | −0.12 | starts in the old context, crosses over |
| `H` 0.1 | +0.05 | −0.09 | −0.13 | −0.11 | brief, then switched |
| `H` 0.2 | −0.02 | −0.11 | −0.09 | −0.07 | no perseveration |
| `H_pre` 0.05 | +0.02 | +0.01 | 0.00 | −0.01 | nothing |

Same ordering as Z, compressed by the dimensionality effect above, and `H_pre` flat at the floor.
The Z results are unaffected in substance — a ~0.05 drift bias against a 0.89 -> 0.16 range — but
the hidden channels would be misread without the floor.

Colour was ruled out as an explanation for the modest `cmi`: restricting the contrast to
same-colour cross-block pairs moves it by 0.007 (0.192 -> 0.199 at `Z_lr = 0.2`).

**The obvious follow-up is `pre_gating=True`**, which is a one-line config change and which this
analysis is now set up to read without further plumbing.

---

## Figures

| | Content |
|---|---|
| **R1** `plot_rdm_grid` | The checkerboard. One matrix per (representation, `Z_lr`), seed-averaged, trials chronological, block boundaries in white, a context strip on two edges. A panel per arm is what [figure_style.md](figure_style.md) warns against for curves, but a matrix cannot be overlaid — `plot_candidate_rdms` sets the precedent |
| **R2** `plot_development` | The development curve: `cmi` against training block, one line per individual's `Z_lr`, band = SEM across seeds, `early` and `late` panels. No null band is drawn — the permutation null's width scales with how structured an arm's RDM is, so the widest one comes from the *strongest* arm and reading it as a common threshold would be wrong. Per-arm nulls are in the table; the `'RNN'` line is the empirical floor |
| **R3** `plot_behaviour_validation` | The index against behavioural recovery, one point per (`Z_lr`, seed), both windows. Where the dissociation above is read |
| **R5** `plot_separation_front` | `cmi` as (mini-block since switch) x (training block), one panel per arm. If the front moves toward the switch the bright region grows downward. Mini-block 0 is marked, because `cmi` there cannot be read as context identity |
| **R1b** `plot_rdm_grid(reps=('Z','H','H_pre'))` | The three representations side by side. Colour scale is **per row** — absolute dissimilarity is not comparable across dimensionalities |
| **R6** `plot_template_index` | The directional index against mini-block, with behavioural `norm_err` overlaid dashed on the same scale; plus the mini-block-0 index against training block. Where the 0.05 case is settled |
| **R4** `plot_frame_contrast` | Cue vs outcome, per arm. An **H** panel: Z is a free parameter fitted across the window rather than a function of the current frame's contents, so its two frame types coincide to within ~0.02 at every arm — useful as a check that the frame split does what it says, and nothing more |

Seed-averaging of RDMs is elementwise and legitimate only because `sample_trials` gives every seed
the same row layout and pins the first retained block to context 0 (which rotation a seed starts
with is a coin flip, and dissimilarity does not care about the label). Patterns are never pooled
across seeds — different networks have different hidden bases.

---

## Translation to fMRI

| Model | Human |
|---|---|
| one pattern per trial, cue frame | one beta per trial from the cue/delay period of an ROI |
| 64 hidden units | voxels in the ROI |
| pooled z-scoring across the sampled trial set | the same, per voxel |
| euclidean distance | euclidean (crossnobis is the noise-normalised standard and is a later variant) |
| `early` / `late` mini-block windows | the same, in trials since the block switch |
| `cmi` against training block | `cmi` against run / session index |
| directional index vs `settled(context)` from blocks b-1 / b-2 | the same, with the settled pattern taken from the previous block and the previous same-context block |
| `'RNN'` arm as floor | permutation null over the block→context map |

Three requirements the model analysis makes explicit:

1. **Cue and outcome must be separable in the design**, or the RDM carries the stimulus.
2. **Sample early in the block.** The late-window index ranked a behaviourally impaired model
   best; the early-window index tracked behaviour at ρ = −0.66.
3. **Use a directional measure early in the block, not an RDM contrast.** An RDM is invariant to
   relabeling the contexts, so early in a block it cannot separate "inferred the new context
   quickly" from "held the old one confidently" — and in this model the honest answer turned out
   to be the second one. A template index anchored on the previous blocks' settled patterns is
   the fix, and it needs no extra scanning, only the block-order bookkeeping.

---

## Traps

1. **`len(logger.inputs)` is 1 after compaction.** Use `logger.phases` (written pre-compaction)
   via `rotation_curriculum_analysis._phase_range`, which is what `blocked_range` does.
2. **`hlcids` restart at 0 in every phase and every stage**, so the passive warm-up and the blocked
   phase both number from 0. Always slice to `'Learning and inference'` first.
3. **The colour one-hot rides the cue frame only.** An outcome frame's colour is on the frame
   before it; `frame_type` (0=cue, 1=outcome) locates it.
4. **Block 0's boundary is the phase start, not a switch** — see the window section.
5. **`extract_decode_samples` used to hard-require hidden states.** It now takes
   `require_hidden=False`, under which `'H'` is simply absent from the returned dict, so reaching
   for it raises a clean `KeyError` rather than reading zeros. `verify_hidden_state_alignment`
   still raises unconditionally, which is correct — it exists to check them.
6. **The directional index is NaN for the first two blocks of every run**, by construction — no
   b-1 / b-2 to build templates from. `np.nanmean` handles it; the empty-slice warnings are
   suppressed where they are expected rather than left to imply a fault.
7. **Mini-block naming differs between the two analyses.** Behavioural `curve_summary`'s `mb1` is
   trials `1..n_colors`, which is mini-block index **0** here. Compare through
   `behavioural_by_miniblock`, not by name.
8. **`TEMPLATE_WINDOWS` stops at mini-block 4 on purpose.** In the shortest block (9 complete
   mini-blocks, the geometric draw's floor) `'last'` resolves to 8, so a window at 8 would collide
   with it and the block would be dropped whole.
9. **`verify_hidden_state_alignment` is only valid where the weights are frozen.** The output layer
   is updated every batch in S1/S2, so the final readout is not the one that produced an early
   logged prediction. Run it on S3.

---

## Verification

All with `.venv/bin/python`.

- **Self-test** — `python rotation_curriculum_rdm.py` runs nine synthetic cases before touching
  any data. For `cmi`: a context effect that ramps over blocks (recovered as a rising curve,
  0.033 → 0.456), no context effect (`cmi` 0.003, inside its null), pure drift (`cmi_dc` 0.03
  while `cmi` goes negative), and a colour-only effect (0.000). For the directional index: a
  perseverative model scores 0.910, a correct one 0.092, an ambiguous one 0.499, the reference
  window itself 0.092 — **and on the very same data `cmi` returns 0.699 vs 0.714**, which is the
  case for the index existing at all.
- **Window structure, on real data** — every window holds exactly one trial per colour, in colour
  order, at all three frame settings; 13 blocks with strictly alternating context; 0 blocks
  dropped.
- **Negative control** — the `'RNN'` arm's Z is bit-identical across trials and its RDM is
  identically zero.
- **Hidden-state alignment** — `verify_hidden_state_alignment` on a frozen-weight stage of a smoke
  run: `max |output_layer(h) − logged prediction| = 1.8e-07`.
- **Compaction** — `hidden_states` survives `compact_logger` as one concatenated float32 array and
  `flatten_hidden_states` reads it back aligned (checked: 1942 timesteps × 64 on a short S1).

---

## See Also

- [rotation_curriculum.md](rotation_curriculum.md) — the three-stage curriculum, the grid, the
  behavioural measurement this index is compared against
- [rotation_geometry.md](rotation_geometry.md) — RSA across rotation angles; `build_rdm` lives there
- [rotation_decoding.md](rotation_decoding.md) — Z vs hidden decoding; `extract_decode_samples`
- [logging.md](logging.md) — `hidden_states` vs `hidden_trace`
- [figure_style.md](figure_style.md) — `FigSize` presets and colour conventions
