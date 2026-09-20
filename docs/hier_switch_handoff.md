# Hierarchical cue→rule reversal — handoff

Phase 1 (build the task, get NeuraGEM to learn it and to discover the contexts on its own)
and phase 2 (the analyses that put the model next to the paper) are both built. The group
results are in §5b. Everything here was true on 2026-09-19; check the code before relying
on a detail.

Read, in this order:
1. this file;
2. `docs/hier_switch_task.md` — the task, the model, the full tuning log (v1–v16) and the
   inference tests on a saved model;
3. `docs/hier_switch_analyses.md` — **the phase-2 pipeline**: the session format, the trial
   table, every analysis definition, and §9 **the toolbox** (every function a new analysis
   can call, and the data already on disk);
4. `docs/hier_switch_methods.md` — the same work written as a methods section, with one
   caption per figure: what was run, what is plotted, what it shows;
5. `hier_switch/hier_switch_config.py` — every knob, with the reason for its value;
6. `hier_switch/hier_switch_train.py`, `hier_switch/hier_switch_analyses.py` and
   `hier_switch/hier_switch_test_inference.py`;
7. `docs/figure_style.md`, and `docs/flanker_task.md` §"Five analysis conventions" — the
   flanker project is the house precedent for how analyses and figures are built here;
8. `/oscar/AGENTS.md` — the cluster's site policy for agents (summarised in §10).

**Start with §5b** (what the analyses found), §5c (the story figure) and §7 (what is left).

**Update, 2026-09-20.** The analyses were reorganised around one figure that tells the whole
story (`exports/hier_switch/group/figures/story.pdf`, five rows of four panels, captioned row
by row in `docs/hier_switch_methods.md`). Three things changed in the analyses themselves and
one stage was built:
- **Context is now labelled relative to training** (`ctx_rel`), because three of the six
  networks finished training in context 1 and three in context 0, so the raw label pooled two
  different things. See §5c.
- **A new encoding analysis** (`encoding_table`) asks what each of the model's signals carries,
  with a dimension-matched control. It reproduces the paper's mixed-cortex / demixed-thalamus
  dissociation. See §5c.
- **Reversal-aligned versions** of the integration index and cue velocity, and of the latent,
  its update and its gradient.
- **E4, the trial hooks, is built** (`hier_switch_hooks.py`) and its 15 conditions are defined,
  but **the sessions have not been run** — that is the one thing outstanding. See §7.

---

## 1. The scientific question

Lam, Mukherjee, Wimmer, Nassar, Chen & Halassa, *Prefrontal transthalamic uncertainty
processing drives flexible switching*, Nature 637:127 (2025), doi 10.1038/s41586-024-08180-8.

**The task.**
- Tree shrews hear a 16-pulse cue: 9 informative pulses, high-pass (HP) or low-pass (LP),
  plus 7 white-noise pulses. The dominant type is the **cue**.
- The cue sets the **rule**: attend vision or audition. The animal picks the side of the
  attended target.
- The cue→rule mapping reverses covertly every 30–60 trials.

**Two uncertainties.**
- *Cueing uncertainty:* conflict = non-dominant / dominant pulse count.
- *Rule uncertainty:* has the context reversed?

**Key findings to match.**
- **Switching is slower when the first post-reversal trials are high-conflict** (Fig 1f).
  An error on an ambiguous cue is blamed on the cue, not on the world.
- **ACC carries a conflict-weighted error**: ε_CW = error / (1 − P(correct cue)), averaged
  over ~5 trials.
- **MD is low-dimensional and demixed.** MDContext tracks rule uncertainty; MDConflict
  tracks cue conflict.
- **PFC has three cell classes: CueS (transient) → CueL (integrated) → Rule.** Rule and CueL
  are suppressed early after a reversal.
- **Causal tests.** Silencing ACC→MD delays switching; activating MD speeds it up.
- **The paper models behaviour with a hand-built two-state model (2SM), not an RNN.**

**Why NeuraGEM fits.** Z is updated by gradient descent on the response error,
dL/dZ = (y − target)·∂y/∂Z. The ∂y/∂Z factor shrinks when cue evidence is weak, because the
context has little leverage on an output that is ambiguous anyway. So the latent update
should be **conflict-weighted by construction**: ACC→MD credit assignment with no
hand-built ε_CW. That is the headline hypothesis. The second one is that Z's uncertainty
sets integration speed and response speed.

---

## 2. The roadmap (the original plan, with status)

The original plan file lives outside the repo, at
`~/.claude/plans/i-want-to-implement-sequential-ocean.md`. This section carries its content
forward. ✅ done · 🟡 partly · ⬜ not started.

### Model ↔ circuit correspondence (hypotheses to test, not assumptions)

| Paper | NeuraGEM | Test |
|---|---|---|
| ACC, conflict-weighted error ε_CW | dL/dZ per trial (`logger.gradients_corrections`) | \|dL/dZ\| against ε_CW computed with the paper's formula |
| MDContext, rule uncertainty | Z belief and entropy (softmax over 2 units) | Z against a Bayesian context posterior |
| MDConflict | **no analogue**: Z is fixed within a trial | say so; the hidden state carries conflict instead |
| PFC CueS / CueL / Rule | LSTM hidden state | per-unit regression; cue and rule axes |
| 2SM "exploration" state | Z crossing the uniform gate → outputs near 0 | reversal-aligned rate of undecided / low-\|decision\| trials |
| MD activation / ACC→MD silencing | Z_lr boost / latent update off after a reversal | switch latency |

### Experiments

| | Experiment | Status |
|---|---|---|
| E1 | **Core session.** Train (passive, then active; discovery) → frozen-weight test on the paper's 30–60 blocks | ✅ 6/10 seeds discover (v13); all six retrained and saved as `tune_v15/NG_s*/model.pt`, each reproducing its v13 numbers exactly |
| E2 | **Z-clamp probe.** Freeze weights and the latent update; hold Z at a grid of gates; measure RT, accuracy, undecided rate, the integration index and the cue/rule build-up | ✅ `hier_switch_perturb.clamp_grid`: softmax contrast ladder and a sigmoid gain × contrast grid, per seed |
| E3 | **Group sweep.** ≥ 10 seeds × {NG, RNN} plus the ideal observer | ✅ 6 NG (the discoverers) × 6 conditions and 10 RNN × 3 conditions, all recorded and analysed; `hier_switch_group.py aggregate` prints the prediction table. Oracle → inference not repeated at group level |
| E4 | **Perturbations.** Latent update off for the first 4 post-reversal trials (ACC→MD silencing); the latent driven for the first trials (MD activation); momentum; forced errors | 🟡 **Built, not run.** `hier_switch_hooks.py` plus the default-off call sites in `_latent_update_step`; 15 conditions defined and `run_manipulations.sh` ready. Needs ~90 SLURM sessions and the user's go-ahead. Forced errors still need a dataset knob. See §7 |

### Predictions

- **P1.** Accuracy falls with conflict and is equal across contexts (Fig 1e). *Seen in
  training: the Oracle goes 1.00 → 0.65 from conflict 0 to 0.8; ideal observer 1.00 → 0.70.*
- **P2 (headline).** Switch latency is longer when the early post-reversal trials are
  high-conflict (Fig 1f). The ideal observer gives the normative size of the effect.
- **P3.** Z entropy rises briefly after a reversal; the rise is sharper and shorter when
  the early trials are low-conflict (Fig 1k,l). Early in the transition, Z uncertainty
  correlates negatively with cue conflict (Fig 1m).
- **P4.** |dL/dZ| tracks ε_CW.
- **P5.** In the hidden state, the cue axis is the same in both contexts while the rule axis
  flips (Fig 2c). After a reversal the rule axis weakens and the cue axis holds.
- **P6 (speed).** RT rises with conflict. Reversal-aligned RT is **non-monotonic**:
  perseverative errors on trials 1–2 are fast and confident; RT peaks where Z is most
  uncertain, together with the undecided-rate peak (the "exploration" analogue); then it
  recovers. The paper does not report RT, so this is a new prediction.
- **P7 (E2).** Clamping Z toward uniform slows the rule-axis build-up and RT, and leaves the
  cue-axis build-up untouched.
- **Informative failures.** If P2 fails, gradient inference is not doing hierarchical credit
  assignment. If P7 fails, the gate is not acting as gain on integration.

### Analyses

Every analysis below is built; the definitions live in `docs/hier_switch_analyses.md` and
the numbers in §5b. ✅ = built and run on the group.

| | Analysis | Where |
|---|---|---|
| A1 | Psychometric: accuracy / RT / undecided by conflict × context, against the observer's ceiling | ✅ `behaviour()` |
| A2 | Reversal-aligned accuracy, Z side, \|decision\|, RT and undecided (−5…+15), split by early-reversal conflict — forced (the paper's design) or a median split | ✅ `behaviour()`, `reversal_aligned()` |
| A3 | Perseverative errors, trials to criterion, three switch criteria | ✅ `_switch_stats` |
| A4 | RT and undecided rate; fast-error check with a decided-only companion | ✅ `behaviour()['rt']` |
| B1 | Z belief against the ideal-observer posterior around reversals | ✅ `latent()['belief']` |
| B2 | Z-uncertainty peak height, position and width, split by early-reversal conflict; Fig 1m | ✅ `latent()['uncertainty']` |
| B3 | Update rule: what explains the move along the context axis | ✅ `latent()['update_rule']` |
| B4 | \|dL/dZ\| against ε_CW, per trial and over the paper's 5-trial window | ✅ `latent()['eps_cw']` |
| **B5** | **How each trial moves Z**: the update along the context axis and along the gain axis, by conflict × outcome × context × whether Z held the true context, against the observer's normative update | ✅ `z_updates`, `z_update_table`, `normative_table`; definitions in `docs/hier_switch_analyses.md` §4 |
| C1 | Per-timestep cross-validated decoding of cue, rule, context and conflict from the hidden state | ✅ `hier_switch_hidden.decode` |
| C2 | Per-unit CueS / CueL / Rule classification against a permutation null | ✅ `unit_classes` |
| C3 | Cue and rule axis magnitude reversal-aligned; the paper's integration index and cue velocity | ✅ `hidden_report`, `integration` |
| D | Z clamp (E2) ✅ `hier_switch_perturb`; dynamic perturbations (E4) 🟡 built, not run |
| F | **What is encoded where**: every signal (hidden state, its top 2 PCs, Z, ΔZ, dL/dZ) against every variable, by decoding with a per-cell null and by drop-one demixed variance | ✅ `hier_switch_hidden.encoding_table`; §5c and `docs/hier_switch_analyses.md` §6b |
| E | Ideal observer (numpy): frame-wise cue posterior, hazard-1/45 context filter, its own choice and switch latency | ✅ `hier_switch_observer.py` (0.93 cue accuracy at σ 0.5, as the tuning log had it) |

### Design choices already settled, which the analyses must respect

- **All blocked. Never interleaved.** The user was emphatic about this. Contexts strictly
  alternate; only block length changes.
- **Test on the paper's 30–60-trial blocks**, fresh trials (own RNG stream), weights frozen,
  Z inferred.
- **No context output head.** The only supervision is the correct side, which carries the
  same information as binary reward in a 2-choice task. Context is read from Z.
- **SGD on Z, not Adam.** Adam normalises away the gradient magnitude, which is the
  conflict weighting under test.
- **Free-response variant (not built).** Targets visible from t = 0, so RT measures
  integration directly. It would be a `target_onset` knob in `set_timing()`; the plan kept
  it as a later variant. Follow the flanker target-delay rule: nothing gets compensated.
- **Forced early-reversal conflict (not built).** `reversal_conflict = None | 'low' |
  'high'` would force the first k = 4 post-reversal trials' conflict, a controlled version
  of Fig 1f. It is test-stage only and off by default.

---

## 3. The model as it stands

**One trial is one batch:** 25 timesteps. The LSTM hidden state resets every trial, so **Z
is the only thing carried across trials**.

| frames | content |
|---|---|
| 0–15 | pulses: one-hot `[HP, LP, WN]` + Gaussian noise (`pulse_noise_std` 0.5) |
| 16–17 | delay |
| 18–24 | targets `[vis, aud]`, ±1 with aud = −vis; response window |

`correct_side = vis × cue_sign × context_sign`. It is a three-way parity, so a reversal flips
every answer.

**Channels.** Labels ride in the input but are masked from the model. Read them from each
trial's **last** frame: `vis` is 0 before the targets.

```
idx  0   1   2   3    4    5    6         7        8
     HP  LP  WN  vis  aud  cue  conflict  context  correct
in   1   1   1   1    1    0    0         0        0     (input_feed_mask)
out  0   0   0   0    0    0    0         0        1     (output_loss_mask)
```

**Phases** (`logger.phases` names):
1. `'no inference learning'` — **passive**: 2 × 2000 trials, one block per context. Weights
   only; Z held at `Z_init`.
2. `'Learning and inference'` — **active**: 5000 trials on 200-trial blocks. Weights + Z.
3. `'Inference only'` — **test**: 1000 trials on 30–60-trial blocks. Weights frozen; Z
   inferred, continuing from where training left it.

These are v13's verified phase lengths, the configuration behind the 6/10 in §5, and they
are the defaults: `hier_switch_train.py` with no arguments trains seed 0, which discovers
(test 0.832, steady 0.91). A longer passive phase (2 × 3000) with 4000 active trials is an
untested alternative.

**Defaults** (`HierSwitchConfig`):

| setting | value |
|---|---|
| network | LSTM, 64 units |
| Z | 2 units, softmax at temperature 0.5, `Z_init = 0` (uniform gate) |
| gating | multiplicative, pre-gating, Bernoulli mask p = 0.3 |
| Z optimizer | SGD, `Z_lr = 1e4` |
| Z weight decay | `Z_decay = 3e-6`, plain L2 (`Z_decay_mode 'grad'` ≡ SGD `weight_decay`); Z_lr·Z_decay = 0.03 |
| dL/dZ pooling | uniform over the trial (steepness 0) |
| weights | Adam, `WU_lr = 1e-3` |
| speed pressure | `temporal_loss_weights` 0 until t = 19, then `exp(−0.178·i)` |

---

## 4. What the numbers mean

| Term | Definition |
|---|---|
| `decision` | the response output averaged over the response window (t 19–24) |
| `correct` | sign(decision) == correct side |
| `since` | trials since the last reversal; 1 = first trial of a block |
| `z` / `z_in` | `z` is the logged Z **after** the trial's own latent update. `z_in` (previous trial's `z`) is the state the trial actually ran under. Use `z_in` for "what drove this trial" and `z − z_in` for "what this trial taught" (flanker convention 2) |
| p(gate) | `softmax(z / 0.5)`; Z unit 1's share of the gate |
| gain, `mean(z)` | the common mode of the Z units. **Identically 0 under the softmax.** The softmax's gradient sums to zero across units and weight decay only shrinks the mean, so it never moves: measured \|mean(z)\| < 3e-6 over whole runs. It is a live axis only when the softmax is replaced (sigmoid, no activation), where it moves more than the contrast: sigmoid at test, per-trial SD 0.56 against 0.29 |
| contrast | (z₁ − z₂)/2, the only thing the softmax sees; for two units it *is* the context axis |
| steady-state accuracy | trials 11+ into a block |
| `cross_trial` | first trial after a reversal whose 3-trial mean accuracy reaches 0.5 |
| `s`, `tipped`, `z_evidence`, ε_CW | the phase-2 per-trial quantities; defined in `docs/hier_switch_analyses.md` §3–§4 |
| Z d′ | separation of the two contexts along the axis between their mean `z_in` |
| discovered | test steady-state ≥ 0.8 **and** Z d′ > 1.5 (`hier_switch_tune.py arms`) |
| passive ok | the passive phase's last block ends ≥ 0.75: the task itself was learned |

---

## 5. Where tuning landed (details in the doc's tuning log)

**Results.**
- **Discovery: 6/10 seeds (v13)** — s0, s1, s3, s5, s6, s9.
  - Test: 0.75–0.85 overall, 0.82–0.93 steady-state, chance crossed at trial 3–4.
  - s4 is partial: Z splits but the output half-hedges.
  - s2, s7, s8 **never learn the task** in the passive phase, so they never get to discovery.
  - Of the seeds that learn the task, 6 of 7 discover.
- **v14:** `WU_lr = 3e-3` in both phases → 0 of 10 seeds learn the task, even in the
  passive phase, including the six that discover at 1e-3. Faster weight learning breaks the
  initial learning. Keep 1e-3.
- **Oracle → inference.** Trained with the context given, then Z inferred at test. Best:
  0.84 overall, 0.89 steady, crossing at trial 3, at Z_lr·Z_decay ≈ 0.1. The Oracle's
  training accuracy is 0.895; the ideal observer given the context is 0.923.

**Knobs, and what they do.** Change them through `make_config(model, **overrides)`; see §6.

| Knob | Effect |
|---|---|
| **Z_lr·Z_decay** (the strength of weight decay per update) | Discovery needs ≈ 0.03. At 0, Z never flips with the context. At ≥ 0.06 in training, Z is held at the middle before the weights specialise. For inference on already-trained weights, 0.1 switches fastest (trial 3 vs 10 without decay); 0.3 switches at once but cannot hold the context |
| **Z_lr** | 1e4 splits Z earliest in training. Without decay, ≥ 1e4 thrashes at inference. The usable range scales ~1/temperature |
| **softmax_temp** | 0.5. At 1.0 the gate is too soft: even the Oracle cannot learn |
| **latent_activation** | softmax. In training, `'none'` collapses to a dead gate (Z → 0) and `'sigmoid'` turns Z into a gain, not a context code. At test on a softmax-trained model, see §6b |
| **passive phase** | Essential: early errors carry no context information and drag Z to uniform. 2 × 1500 was too short (1–2/6 seeds learn); 2 × 2000 (the default) gives 7/10; 2 × 3000 is untested |
| **block length** | Without the context, learning the task needs ~1800 trials in one context, so 30–60 or 600-trial training blocks never learn. After passive learning, 200-trial active blocks work. 1000-trial blocks never hedge but give few reversals |
| **WU_lr** | **Not a useful lever; leave it at 1e-3.** 5e-4 changed nothing (v11). 3e-3 breaks learning entirely: 0/10 seeds learn the task (v14) |
| steepness (dL/dZ pooling) | 0 vs 2: no difference |
| gating site | pre beats post |
| Z_dim | 2. At 4, the model never learned (undiagnosed) |

**The failure signature to watch for.** Training either converges or hedges, with little in
between. A hedging model's output shrinks toward 0 (|decision| ≈ 0.05), the Z gradient
vanishes, and Z sits at the middle. `show_blocks.py <tag>` shows it block by block.

---

## 5b. What the phase-2 analyses found

**The group.** Six NG seeds (the v13 discoverers, retrained and saved as v15) and ten RNN
baselines (v16), each tested on the paper's 30–60-trial blocks with frozen weights. The NG
seeds also ran the paper's *controlled reversals*: paired sessions in which the first five
trials of every block are forced to low (7:2)  or high (6:3) conflict and every other trial
is identical, so the low/high comparison is within-seed and within-trial-stream. Every row
below is a difference or a slope whose sign a prediction fixes, with the count of seeds
carrying that sign; the full table is `exports/hier_switch/group/group.json`, the nine
figures are in `exports/hier_switch/group/figures/` with a caption each in
`docs/hier_switch_methods.md`, and the definitions are in `docs/hier_switch_analyses.md`.

**Held up (6/6 seeds unless noted).**

| Prediction | Result |
|---|---|
| **P1** accuracy falls with conflict | 0.32 from conflict 0 to 0.8; steady accuracy 0.878 ± 0.014 against the observer's ceiling of 0.924 with the context given. The two contexts differ by 0.036 |
| **P2** switching is slower after high-conflict reversals | Z-side latency +0.61 ± 0.08 trials (6/6), decided-and-correct +0.66 ± 0.09 (6/6), the raw behavioural criterion +0.21 ± 0.13 (4/6). The ideal observer's own cost is +1.07, so **the model shows about 60 % of the normative effect** |
| **P3** Z uncertainty rises after a reversal | +0.27 ± 0.03 over baseline, peaking at trial 3 in every seed, and it lasts 1.2 ± 0.3 trials longer after high-conflict reversals (5/6) |
| **P4** the gradient is the ACC-like conflict-weighted error | The 5-trial mean \|dL/dZ\| tracks ε_CW at r = 0.58 ± 0.02; ε_CW itself averages 1.10 in the steady state, as the paper's does. On error trials \|dL/dZ\| falls from 3.4e-5 at conflict 0 to 2.0e-5 at 0.8; on correct trials it is flat and ~5× smaller |
| **B5** the update is conflict-weighted by construction | The model's mean move per cell tracks the observer's log-odds update at r = 0.84 ± 0.02. In the regression `s ~ err + err × conflict` the interaction is −0.17 ± 0.04 (6/6) and adds R² (6/6). The stale-error slope against conflict is −0.09 ± 0.05 (5/6) |
| **P5** the rule code collapses after a reversal, the cue code does not | Rule decoding from the hidden state drops 0.39 ± 0.02 between steady state and the first 5 trials (6/6) while cue decoding drops 0.013 ± 0.012. The cue axis agrees across contexts (cos 0.67 ± 0.02) while the rule axis flips |
| **P6** RT (a new prediction: the paper reports none) | RT rises 1.7 ± 0.3 timesteps from conflict 0 to 0.8 (6/6); reversal-aligned RT is non-monotonic — fast perseverative errors on trials 1–2, a peak 1.3 ± 0.1 later at the trials where Z is most uncertain, then recovery (6/6); errors early after a reversal are 1.9 ± 0.3 timesteps faster than steady-state errors (6/6) |
| **B1** Z tracks the observer | r = 0.63 ± 0.04 at lag 0 in every seed. The model's belief recovers a little more slowly than the observer's |
| **P7 / E2** the gate is a gain on integration, not on the cue | Clamping Z for a whole session (weights frozen, latent update off), 6 seeds. At the uniform gate the output collapses: \|decision\| 0.18, undecided 0.85, RT 24.3 of a 25-step trial, integration index 1.33; what little sign is left follows a seed-specific default context (0.67–0.91 against 0.08–0.33 in four of six seeds, near chance in the other two) rather than the block's. At any committed gate the model is 0.89–0.91 on the context that gate selects and 0.10 on the other, \|decision\| up to 0.83, undecided 0.10, RT 19.1, index 1.44–1.58. **The cue velocity does not move at all** across the ladder (0.188–0.193), and rule decoding never reaches criterion in any clamped cell — the gate scales the rule-driven integration and the decision, and leaves the cue build-up alone |
| **Gain** (sigmoid at test, where the axis exists) | Errors push the gain down (−0.15 to −0.24 per trial) and correct trials push it back up (+0.02 to +0.06), at every rung of the decay ladder (5/6, 5/6, 6/6) — the "gain leak" the phase-1 inference tests saw, now measured per trial, and it costs 0.18 of steady-state accuracy against the softmax on the same weights. Under the softmax Δgain is identically 0 (max 1.9e-7): the sum-to-one constraint is what keeps the latent a context code rather than a volume knob. Figure: `gain.pdf` |

**Gain and contrast do different jobs — a clean double dissociation** (the sigmoid clamp
grid, 4 gains × 5 contrasts × 6 seeds, 120 sessions). Averaging over the other axis:

| clamped gain m | cue velocity | integration index | accuracy (matching context) | \|decision\| | undecided |
|---|---|---|---|---|---|
| −1 | 0.126 ± 0.004 | 1.09 ± 0.02 | 0.61 | 0.16 | 0.92 |
| 0 | 0.188 ± 0.003 | 1.42 ± 0.03 | 0.83 | 0.45 | 0.46 |
| +1 | 0.231 ± 0.006 | 1.61 ± 0.05 | 0.76 | 0.64 | 0.28 |
| +2 | 0.257 ± 0.008 | 1.54 ± 0.05 | 0.69 | 0.53 | 0.41 |

| clamped contrast d | cue velocity | integration index | accuracy (matching context) | \|decision\| | undecided |
|---|---|---|---|---|---|
| 0 | 0.200 ± 0.014 | 1.38 | 0.62 | 0.26 | 0.79 |
| 0.5 | 0.200 ± 0.013 | 1.38 | 0.69 | 0.34 | 0.64 |
| 1 | 0.201 ± 0.012 | 1.39 | 0.76 | 0.48 | 0.44 |
| 2 | 0.201 ± 0.008 | 1.49 | 0.83 | 0.68 | 0.25 |

**The gain sets how fast the hidden state integrates the cue** — cue velocity rises
monotonically with it in 6 of 6 seeds — **and the contrast sets which context is applied and
how decisive the output is**, leaving the cue velocity untouched to three decimal places
across the whole ladder. Accuracy is non-monotonic in gain (best at 0 to +1: too much gain
saturates the gate and the two units stop differing). This is the direct answer to "does Z's
gain modulate cue integration": yes, and it is the *only* thing that does.

**Did not hold, or is weaker than the paper's.**

- **Fig 1m** (rule uncertainty and cue uncertainty anticorrelated early in a transition) is
  **not** reproduced: the peak of Z uncertainty against the block's early conflict is
  +0.06 ± 0.06, and the per-trial version is ~0. In this model Z carries the *past* trials'
  conflict, not the current trial's, so the paper's within-trial version has no analogue.
- **The exploration regime signatures are weak.** The integration index is lower in the
  first five trials after a reversal (0.03 ± 0.02, 4/6) and the cue velocity is higher
  (0.007 ± 0.013, 4/6) — both in the paper's direction, neither convincing.
- **P(tipped | error) is not monotonic** in conflict (2/6 for the endpoint contrast): it
  peaks at conflict 0.29. The size of the update falls with conflict, but whether that
  update crosses the midpoint also depends on how far Z already was.
- **CueS is nearly absent in NG**: 3 % of tuned units, against 64 % CueL and 33 % Rule. The
  paper's three classes are all substantial in PFC.

**NeuraGEM against the backprop RNN** (10 seeds, same task, same test):

| | NG | RNN |
|---|---|---|
| steady-state accuracy | 0.878 | 0.502 |
| undecided rate (\|decision\| < 0.5) | 0.15 | **0.999** |
| context separation in Z (d′) | 6.08 | 0 (Z never moves) |
| rule decoding from the hidden state | 0.906 | 0.511 |
| cue decoding from the hidden state | 0.916 | 0.788 |
| context decoding from the hidden state | 0.997 | 0.549 |
| unit classes (fraction of tuned units) | CueS .03 / CueL .64 / Rule .33 | CueS .41 / CueL .58 / Rule .01 |
| integration index | 1.80 | 1.09 |

This is Brabeeba's table, and it comes out as he framed it: with plastic weights but no
latent, the RNN still reads the **cue** out of the pulses, but it has no rule code, no
context code and no way to switch — on 30–60-trial blocks it hedges on essentially every
trial (undecided 0.999), so its "accuracy 0.50" and its switch latency are coin flips on a
near-zero output and should never be quoted as behaviour. NG's context decoding from the
hidden state is near 1.0 for the trivial reason that Z gates that hidden state.

**Caveats to carry.** Every NG number is over the six seeds that discovered the contexts —
4 of 10 seeds are excluded, three because they never learned the task at all. The
behavioural switch criterion is contaminated by hedging, which is why the Z-side and
decided-correct criteria are reported beside it. The RNN never saw the sigmoid or clamp
conditions, and the Oracle was not re-run at group level.

## 5c. The story figure, and what reorganising found

`exports/hier_switch/group/figures/story.pdf` — five rows of four panels, a–t, each row also
written on its own. Every panel is captioned in `docs/hier_switch_methods.md`, which is the
place to read this from. Two earlier figures were folded into it: `switching.pdf` is retired
and `behaviour.pdf` lost its reversal-aligned RT panel. No builder was deleted.

**Three findings that were not visible before.**

**1. The two contexts are not interchangeable.** A network leaves training with its weights
and its latent sitting in the context of the last block it saw, and that context stays
easier: accuracy is higher in it at every conflict level, by 0.042 on average, in the same
direction in every seed. Three of the six networks finished in context 1 (s0, s3, s5) and
three in context 0 (s1, s6, s9), so the old per-context split averaged the two halves of a
real asymmetry against each other. Everything per-context now goes through `ctx_rel`
(0 = last trained), read from the final trial of the active phase.

**2. The latent is a pure context code and the hidden state is not — and it is not a
dimensionality effect.** From the encoding table (§6b of the analyses doc), decoding minus
each cell's own shuffle null, 6 seeds:

| signal | cue | rule | context | conflict |
|---|---|---|---|---|
| hidden state (64 units) | +0.42 | +0.33 | +0.40 | +0.35 |
| hidden state, top 2 PCs | +0.41 | +0.32 | **+0.09** | +0.02 |
| Z (2 units) | −0.02 | −0.00 | **+0.41** | −0.01 |
| dL/dZ | +0.07 | +0.06 | +0.14 | (outcome +0.16) |

Every hidden row is 6/6 seeds; every "flat" entry is 0/6. A hidden unit is tuned to 4.5 of
seven variables on average, while context alone explains 0.65 more of Z's variance than the
best other variable (6/6). **The dimension-matched control is what makes this a result rather
than an artefact**: the hidden state's top two components carry cue and rule as well as all
64 do, and carry context barely at all — so the hidden state's near-ceiling context decoding
is neither a unit-count effect nor something the network computes. Context reaches it only
because Z gates it, and it sits outside the directions that dominate its variance.

**3. The latent's state is persistent and its update is transient.** Aligned on a reversal:
the position on the context axis goes +1.0 → −0.99 on trial 1 and climbs back over 4–6
trials, while the update peaks at trial 2 (0.10 → 0.56 of the prototype distance, back near
baseline by trial 8) and the raw gradient does the same (7.5e-6 → 4.4e-5 → 1.7e-5). The
paper's thalamic switch response is transient and its context code is persistent; here both
are present as the derivative and the integral of one another. That correspondence is not in
the paper and is only available because the state and its update are separately measurable.

**Reproduced, and newly quantified.** The paper's Fig 1f now also holds on *uncontrolled*
reversals: binning each session's own reversals into three equal-count bins of early
conflict, the decided switch latency is 4.6 / 5.3 / 6.1 trials, slower in the most ambiguous
bin than the least in **6 of 6** seeds; the ideal observer pays 2.6 / 2.9 / 3.5 on the same
trials. The raw behavioural criterion does not show it (4.1 / 4.6 / 4.3) — hedging again.

**Still weak, and now explained.** The paper's exploration-regime signature barely appears:
the integration index falls by only 0.079 ± 0.025 in the first five post-reversal trials
(5/6) and the cue velocity does not move (+0.001 ± 0.011, 4/6). The clamp grid says why this
is coherent rather than contradictory — **cue velocity is set by the gain, and under the
softmax the gain cannot move**. The one measure the paper uses to define the exploratory
regime is the one this model's latent has no way to change. Testing that properly means
training with the sigmoid (§6b), which is still not done.

## 6. How to run things

```bash
.venv/bin/python hier_switch/hier_switch_dataset.py            # generator self-check
.venv/bin/python hier_switch/hier_switch_train.py NG 0         # train + test + panels, seed 0
./hier_switch/run_tune.sh <tag> [range]                        # SLURM array from GRIDS[tag]
.venv/bin/python hier_switch/hier_switch_tune.py arms <tag>    # seeds grouped by setting
.venv/bin/python hier_switch/hier_switch_tune.py collect <tag>  # one row per run
.venv/bin/python hier_switch/show_blocks.py <tag> [run ...]    # per-block convergence

# phase 2 (docs/hier_switch_analyses.md)
.venv/bin/python hier_switch/hier_switch_analyses.py           # synthetic self-test
.venv/bin/python hier_switch/hier_switch_observer.py           # observer self-check
.venv/bin/python hier_switch/hier_switch_test_inference.py <model.pt> [condition ...]
./hier_switch/run_sessions.sh                                  # record + analyse, one task per model
./hier_switch/run_clamp.sh 0-11                                # the Z-clamp grids
.venv/bin/python hier_switch/hier_switch_group.py aggregate    # the prediction table
.venv/bin/python hier_switch/hier_switch_figures.py [session]  # group, or one session
.venv/bin/python hier_switch/hier_switch_figures.py story      # only the story figure
.venv/bin/python hier_switch/hier_switch_hooks.py              # trial-hook self-test
./hier_switch/run_manipulations.sh                             # E4: the ~90 new sessions
```

- **Python** is the repo `.venv` (it links to `~/venvs/neo`). System python has no torch.
- **Runtime:** one seed (passive + active + test) takes ~5–6 min on a SLURM node. The login
  / interactive node has 2 CPUs.
- **Determinism:** a seed is deterministic, covering both weight init and data
  (`env_seed = seed`). Rerunning a seed reproduces it; a v13 seed can be regenerated rather
  than stored.
- **API.**
  - `make_config(model_type, **overrides)` → config. `model_type` is `'NG' | 'RNN' |
    'Oracle'`. `n_train_trials` and `n_passive_trials` go through setters; everything else
    is a plain assignment.
  - `build_model(cfg, seed)` → seeded model with Z at `Z_init`.
  - `run(model_type, seed, run_name=..., save_model=False, **overrides)` →
    `(logger, model, cfg)` via `train_model`. `save_model=True` writes
    `<export_path>/model.pt`; `load_model(path)` → `(model, cfg)`.
  - `run_test(model, cfg, Z_lr=..., run_name=..., **latent)` runs a test session on a
    **copy** of a trained model with any latent settings. It patches the live optimizer's
    lr and weight_decay; this is how the Oracle Z_lr sweeps were done.
  - `extract_trials(logger, cfg)` → per-trial dict. `summarize(trials, phase, last_frac)`,
    `block_table(trials, phase)`.
- **Inference tests on a saved model.**
  `hier_switch_test_inference.py <model.pt> [condition ...]` runs named latent conditions
  (a condition whose `session.npz` exists is skipped unless `HIER_SWITCH_FORCE=1`)
  (`CONDITIONS`) on copies of the model, on identical test trials, and **records** each one.
  With no condition named it runs the default set for that model type (NG or RNN).
  - Results go to `exports/hier_switch/inference_tests/<model>/<condition>/`:
    `session.npz` (what the analyses read), `summary.json`, `trials.npz`, the panels, and
    `results.json` once `hier_switch_group.py analyse` has run.
  - A one-page comparison, `inference_summary.pdf`, is rebuilt from the softmax / sigmoid /
    no-activation conditions on disk.
- **Tuning.** `hier_switch_tune.py` holds every grid (`GRIDS`, v1–v16) as a record. An entry
  may carry `tests=[...]` to run several test sessions on one trained model, and
  `save_model=True`. `v15` is the six saved NG discoverers, `v16` the ten saved RNN
  baselines (saved *before* their test phase, which is the one phase where an RNN's weights
  still move).
- **Git.** Phase 1 is on `main` (`1a1a09c`, `208e7d5`). Phase 2 is uncommitted in the
  working tree at the time of writing. The tree also carries the user's own unrelated work
  (flanker, rotation, a `run_flanker.py` rename). Do not commit those with this work.

---

## 6b. Open thread: removing the softmax

Softmax discovery works. The current side question is whether the softmax can be removed,
tested so far only **at test time** on a softmax-trained model (v15 seed 0, with
`hier_switch_test_inference.py`):

| condition at test | result |
|---|---|
| no activation | fails at every Z_lr: the gain ratchets down and the model hedges, or blows up |
| sigmoid | never collapses and switches as fast (crosses chance at trial 3), but holds the context worse: 0.67 overall / 0.71 steady against the softmax's 0.83 / 0.91 |

In both cases the two Z units drift together (gain) as well as apart. The softmax's
sum-to-one constraint is what removes that direction. The natural next step, not yet run,
is to *train* with the sigmoid from the start, now that the passive phase exists. Details
are in `docs/hier_switch_task.md` (end of the tuning log).

## 7. What is left

1. **E4, the manipulations — built, not yet run. This is the next thing to do.**
   `hier_switch/hier_switch_hooks.py` holds the hook (§6c of the analyses doc), the call
   sites are in `_latent_update_step`, the 15 conditions are in
   `hier_switch_test_inference.CONDITIONS`, and `hier_switch/run_manipulations.sh` submits
   them. The self-test (`.venv/bin/python hier_switch/hier_switch_hooks.py`) checks each kind
   against a trained model. What is missing is the compute: **6 seeds × 15 conditions ≈ 90
   sessions at 2–3 min each**, which is a SLURM array and needs the user's go-ahead, and then
   `hier_switch_group.py aggregate` plus the row-5 panels.

   The conditions, all on the forced low/high pairs so each has its own control:
   - **`softmax_lu0`** — the latent update scaled to zero on trials 1–4. The paper's ACC→MD
     silencing (Fig 4h); predicted to delay switching. The gradient is still computed and
     logged, so the error signal survives, as it does in the paper.
   - **`softmax_lu3`, `softmax_lu10`** — the same scaled up. **The paper never stimulated
     ACC**, so this is ours and must be labelled as ours wherever it appears.
   - **`softmax_blast`, `sigmoid_blast`** — both latent units driven to 1 at the first
     post-reversal feedback, then the gradient takes over. The MD-activation analogue
     (Fig 5d). Under the softmax this is a *reset to the uniform gate*, because the softmax
     is shift-invariant; only under the sigmoid is it also a gain boost. Run both, and say
     which is which.
   - **`softmax_mom0.5`, `softmax_mom0.9`** — momentum on the latent update for trials 1–5.
     Not a control: a question about whether a run of errors can build on itself the way the
     paper's cortical error signal does. The panels are the reversal-aligned latent, update
     and gradient (story row 5), not behaviour.
   - **Forced errors** (three flipped-feedback trials mid-block) — still not built; needs a
     dataset knob. Lowest priority.

   Scaled, clamped and momentum trials are already excluded from `z_updates` and kept in
   behaviour. The group table already has the rows (`_manipulation_rows`); they simply have
   no data yet.
2. **The weak results, if they matter to the story.** Fig 1m has no analogue here (§5b) and
   the exploration-regime signatures are weak. Both are reported as they are. If the
   integration index is to carry weight, the honest next step is more reversals per seed
   (the rc sessions are 2000 trials ≈ 45 blocks) rather than a different index.
3. **Sigmoid-*trained* models** (§6b). The gain analyses currently run on a softmax-trained
   model with the sigmoid swapped in at test, which holds the context worse (0.71 steady
   against 0.91). Training with the sigmoid from the start, now that the passive phase
   exists, is the clean version. **§5c raises the stakes on this**: cue velocity is set by
   the gain and the softmax has no gain axis, so the paper's exploration-regime signature
   cannot appear in a softmax-trained model *in principle*. A sigmoid-trained model is the
   only way to find out whether that signature is absent from the mechanism or only from
   this gate.
4. **The stateful feedback-RNN** (Brabeeba's "thalamocortical RNN trained with normal
   backprop"): `stateful_hidden=True` across trials plus an input channel carrying the
   previous outcome, truncated BPTT, and its own tuning. §5b shows what the current RNN
   lacks — it hedges on 99.9 % of test trials — so this is the baseline that would make the
   comparison fair. It is a model change: huddle before starting.
5. **Per-timestep Z within a trial** (`latent_aggregation_op='none'`) as an MDConflict
   analogue, and the free-response variant (`target_onset`) for an RT that measures
   integration directly. Both are model changes; both were deferred deliberately.

The B5 spec that used to sit here (§7a) is now `docs/hier_switch_analyses.md` §4, which
documents it as built rather than as a plan.

---

## 8. Conventions and gotchas

- **Z_lr is baked into the optimizer at construction.** Assigning `config.Z_lr` alone does
  nothing. Patch `model.Z_optimizer.param_groups` (`run_test` does). The same goes for
  `weight_decay` in `'optimizer'` mode.
- **`model.config` is the config object passed in, not a copy.** `train_model`'s test phase
  calls `reconfigure_for_prediction` on it, which freezes the weights and switches to the
  test RNG stream. Deep-copy before reusing a config.
- **`predict_first_frame=True`.** The output at t has seen frames < t. The first output
  that has seen the targets is t = 19 (`response_start_timestep`).
- **RNG streams**, keyed on `(env_seed, stream)`: active training 0, test 1, passive 2.
- **`block_size = trial_len` is bookkeeping** (one base-class "block" = one trial). The
  context blocks come from `HierSwitchDataset._context_schedule`. Everything is blocked.
- **Use the setters:** `set_timing()`, `set_n_trials()`, `set_n_passive_trials()`. A bare
  assignment leaves derived lengths stale.
- **`plot_logger_panels` has `hier_switch` branches** (`behavior`, `corrects`). Its x-axis
  is timesteps: 25 per trial.
- **`run_test` turns the latent update on when it finds it off.** For the RNN baseline and
  for any Z-clamp run, pass `test_no_of_steps_in_latent_space=0` explicitly, or a model
  trained without LU will quietly start inferring Z.
- **A model pickled before a knob existed does not have it.** `model.pt` carries its whole
  config, so anything reading a new attribute off a saved config needs a `getattr` default
  (that is why `reversal_conflict_levels` lives in the dataset module as well).
- **The RNN hedges at test, so its accuracy is not behaviour.** On 30–60-trial blocks the
  v16 baselines are undecided on 99.9 % of trials; `sign(decision)` on a near-zero output
  makes accuracy 0.5 and the switch criterion fire after ~3 coin flips. Quote the undecided
  rate with any RNN number.
- **A session with no context axis** (the RNN, or any clamped session) has no `aligned`
  trials; `steady_mask` drops the aligned filter there rather than returning an empty mask.
- **Pyright false positives.** Pyright cannot resolve the `hier_switch_*` imports
  (`extraPaths` is set; the LSP may need a restart). Treat "unknown import/attribute" on
  those modules as noise.
- **Shell.** The interactive shell is zsh: avoid nesting heredocs inside `bash -c '...'`,
  which fails to parse. Never `pkill -f` with a pattern that also appears in your own
  command line: it kills your own shell.
- **Stale artifact.** A tuning dashboard artifact
  (https://claude.ai/artifact/HUugWtSUhgcoGNR3RG5LK3) covers only v1–v6 and is out of date.

---

## 9. Working with the user

- **Check in at branch points, not every step.** A few rounds of refinement on the agreed
  knobs or analyses is fine: a narrower range, more seeds, a neighbouring value. Check in
  before starting to tune a variable nobody has put in play, or before changing the
  experiment's design. At that point, report compactly and let the user choose.
- **Standard, explainable mechanisms only.** The user wanted plain weight decay rather than
  a custom test-only decay switch, and removed an off-centre `Z_init` trick from the docs
  as not viable. New behaviour goes behind a default-off flag, and nothing gets built that
  would need special explaining in a paper.
- **Blocked design, following the study.** The flanker project's random-interleaving rule
  does *not* apply here.
- **Figures are paper panels.** Use the `plot_style.FigSize` presets (never a literal
  figsize), model colours from `plot_style.get_model_color`, and no per-model panels. See
  `docs/figure_style.md`.
- **Don't normalise away the effect under study.** Leave timing and reference frames alone;
  raise a compensation as a question rather than building it in.
- **Keep the scope minimal and verification brief.** Don't re-run a check that passed.
- **Summaries: say what you measured, and flag single-seed claims.**

---

## 10. Site policy (`/oscar/AGENTS.md`, enforced by hooks)

- **Heavy work belongs in a Slurm job.** The user usually works inside an interactive
  allocation (2 CPUs). A single ~6-minute training or a test battery can run there in the
  foreground, single-threaded (`OMP_NUM_THREADS=1`).
- **To submit compute work, show the user the sbatch script first.** Never submit a job
  array without explicit approval. `hier_switch/run_tune.sh` submits an array, so it always
  needs a go-ahead.
- **Work single-threaded:** one tool call at a time, no background jobs, no parallel
  subagents. Wrap commands in `flock -n /tmp/${USER}.agent-lock ...`, and stop if the lock
  is held.
- **Never poll.** No `watch`, and no loops around `squeue` / `sacct`. Run `squeue --me` once
  and report.
- **`--dependency=afterok:<array>` needs *every* task of that array to succeed.** One failed
  task leaves the dependent array pending on a condition that can never be met, and SLURM
  eventually cancels it. Chain arrays only when the first one is known to be reliable, or
  submit the second by hand afterwards.
- **Recorded sessions are large**: ~3 MB per 1000 trials with hidden states, so the phase-2
  exports are ~840 MB under `exports/hier_switch/` (git-ignored). The hidden states dominate;
  a session can be re-recorded from its `model.pt` at any time.
- **No recursive traversal of /oscar.** No `find`, `du`, `grep -r` or `rg` without a narrow
  explicit path. Use `ls` on specific directories.
