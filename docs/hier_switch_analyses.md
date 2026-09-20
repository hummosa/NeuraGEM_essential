# hier_switch phase 2 — the analyses

Phase 1 built the task and got NeuraGEM to discover the contexts on its own
(`docs/hier_switch_task.md`). Phase 2 puts the model next to Lam et al. 2025: what each
trial does to the latent Z, what the hidden state carries and when, and how both compare
with an ideal observer. This file is the reference for the definitions; the state of the
project and what to do next are in `docs/hier_switch_handoff.md`.

Paper: Lam, Mukherjee, Wimmer, Nassar, Chen & Halassa, *Prefrontal transthalamic
uncertainty processing drives flexible switching*, Nature 637:127 (2025).

---

## 1. The pipeline

```
model.pt ──► hier_switch_test_inference.py ──► session.npz ──► load_session ──► trial_labels
  (v15 NG,      one test session per            (trials +         (dict of         (adds the
   v16 RNN)     named condition, weights        per-timestep       arrays +         trial table)
                frozen, Z restarted)            outputs,          'meta')              │
                                                pulses, hidden,                        ▼
                                                recovered dL/dZ)         behaviour / latent /
                                                                         z_updates / hidden /
                                                                         ideal_observer
                                                                                       │
                     hier_switch_group.py  ──►  results.json per session  ──────────────┘
                              │
                              ├─► aggregate: group.json + the prediction table
                              └─► hier_switch_figures.py: the panels
```

| file | what it holds |
|---|---|
| `hier_switch_analyses.py` | `load_session`, `trial_labels`, `select`, `z_updates` (B5), `behaviour` (A1-A4), `latent` (B1-B4), `normative_table`, and a synthetic self-test (`python hier_switch/hier_switch_analyses.py`) |
| `hier_switch_observer.py` | the ideal observer (E); self-check with no arguments |
| `hier_switch_hidden.py` | the hidden state X (C1-C3): axes, decoding, integration index, unit classes |
| `hier_switch_perturb.py` | the Z-clamp grid (E2) |
| `hier_switch_group.py` | one report per session, the group table, the SLURM task entry point |
| `hier_switch_figures.py` | `spec_*` panels, drawn identically for one session or the group |
| `run_sessions.sh`, `run_clamp.sh` | the two SLURM arrays |

Run it:

```bash
.venv/bin/python hier_switch/hier_switch_analyses.py          # synthetic self-test
.venv/bin/python hier_switch/hier_switch_observer.py          # observer self-check
.venv/bin/python hier_switch/hier_switch_test_inference.py <model.pt> [condition ...]
.venv/bin/python hier_switch/hier_switch_group.py list        # the model × condition grid
./hier_switch/run_sessions.sh                                 # record + analyse (SLURM array)
./hier_switch/run_clamp.sh 0-11 [after_jobid]                 # the Z-clamp grids
.venv/bin/python hier_switch/hier_switch_group.py aggregate   # the prediction table
.venv/bin/python hier_switch/hier_switch_figures.py           # group figures
.venv/bin/python hier_switch/hier_switch_figures.py <session> # the same panels, one session
```

## 2. Recording a session

`session.npz` is a superset of the old `trials.npz` (still written, for the old readers):

| key | shape | note |
|---|---|---|
| `decision, correct, cue, vis, conflict, context, since, z, z_in, phase` | (n,) | as before |
| `out` | (n, 25) | the response output at every timestep — RT comes from this |
| `pulses` | (n, 16, 3) | the noisy frames the model saw — the observer reads these |
| `hidden` | (n, 25, 64) | float16; only with `record_hidden` |
| `grad` | (n, 2) | the pooled error gradient, recovered from Z (below) |
| `lu_scale`, `clamped`, `z_momentum` | (n,) | 1 / False / 0 unless a trial hook set them |
| `grad_eff` | (n, 2) | the step actually taken, in gradient units: `−Δz/Z_lr`. Equal to `grad` under a plain SGD step; with momentum on, `grad_eff − grad` is what the velocity added |
| `meta` | json | model, condition, Z_lr, Z_decay, `Z_momentum`, activation, temperature, rt_threshold, seed, `reversal_conflict`, `perturb` (the trial hook's spec), whether Z restarted |

**Two routes to `grad`.** A recorded session now saves the gradient the optimizer actually
saw (`logger.gradients_corrections`, minus the decay term it carries), which is exact
whatever the optimizer did. The older route — recovering it from Δz — is only valid while
the step is a plain unscaled SGD one, and is still used for sessions recorded before this.
The distinction matters for exactly one condition: with the latent update scaled to zero,
Δz is zero but the gradient is not, and that surviving error signal is the whole point of
the manipulation.

Three facts the whole pipeline rests on:

- **One plain-SGD latent step per trial, no momentum**, so
  `z = z_in − Z_lr·(g + Z_decay·z_in)` exactly, and the pooled error gradient comes back
  from the saved Z alone: `g = −(z − z_in)/Z_lr − Z_decay·z_in`. Checked against
  `logger.gradients_corrections` (which already contains the decay term): they agree to
  7e-12, i.e. 1e-6 of the gradient's own size.
- **`z` is logged after the trial's own update**; `z_in` is what the trial ran under.
- **The per-timestep hidden-state recorder is default-off and changes nothing.** With
  `record_hidden` on and off the predicted outputs are bit-identical (asserted), because
  only the acting (weight-update) forward is recorded and the latent re-forwards are not.

**Conditions** (`hier_switch_test_inference.CONDITIONS`): `softmax` as trained; the
`nosoftmax_*` and `sigmoid_*` ladders from the phase-1 open thread; the paired
`*_rc_none / _rc_low / _rc_high` triplets; and `rnn` for the v16 baselines.

**The paired triplets** are the paper's controlled reversals. With
`config.reversal_conflict = 'low' | 'high'`, the first 5 trials of every *test* block are
forced to 7:2 or 6:3, the paper's two levels. The level is drawn as usual and then
overridden, so the RNG stream is untouched and every other trial of the session is
identical to the unforced partner's — the three sessions are paired trial by trial. The
training stream never sees it (asserted in the dataset self-check).

## 3. The trial table (`trial_labels`)

`select(sess, **crit)` composes masks from these; a scalar matches by equality, a list by
membership, a `(lo, hi)` tuple is an inclusive range, a callable gets the array. The
start-up block of a Z-restarted session is dropped unless `include_transient=True`.

| field | definition |
|---|---|
| `block`, `block_len`, `pos_from_end` | block index (a block starts at `since == 1` or at a phase boundary), its length, trials left in it |
| `reversal`, `transient`, `complete` | the block began with a real reversal; it is the start-up block of a restarted session; it ended in a reversal rather than being truncated |
| `err`, `rule`, `conf_level` | `~correct`; `cue × ctx_sign` (+1 = attend vision); 0-4 |
| `axis`, `mid`, `d` | the context axis: the direction between the two contexts' mean `z_in` over steady-state trials (`since ≥ 11`) of the primary phase, the midpoint's projection, and the prototype distance |
| `proj_in`, `held`, `aligned`, `stale` | `z_in` on that axis; the side it sits on (−1 if Z is NaN or there is no axis); whether that side is the true context |
| `z_evidence` | `(proj_in − mid)/(d/2)` signed toward the true context: +1 on the true prototype, −1 on the other, 0 in the middle. The graded `aligned` |
| `contrast_in`, `gain_in`, `gate_*` | `(z₀ − z₁)/2` and `mean(z)` of `z_in`, raw and in gate units |
| `rt`, `decided` | `flanker_analyses._interpolated_rt` on `out` from t = 19, threshold `meta['rt_threshold']` (0.5); an undecided trial sits at the trial end (25) and is counted, never dropped |
| `early_conf`, `early_class` | per block: the mean conflict of trials 1-5; 0 = low / 1 = high, the forced level when `reversal_conflict` is set, else a median split over the phase's reversal blocks |
| `switch_trial` | per block, the paper's criterion: the first correct trial with another correct within the next two. `z_switch_trial`: the first trial Z holds the true context. `dec_switch_trial`: the paper's, on decided trials only |
| `pre_switch` | `since < switch_trial` (all of a block that never switched) |
| `p_c`, `err_w`, `eps_cw` | the session's own steady-state accuracy at that conflict level, clipped at 0.99; `err/(1 − p_c)`; its mean over the previous 5 trials — the paper's ε_CW, not reset at reversals (the animal's ACC does not know where they are) |
| `ctx_rel`, `last_trained_context` | the context **relative to training**: 0 = the one the model saw in the last block of the active phase, 1 = the other. Which context is labelled "0" is an accident of the data stream, but the model is not symmetric about it — it leaves training with its weights and its Z sitting in the last block's context, and at the uniform gate it falls back on a default context. Every per-context split is reported on `ctx_rel`. Read from the sibling `trials.npz` of `model.pt` by `last_trained_context()`; falls back to the raw label, with a warning, when that file is gone. Across the six NG seeds it is context 1 for s0, s3, s5 and context 0 for s1, s6, s9, so pooling on the raw label would average two different things |

**Why the undecided rate travels with accuracy.** With Z at the middle of the axis the
outputs sit near 0 and `sign(decision)` is a coin flip, so "accuracy crossed 0.5" can mean
hedging rather than switching. Every reversal-aligned accuracy panel has the undecided
rate and the Z-side curve beside it.

## 4. B5 — how each trial moves Z (`z_updates`)

Split the update into the part weight decay would have made anyway and the part this
trial's error made:

```
dz       = z − z_in
dz_decay = −Z_lr·Z_decay·z_in          (toward the middle gate)
dz_err   = dz − dz_decay
s        = (dz_err · axis)/d, signed +1 toward the context Z was NOT holding
tipped   = the full update moved z across the midpoint
dgain    = Δ mean(z); dgate_gain = Δ mean(gate(z))
```

`s` is the fraction of the distance between the two context prototypes that one trial's
error covered. Trials are excluded when Z is NaN, the session has no axis, the latent
update was off or scaled, or Z was clamped.

Cells: aligned/stale × error/correct × conflict level (× context, which should mirror).
`z_update_table` returns them. **Under the softmax `dgain` is identically zero** — the
gradient sums to zero across the two units and decay only shrinks the mean — so it is
reported once as a check (measured max |Δgain| ≈ 6e-7) and the gain analyses run on the
sigmoid-at-test ladder, where the axis is live.

**The normative comparison** (`normative_table`) puts the model's mean `s` per cell next to
the ideal observer's log-odds update for the same trials, signed the same way (toward the
context *Z* was not holding, not the one the observer doubted).

## 5. The ideal observer (`hier_switch_observer.py`)

Two stages, both on the trials the model actually saw:

1. **Cue posterior from the pulse frames.** Each frame is a noisy one-hot, so its
   likelihood under type k is `N(x; e_k, σ²I)`; the observer does not know the conflict
   level, so it marginalises over the five with a frame-wise (i.i.d.) mixture. That is the
   version quoted in the tuning log: **the cue is read correctly on 93 % of trials at
   σ = 0.5**, by level 1.00 / .995 / .99 / .93 / .74. The likelihood is slightly
   misspecified — the generator shuffles a fixed multiset rather than drawing frames
   independently — which shows up only as understated confidence at high conflict; the
   cue it picks is unaffected.
2. **Context filter.** Hazard 1/45, predictive belief `b = (1−h)p + h(1−p)`, feedback
   likelihood `P(correct side | context) = q or 1 − q` (the two contexts' likelihoods sum
   to 1, so the feedback's log-odds update is ±log(q/(1−q))).

It returns the per-trial cue posterior, the belief before and after the feedback, that
log-odds update, its own choice and switch latency, and `p_c` per conflict level — the
accuracy ceiling, and the paper's `P_corrcue`.

## 6. Behaviour (A1-A4), latent (B1-B4), hidden (C1-C3)

- **A1** psychometric: accuracy, RT and the undecided rate by conflict, per context, steady
  state, against the observer's ceiling.
- **A2** reversal-aligned accuracy / Z-side / |decision| / RT / undecided from 5 trials
  before to 15 after, split by `early_class`; switch latency under all three criteria,
  with the observer's as the normative size of the effect.
- **A3** perseverative errors and trials to a 3-in-a-row criterion
  (`mean_prediction_analysis._find_criterion`).
- **A4** RT by conflict and by trials since the reversal, and the fast-error check (early
  errors vs steady-state errors), with a decided-only companion.
- **B1** Z on the context axis against the observer's belief: correlation, best lag, and
  both curves around a reversal.
- **B2** Z uncertainty (`1 − |z_evidence|`): peak height, position and half-width after a
  reversal by `early_class` (Fig 1k,l), and its correlation with conflict (Fig 1m) —
  per trial, with the previous trial's conflict, and per block (peak against the block's
  early conflict), since the model's Z carries the *past* trials' conflict, not this one's.
- **B3** which of `err`, `err × conflict` and `err × the observer's log-odds` explains `s`.
- **B4** the pooled |dL/dZ| against `err/(1 − p_c)` per trial, and its 5-trial mean against
  ε_CW: the ACC analogue. The contrast component `|g₀ − g₁|/2` is the one the softmax can
  act on; the gain component is reported beside it.
- **C1** per-timestep cross-validated logistic decoding of cue, rule, context and conflict
  from the hidden state, steady state vs the first 5 trials after a reversal; build-up =
  the first timestep at 0.75.
- **C2** the paper's three classes per hidden unit: cue selectivity is fitted inside each
  context (where cue and rule coincide), signs that agree across contexts mean the unit
  tracks the cue and signs that disagree mean it tracks the rule, and the fraction of
  selectivity in the second half of the cue period splits CueS from CueL. A
  label-permutation null decides which units are tuned at all.
- **C3** the cue and rule axes' magnitude around a reversal, and the paper's **integration
  index** (activity along the orthogonalised CueL-and-Rule dimension, second half of the
  cue period over the first) and **cue velocity** (the fastest rise along the cue axis in
  the first half), per condition.

**Timing.** `predict_first_frame=True`, so the output and hidden state at step t have seen
input frames < t. Pulses are frames 0-15 → the cue period of the hidden state is steps
1-16, its first half 1-8 and its second 9-16; the response window is 19-24.

**Cue and rule are collinear inside a block** (`rule = cue × ctx_sign`), so every axis fit
and every decoder pools trials from both contexts. A per-context refit is what tests P5
(the cue axis agrees across contexts, the rule axis flips).

## 6b. What is encoded where (`hier_switch_hidden.encoding_table`)

The paper's central representational claim is that PFC mixes task variables while MD
demixes them: most MD neurons are selective to *one* of cueing conflict or rule context,
most PFC neurons to several (Fig 2k–l, 2o–p). We do not take that claim on, and we do not
try to reproduce its PFC cell classes. We ask the same question our own way, on the
signals this model actually has, and report the answer whichever way it falls.

**Sources** — the per-trial feature matrices we ask "what does this carry?" of:

| source | what it is | the paper's counterpart |
|---|---|---|
| `hidden_t16` | the 64 hidden units at the end of the cue period | PFC |
| `hidden_t24` | the same at the end of the trial | PFC |
| `hidden_pc2` | the top **two** principal components of `hidden_t16` | the control (below) |
| `z_in` | the latent the trial ran under — the only thing that crosses trials | MDContext |
| `step` | `z − z_in`, the trial's own latent update | MD's transient switch response |
| `grad` | the error gradient on Z, as `[g_contrast, |g_contrast|]` | the ACC error signal |

**Why `hidden_pc2` is there.** A 64-unit hidden state will out-decode a 2-unit Z on almost
anything, simply by having 32× the dimensions to do it with. That is the objection to
reading the paper's PFC-vs-MD comparison at face value, so the same hidden state cut down
to two dimensions is carried in every panel: a hidden-vs-Z difference that survives against
`hidden_pc2` is not a unit-count effect. It is fitted on the same trials, and drawn hollow.

**Variables**: cue, rule, context (`ctx_rel`), conflict, outcome, and the paper's two
uncertainties taken from the **ideal observer** rather than from the model, so they are
properties of the trial sequence and not of the thing being decoded — rule uncertainty
`1 − |2b − 1|` on the observer's predictive belief, cue uncertainty `1 − max(q, 1 − q)` on
its cue posterior.

**Two measures per (source, variable).**

1. **Decoding.** Cross-validated balanced accuracy (binary variables) or ridge R²
   (continuous), 5 folds, **with a shuffled-label null computed per cell** — so "above
   chance" is measured rather than assumed. Conflict is decoded the way the paper contrasts
   it: the two most ambiguous levels against the two least, dropping the middle.
2. **Demixed variance.** Every column of the source is fitted jointly on all the variables
   at once; dropping one variable and re-fitting gives the variance **uniquely**
   attributable to it. Variance two variables share is credited to neither and is reported
   as `shared`; what nothing explains is `residual`. The parts sum to 1, so a stacked bar
   reads directly: one tall segment means demixed, several means mixed. For the hidden
   state there is also a per-unit count of how many variables a unit is tuned to against a
   permutation null — the paper's Fig 2k histogram.

**One limit to state before reading the uncertainty columns.** Every decoder here is
linear, and both uncertainties are *magnitudes* — rule uncertainty is `1 − |2b − 1|`, cue
uncertainty is `1 − max(q, 1 − q)`. A signed two-unit source cannot produce a magnitude
under a linear map, so Z scoring ~0 on rule uncertainty is a statement about the decoder,
not about Z: the same quantity measured properly, as `1 − |z_evidence|`, does rise after a
reversal and is wider after a high-conflict start (§6, B2). Read the uncertainty columns as
"can a linear read-out of this signal recover it", and take the sign-carrying columns —
cue, rule, context, outcome — as the substantive comparison.

Run over **every** trial of the primary phase, not only the steady state: the trials right
after a reversal are where the latent update and its gradient do their work, and a
steady-state-only table would leave the error signal almost nothing to carry. The
steady-state version is returned beside it as `decoding_steady`.

## 6c. Trial hooks — the paper's optogenetics (`hier_switch_hooks.py`)

Lam et al. manipulate the switch itself: ACC→MD terminals silenced during the feedback of
the first four post-reversal trials (switching is delayed, Fig 4h), and MD driven during
the feedback of the first five trials after a **high-conflict** reversal (switching speeds
up and PFC cue velocity rises, Fig 5d,g). Both act on a few trials at a known position in
the block and then stop.

A `TrialHook` is exactly that. `predictive_learning` calls `hook.pre()` before each trial's
latent update and `hook.post()` after it; `config.trial_hook` is absent everywhere else, so
**nothing that ran before this existed changes**. The spec is a plain dict, so it travels in
the session's meta:

| spec | what it does | the paper |
|---|---|---|
| `dict(kind='lu_scale', k=0, trials=[1, 4])` | multiplies the latent learning rate | ACC→MD silencing |
| `dict(kind='lu_scale', k=3 or 10, trials=[1, 5])` | the same, upward | **ours, not theirs** — the paper never stimulated ACC |
| `dict(kind='z_set', z=[1, 1], trials=[1, 1])` | drives both latent units at the first feedback, then lets the gradient take over | MD activation (SSFO) |
| `dict(kind='momentum', mu=0.9, trials=[1, 5])` | gives the latent update a memory for the window | — |

Three things to know.

- **`k = 0` does not silence the gradient.** It zeroes the *step*; the gradient is still
  computed, pooled and logged, which is why the session saves the logged gradient rather
  than recovering it from Δz. In the paper too, the ACC error signal survives the silencing
  of its output to MD.
- **Z = (1, 1) is not the same move under the two gates.** The softmax is shift-invariant,
  so (1, 1) is the *uniform* gate: a reset to maximal uncertainty. Under the sigmoid both
  units open to 0.73, which is a genuine gain boost. Both are run, and each panel says which.
- **Momentum's buffer is cleared when a window opens**, because Z is one Parameter for the
  whole session and would otherwise carry velocity from the previous reversal.

Trials the hook touched are marked `lu_scale`, `clamped` and `z_momentum`, and `z_updates`
excludes them — a scaled, clamped or momentum-carrying Δz is not a clean measurement of
"what this trial's error taught Z". They stay in behaviour, which is what the manipulation
is asking about.

**Momentum is a question, not a control.** Z here is persistent where the paper's MD switch
response is transient, and the paper's ACC signal builds up over consecutive errors, which
a memoryless gradient cannot do. Momentum is the smallest change that would let the latent
update build up the same way, so the panels show what it does to Z, to the update and to
the gradient around a reversal, rather than comparing it against a matched control.

## 7. E2 — the Z clamp (`hier_switch_perturb.py`)

Weights frozen, latent update off (`test_no_of_steps_in_latent_space=0`), Z held at a
chosen raw value `(m + d, m − d)` for a whole session: `m` is the gain, `d` the contrast.
Each cell's metrics can be rebuilt from the saved sessions without re-running the model:
`hier_switch_perturb.py recompute [tag] [activation]`.
The softmax grid moves `d` only (it has no gain direction); the sigmoid grid moves both,
including the literal (1, 1) the "MD activation" analogue asks for. Each cell is a session
like any other, so the same behaviour and hidden-state functions read it, and every metric
is split by whether the clamped gate matches the block's context. The context keeps
reversing inside a clamped session, so every cell tests its gate against both contexts; that
is a within-session control rather than a confound, because the context label never enters
the input (it only picks the answer key), so the two halves differ in scoring alone. Split
the behavioural measures, pool the hidden-state ones. **Which context a clamped
gate selects is read off behaviour** — the context the model is more accurate in — with both
contexts' accuracies reported beside it (`acc_ctx`). Geometry was tried first and does not
work: the gain direction takes a sigmoid cell off the context axis, and the raw middle gate
(0.5, 0.5) makes the model behave as context 0 rather than sitting between the two. Picking
the larger of two numbers biases a cell that is at chance in both contexts upward by ~0.02,
which is why the raw pair is always printed.

## 8. The group level

`hier_switch_group.py` runs every model × condition (6 NG seeds that discovered the
contexts, 10 RNN baselines) and writes one `results.json` per session; `aggregate` turns
them into `exports/hier_switch/group/group.json` and prints the prediction table. Every row
is a difference or a slope whose sign a prediction fixes, reported as **the mean, the SEM,
the per-seed values, and how many seeds carry the predicted sign** — the flanker lesson,
where single-session effects repeatedly failed to survive across seeds.

Figures: `hier_switch_figures.py`, one `spec_*` builder per panel, taking a list of per-seed
reports. A single session is a list of one, so the single-session and group figures are the
same panels. Every figure is captioned in `docs/hier_switch_methods.md`.

**`story_figure()` is the one to look at**: five rows of four panels (behaviour, what is
encoded where, the clamped gate, the reversal, the latent signals), written both as one
`story.pdf` and as a PDF per row so a row can be reworked on its own. It grows a sixth row,
the manipulations, as soon as those sessions are on disk. `group_figures()` writes it last,
along with the supplementary figures — the panels the story does not carry. Two earlier
figures were folded into it: `switching.pdf` is no longer written and `behaviour.pdf` lost
its reversal-aligned RT panel. **No builder was deleted**, so either figure can be brought
back in one line.

**Styling rules the panels follow** (on top of `docs/figure_style.md`): hue is the model
(`plot_style.get_model_color`); **outcome has a hue and a marker of its own** — correct teal
circles solid, error crimson triangles dashed (`plot_style.outcome_line`), because a dashed
line alone cannot be read in a paper-sized legend; early-reversal conflict is a shade of the
model's hue (lighter = low, full = high, dashed for high); every mean carries SEM and
per-seed dots. Legend handles are 2.4 long so a dash shows as a dash.

---

## 9. The toolbox — what a new analysis can call

Everything below is importable from `hier_switch/` (each module puts the repo root and its
own directory on `sys.path`, so `sys.path.insert(0, 'hier_switch')` then a plain import
works from the repo root). Run with `.venv/bin/python`.

### Data already on disk

| path | what it is |
|---|---|
| `exports/hier_switch/tune_v15/NG_s{0,1,3,5,6,9}/model.pt` | the six trained NG networks that discovered the contexts (whole pickled models; `.config` travels with them) |
| `exports/hier_switch/tune_v16/RNN_s{0..9}/model.pt` | the ten backprop baselines, saved *before* their plastic-weight test phase |
| `exports/hier_switch/tune_v13/NG_s{0..9}/trials.npz` | the original 10-seed run: per-trial arrays only (no hidden states, no pulses) |
| `exports/hier_switch/inference_tests/<tag>/<condition>/session.npz` | a recorded test session — everything in §2. `<tag>` is e.g. `tune_v15_NG_s3` |
| `…/<condition>/results.json` | every analysis number for that session (`session_report`) |
| `exports/hier_switch/clamp/<tag>/<cell>/session.npz` | one clamped session per grid cell, plus `clamp_grid_{softmax,sigmoid}.json` |
| `exports/hier_switch/group/group.json`, `…/figures/*.pdf` | the group table and the nine figures |

`hier_switch_group.models()` lists the model × condition grid; `session_dirs(model_type,
seed, conditions)` turns a row into paths. The manipulation conditions
(`NG_MANIPULATIONS`) are off unless `HIER_SWITCH_MANIP=1`, so `list` and `task` keep
describing what is on disk; `hier_switch_test_inference.main` skips any condition whose
`session.npz` already exists unless `HIER_SWITCH_FORCE=1`.

### Reading and labelling

```python
from hier_switch_analyses import load_session, trial_labels, select, PRIMARY, STEADY
sess = trial_labels(load_session('exports/hier_switch/inference_tests/tune_v15_NG_s0/softmax_rc_none'))
m = select(sess, phase=PRIMARY, err=True, stale=True, conf_level=[3, 4])
```

- `load_session(path)` — session.npz, trials.npz or a folder; returns a dict of per-trial
  arrays plus `n` and `meta`. The single reader; nothing else should open an npz.
- `trial_labels(sess)` — adds the trial table (§3) in place and returns it.
- `select(sess, **crit)` — boolean mask; scalar = equality, list = membership, `(lo, hi)` =
  inclusive range, callable = predicate. Drops the start-up block unless
  `include_transient=True`.
- `gate(z, activation, temp)`, `recover_grad(z, z_in, Z_lr, Z_decay)` — the two conversions
  every analysis needs.

### Analyses (all numpy, all take a labelled session)

| call | returns |
|---|---|
| `z_updates(sess)` | per-trial `dz`, `dz_err`, `dz_decay`, `s`, `tipped`, `dgain`, `dgate_gain`, `ok` |
| `z_update_table(sess, upd, by=('state','err','conf_level'))` | those quantities per cell, with n and SEM |
| `behaviour(sess)` | psychometric, reversal-aligned curves, the three switch latencies, RT |
| `reversal_aligned(sess, values, window=(-5,15), blocks=None)` | any per-trial array averaged around reversals |
| `latent(sess, obs, upd)` | belief vs observer, uncertainty peaks, update-rule regressions, ε_CW |
| `normative_table(sess, upd, obs)` | the model's update against the observer's, per cell |
| `hier_switch_observer.ideal_observer(sess)` | cue posterior, context belief, its own choices, `p_c`, switch latency |
| `hier_switch_hidden.hidden_report(sess, obs=…)` | axes, per-timestep decoding, integration index, cue velocity, unit classes, the encoding table, the reversal-aligned regime measures |
| `hier_switch_hidden.encoding_table(sess, obs)` | §6b: every source × variable, decoding with its own null plus demixed variance |
| `hier_switch_hidden.integration_aligned(sess, axes)` | the integration index and cue velocity at each trial offset from a reversal (the paper's Fig 3c cut) |
| `switch_by_early_conf(sess, obs)` | the paper's Fig 1f on uncontrolled reversals: switch latency per equal-count bin of the block's early conflict |
| `hier_switch_hidden.fit_axes / decode / integration / unit_classes / steady_mask` | the pieces, if a new analysis wants them directly |
| `hier_switch_hidden.z_side_table(sess, upd)` | what Z carries (context d′) and what its update carries (conflict) |
| `hier_switch_group.session_report(path)` | all of the above for one session, as a json-able dict |

### Running new sessions and perturbations

```python
from hier_switch_train import load_model, run_test          # needs torch
model, cfg = load_model('exports/hier_switch/tune_v15/NG_s0/model.pt')
logger, m, tcfg = run_test(model, cfg, run_name='scratch/probe', n_trials=1000,
                           record_hidden=True, **overrides)
```

- `run_test(model, cfg, Z_lr=…, n_trials=…, run_name=…, perturb=None, **latent)` — a test
  session on a **copy**: weights frozen, Z restarted at `Z_init`, any latent setting
  overridden. It patches the live optimizer, which never re-reads the config. `perturb` is
  a trial-hook spec (§6c) and defaults to None.
- Useful overrides: `latent_activation='sigmoid'|'none'`, `Z_init=[a, b]` (per unit),
  `test_no_of_steps_in_latent_space=0` (freeze Z — **pass it explicitly**, or `run_test`
  turns the latent update on), `record_hidden=True`, `reversal_conflict='low'|'high'`,
  `Z_decay=…`, `rt_threshold=…`.
- `hier_switch_test_inference.CONDITIONS` / `test_condition(...)` — the named conditions and
  the recorder that writes `session.npz`; `DEFAULTS[model_type]` is what runs with no names.
- `hier_switch_perturb.clamp_grid(model_path, activation, n_trials, grid, reference)` — a
  whole clamp grid; `run_cell` for one cell; `cell_metrics(path, ref_axis)` for its numbers;
  `recompute(tag, activation)` rebuilds the tables from saved sessions without re-running
  anything.
- `hier_switch_group.analyse(paths)` writes `results.json` beside each session;
  `aggregate()` writes `group.json` and prints the prediction table.

### Figures

```python
from hier_switch_figures import figure, spec_psychometric, spec_gain, band, series, bars
figure([spec_psychometric({'NeuraGEM': reports})], 'out/panel.pdf')
```

Builders take **a list of per-seed reports** (a single session is a list of one):
`spec_psychometric_ctx`, `spec_switch_vs_early_conflict`, `spec_decoding_matrix`,
`spec_encoding_variance`, `spec_decoding_timecourse`, `spec_integration_reversal`,
`spec_trace`, and the earlier
`spec_psychometric`, `spec_reversal`, `spec_switch`, `spec_switch_cost`, `spec_z_belief`,
`spec_z_uncertainty`, `spec_eps_cw`, `spec_gain`, `spec_gain_ladder`, `spec_gain_cost`,
`spec_rt`, `spec_rt_reversal`, `spec_decoding`, `spec_integration`, `spec_unit_classes`,
`spec_clamp_grid`, `spec_clamp_gain`, and the retired `spec_z_update`, `spec_tipped`,
`spec_normative`. Primitives: `band`, `series`, `bars` (with `clusters` and `gap_after`),
`legend`, `shade`, `split_style`, `stack(reports, 'dotted.path')`, `figure(panels, path)`.

### Entry points

```bash
.venv/bin/python hier_switch/hier_switch_analyses.py            # synthetic self-test
.venv/bin/python hier_switch/hier_switch_observer.py            # observer self-check
.venv/bin/python hier_switch/hier_switch_dataset.py             # generator self-check
.venv/bin/python hier_switch/hier_switch_hidden.py <session>    # one session's hidden-state report
.venv/bin/python hier_switch/hier_switch_test_inference.py <model.pt> [condition ...]
.venv/bin/python hier_switch/hier_switch_group.py list|task|analyse|aggregate
.venv/bin/python hier_switch/hier_switch_perturb.py list|task|recompute
.venv/bin/python hier_switch/hier_switch_figures.py [session dir]
.venv/bin/python hier_switch/hier_switch_figures.py story      # only the story figure
.venv/bin/python hier_switch/hier_switch_hooks.py              # trial-hook self-test
./hier_switch/run_manipulations.sh                             # the manipulation sessions
./hier_switch/run_sessions.sh [range]      ./hier_switch/run_clamp.sh [range] [after_jobid]
./hier_switch/run_tune.sh <grid tag> [range]
```

### Things that will bite a new analysis

- **The seed is the unit.** Six NG seeds, ten RNN. Report per-seed values and how many carry
  the predicted sign; single-session effects have failed to survive here before.
- **The RNN hedges on 99.9 % of test trials**, so its accuracy (~0.50) and its switch latency
  are coin flips on a near-zero output. Always quote its undecided rate.
- **A clamped or RNN session has no context axis**, so `aligned`, `s` and `tipped` do not
  exist there; `steady_mask` drops the aligned filter for them and `z_updates` refuses
  outright on a session with the latent update off.
- **Z_lr and weight decay are baked into the optimizer** at construction: patch
  `model.Z_optimizer.param_groups`, which `run_test` does for you.
- **`model.config` is not a copy** — deep-copy before reusing a config.
- **Site policy** (`/oscar/AGENTS.md`): training and sweeps go to Slurm, show the sbatch
  script and get approval before any array, never poll the scheduler, no unscoped traversal
  of `/oscar`. A single recorded session (~1 min per 1000 trials) or any analysis over saved
  sessions runs inline, single-threaded (`OMP_NUM_THREADS=1`).
