# Hierarchical cue→rule reversal — handoff: from tuning to neural analyses

You are picking up a modelling project whose first phase (build the task, get NeuraGEM to
learn it and to discover the contexts on its own) is done. The next phase is the one this
document is for: **systematically build the analyses that compare the model's behaviour,
latent and hidden-state activity to the paper.** Everything here was true on 2026-09-19;
check the code before relying on a detail.

Read, in this order:
1. this file;
2. `docs/hier_switch_task.md` — the task, the model, and the full tuning log (v1–v14);
3. `hier_switch/hier_switch_config.py` — every knob, with the reason for its value;
4. `hier_switch/hier_switch_train.py` and `hier_switch/hier_switch_analyses.py`;
5. `docs/figure_style.md`, and `docs/flanker_task.md` §"Five analysis conventions" — the
   flanker project is the house precedent for how analyses and figures are built here.

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
| E1 | **Core session.** Train (passive, then active; discovery) → frozen-weight test on the paper's 30–60 blocks | ✅ Machinery works. 6/10 seeds discover (v13). Sessions are **not saved** (§6) |
| E2 | **Z-clamp probe.** Freeze weights and the latent update; `set_Z` along p(A) from 1 → 0.5 → 0; run a fixed trial battery at every conflict level. Measure RT, accuracy, undecided rate, and the build-up rate of the cue and rule axes. This is the direct test of "does context uncertainty modulate integration and output speed" | ⬜ |
| E3 | **Group sweep.** ≥ 10 seeds × {NG, RNN, Oracle → inference} plus the ideal observer; optionally a `pulse_noise_std` ladder | 🟡 `hier_switch_tune.py` can run seeds; no group analysis or figures yet |
| E4 | **Perturbations.** Latent update off for the first 4 post-reversal trials (ACC→MD silencing); Z_lr ×k for the first 5 trials of high-conflict reversals (MD activation); forced errors (3 flipped-feedback trials mid-block), with and without a Z_lr boost | ⬜ Needs a per-trial Z_lr / LU-on-off schedule hook in `train_and_infer_functions.predictive_learning`, default-off. The flanker "error-gated Z_lr" deferred design (`docs/flanker_task.md`) is the same hook |

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

| | Analysis | Status |
|---|---|---|
| A1 | Psychometric curve: accuracy by conflict × context | 🟡 `summarize()` → `acc_conf` (numbers, no figure) |
| A2 | Reversal-aligned accuracy (−5…+15 trials), split by early-reversal conflict (median split of the first 3–5 trials' conflict) | 🟡 `acc_since` bins only; no split |
| A3 | Perseverative errors, trials-to-criterion, switch trial | 🟡 `cross_trial` (first trial whose 3-trial mean reaches chance). Reuse `mean_prediction_analysis._find_criterion` for criterion-based measures |
| A4 | RT and undecided rate by conflict and reversal-aligned; fast-error check with a `_decided` companion | ⬜ Reuse `flanker_analyses._interpolated_rt` with `search_from = config.response_start_timestep` (19) |
| B1 | Z belief against the ideal-observer posterior around reversals | ⬜ |
| B2 | Z-entropy peak height and width, split by early-reversal conflict | ⬜ |
| B3 | Update rule: regress ΔZ toward the true context on error × cue strength | ⬜ |
| B4 | \|dL/dZ\| against ε_CW | ⬜ |
| C1 | Per-timestep cross-validated decoding of cue and rule from the hidden state; build-up rate = integration speed; split by conflict and by inherited Z entropy | ⬜ Needs per-timestep hidden states (§5, step 2) |
| C2 | Per-unit CueS / CueL / Rule classification; PCA dimensionality of Z against the hidden state | ⬜ |
| C3 | Rule and cue axis magnitude reversal-aligned | ⬜ |
| D | Z clamp (E2), perturbations (E4) | ⬜ |
| E | Ideal observer (numpy). Cue posterior from the noisy pulse frames (a frame-wise likelihood reaches 0.923 at `pulse_noise_std` 0.5; a plain HP−LP count gets 0.890); feedback likelihood; hazard step 1/45; report the *predictive* belief. Precedent: `rotation_slips_perseveration_analysis.ideal_observer_trials` | 🟡 The accuracy ceiling was computed ad hoc; there is no observer module |

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

These are v13's verified phase lengths, the configuration behind the 6/10 below. A longer
passive phase (2 × 3000) with 4000 active trials is an untested alternative.
3. `'Inference only'` — **test**: 1000 trials on 30–60-trial blocks. Weights frozen; Z
   inferred, continuing from where training left it.

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
| steady-state accuracy | trials 11+ into a block |
| `cross_trial` | first trial after a reversal whose 3-trial mean accuracy reaches 0.5 |
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
| **latent_activation** | softmax only. `'none'` collapses to a dead gate (Z → 0); `'sigmoid'` turns Z into a gain, not a context code |
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

## 6. How to run things

```bash
.venv/bin/python hier_switch/hier_switch_dataset.py            # generator self-check
.venv/bin/python hier_switch/hier_switch_train.py NG 0         # train + test + panels, seed 0
./hier_switch/run_tune.sh <tag>                                # SLURM array from GRIDS[tag]
.venv/bin/python hier_switch/hier_switch_tune.py arms <tag>    # seeds grouped by setting
.venv/bin/python hier_switch/hier_switch_tune.py collect <tag>  # one row per run
.venv/bin/python hier_switch/show_blocks.py <tag> [run ...]    # per-block convergence
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
  - `run(model_type, seed, run_name=..., **overrides)` → `(logger, model, cfg)` via
    `train_model`.
  - `run_test(model, cfg, Z_lr=..., run_name=..., **latent)` runs a test session on a
    **copy** of a trained model with any latent settings. It patches the live optimizer's
    lr and weight_decay; this is how the Oracle Z_lr sweeps were done.
  - `extract_trials(logger, cfg)` → per-trial dict. `summarize(trials, phase, last_frac)`,
    `block_table(trials, phase)`.
- **Tuning.** `hier_switch_tune.py` holds every grid (`GRIDS`, v1–v14) as a record. An entry
  may carry `tests=[...]` to run several test sessions on one trained model.

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

## 7. What to build next — suggested order

1. **Analysis-ready sessions.** `run(..., save_model=True)` now writes
   `<export_path>/model.pt`; reload it with `load_model(path)`. Grid `v15` retrains the six
   v13 discoverers (seeds 0, 1, 3, 5, 6, 9, at v13's phase lengths) with saving on.
   - **Seed 0 is trained and saved** (`exports/hier_switch/tune_v15/NG_s0/model.pt`). It
     reproduced v13 exactly (test 0.832, steady 0.91).
   - The other five need the v15 SLURM array. Under the site policy, show the user the
     sbatch script first.
   - `hier_switch_test_inference.py <model.pt> [condition ...]` runs named inference
     conditions on copies of a saved model.

   Before this, `run()` passed `save_models=False`, and nothing trained was on disk except
   `trials.npz`, `summary.json` and figures under `exports/hier_switch/tune_*`.
   - Write a script that trains the discovering seeds (v13: 0, 1, 3, 5, 6, 9) with the
     defaults and saves the model and config into the run's export path:
     `torch.save(model, path)`, and `torch.load(..., weights_only=False)` to reload.
   - Do **not** use `train_model`'s own `save_models=True`. Its file name is keyed only on
     seed, length and `experiment_to_run`, so different runs overwrite each other.
   - Also keep the test-phase logger, or re-create it with `run_test` on the saved model.
2. **Per-timestep hidden states.** `Logger.hidden_states` keeps only the final state per
   batch.
   - The plan: a default-off recorder in `models.RNN_with_latent.forward` (append `h` per
     step when an attribute such as `hidden_trace` is set; ~3 lines). Plus a replay helper
     that re-runs the frozen model over a logged test session, each trial under its `z_in`.
   - **The replay must reproduce the logged outputs to float precision.** Check that before
     trusting any hidden-state analysis.
3. **Behaviour (A1–A4)**, including RT and the undecided rate, and the reversal-aligned split
   by early-reversal conflict (P1, P2, P6).
4. **Ideal observer (E).** This gives the normative P2 effect size and B1's posterior.
5. **Latent (B1–B4).** The ACC/MD analogues (P3, P4).
6. **Hidden state (C1–C3).** The PFC analogues (P5).
7. **Causal.**
   - The E2 clamp needs no new plumbing: `set_Z`, LU steps 0, weights frozen.
   - E4 needs the per-trial LU schedule hook (default-off; keep existing runs unchanged).
8. **Group level (E3).** ≥ 10 seeds per model. Report the seed count with the predicted sign
   next to every mean (flanker lesson: single-session effects repeatedly failed to survive
   across seeds). Build each figure panel once as a `spec_*` builder, drawn by both the
   single-session and the group script (see `flanker_figure_utils.py`).

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
- **Pyright false positives.** Pyright cannot resolve the `hier_switch_*` imports
  (`extraPaths` is set; the LSP may need a restart). Treat "unknown import/attribute" on
  those modules as noise.
- **Shell.** Never `pkill -f` with a pattern that also appears in your own command line: it
  kills your own shell.
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
