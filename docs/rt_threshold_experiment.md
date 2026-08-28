# Brief: does `rt_threshold` change the conclusions?

## The one fact that shapes this whole task

`rt_threshold` is a **post-hoc analysis parameter, not a simulation parameter.** It is
applied by `flanker_analyses.extract_trials()` to the `train_logger` stored in each result
pickle. Nothing in Stage 1 or Stage 2 depends on it.

**So this experiment requires no retraining and no SLURM jobs.** 400 result pickles are
already on disk and every threshold can be re-applied to them. If you find yourself
submitting jobs or calling `train_model`, stop — you have misread the task.

It is also **not only an RT knob.** The threshold defines the decision point, so it sets
`correct_at_decision` too. Every one of the 11 human signatures depends on it, accuracy
ones included.

## Read these, in this order

| # | What | Why |
|---|------|-----|
| 1 | `flanker_analyses.py` lines 90–130, `_interpolated_rt` | **The most important read.** Documents the crossing rule and the two conventions that were tried and rejected. A trial that never crosses is given `rt = arrows_duration` rather than dropped — that pile-up is the mechanism you are testing. |
| 2 | `flanker_analyses.py` lines 158–200, `extract_trials` | What the threshold produces: `correct_at_decision`, `rt_interp`, `decided`, `cross_idx`. |
| 3 | `run_flanker.py` lines 17–36 (docstring) | The three analysis conventions, stated as project policy. |
| 4 | `flanker_metrics.py` — `condition_masks`, `session_effects`, `SIGNATURES` | The 11 signatures and the sign each should take to match humans. `SIGNATURES` is the scorecard. |
| 5 | `flanker_figure_utils.py` lines 69–90 | `collect_sessions(variant, rt_threshold=...)` and `collect_effects(...)` **already take the threshold as an argument.** Use it. |
| 6 | `flanker_sweep_config.py` | `RT_THRESHOLD = 0.5`, the four `ARMS`, the 5-level `NOISE_LADDER`. |
| 7 | `exports/flanker_random/factorial_corr_jitter/README.md` | What the four runs on disk are. |
| 8 | `docs/flanker_task.md` | Task and model background, if the above is not enough. |

Skip `flanker_near_cong_diagnostic.py` — it is a large standalone diagnostic and not
relevant here.

## The data

    exports/flanker_random/sweeps/factorial_<arm>/<variant>/results_seed-<0..19>.pkl

4 arms (`nojit_pc52`, `jit_pc52`, `nojit_pc58`, `jit_pc58`) x 5 noise levels
(`noise13/10/09/07/04`) x 20 seeds = 400 pickles, ~8.5 MB each. `nojit_pc52` is the
baseline. Load with `flanker_sweep.load_condition(variant)` inside
`with flanker_sweep.use_run(f'factorial_{arm}'):`.

## The experiment

For each (arm, noise level, threshold), recompute the per-seed effects and the scorecard:

1. Thresholds: `0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8` (extend if the edges look interesting).
2. **Load each pickle once**, then loop thresholds inside — unpickling dominates the cost,
   `extract_trials` is cheap. Loading per threshold instead is ~7x slower for no reason.
3. Per cell compute: the undecided fraction (`1 - decided.mean()`), the 11 signature values
   per seed, and each signature's verdict (PASS / null / OPP) by the same rule
   `flanker_sweep_figures.fig_scorecard` uses — Cohen's d across seeds, signed by the human
   direction, verdict from whether the 95% CI clears zero.

Budget roughly 10–20 minutes for all 400 x 7. Use `.venv/bin/python` — the system python
has no torch.

## What a pilot already found

Baseline arm (`factorial_nojit_pc52`), `noise09`, 20 seeds. Reproduce this first as a
correctness check before going wider:

| thr | undecided | matched | opp | cong_acc | cong_rt | pes_BI | pia_BI | peri |
|-----|-----------|---------|-----|----------|---------|--------|--------|------|
| 0.20 | 0.009 | 7 | 1 | 0.2104 | 0.3206 | 0.0223 | −0.0486 | 0.0179 |
| 0.35 | 0.043 | 8 | 1 | 0.2058 | 0.6446 | 0.0620 | −0.0567 | 0.0525 |
| 0.50 | 0.098 | **9** | 1 | 0.2020 | 0.9666 | 0.1080 | −0.0678 | 0.0734 |
| 0.65 | 0.167 | **9** | 1 | 0.1959 | 1.2461 | 0.1156 | −0.0785 | 0.1124 |
| 0.80 | 0.253 | 8 | 1 | 0.1897 | 1.4315 | 0.0133 | −0.0822 | 0.2189 |

Three signatures **change verdict** across this range:

| signature | 0.20 | 0.35 | 0.50 | 0.65 | 0.80 |
|-----------|------|------|------|------|------|
| `dist_effect_acc_cong` | PASS | PASS | PASS | PASS | null |
| `dist_effect_rt_incong` | null | PASS | PASS | PASS | PASS |
| `peri` | null | null | PASS | PASS | PASS |

So this is worth doing: at least one reported conclusion is threshold-dependent.

## Hypotheses to test

- **H1 — accuracy is robust, RT is not.** `cong_effect_acc` moves 0.210 → 0.190 (−10%)
  while `cong_effect_rt` moves 0.321 → 1.432 (**4.5x**) over the same range. Confirm this
  split holds across arms and noise levels.
- **H2 — the RT effects are carried by the undecided pile-up.** Undecided goes 0.9% → 25.3%,
  and those trials are all assigned `rt = arrows_duration`. Recompute the RT signatures on
  **decided trials only** and see how much of the threshold dependence survives. If most of
  it does not, the RT congruency effect is substantially a non-response effect wearing an RT
  costume. Note `_interpolated_rt`'s docstring says dropping them was rejected because it
  biases incongruent RT downward — so decided-only is a diagnostic, not a proposed fix.
- **H3 — PERI's verdict is threshold-dependent.** null at ≤0.35, PASS at ≥0.5, and the value
  moves 0.018 → 0.219 (12x). PERI is the measure that separates the jitter arms from the
  no-jitter arms, so if its verdict is a threshold artifact, the conclusion "jitter reverses
  PERI" needs re-examining. **Test this on all four arms**, not just the baseline.
- **H4 — is 0.5 at a maximum?** The scorecard count peaks at 0.5–0.65 and is lower at both
  ends. Check whether that peak is real and whether it holds across arms and noise levels.
  If the chosen value happens to maximise the score, say so plainly.
- **H5 — is a fixed absolute threshold fair across arms?** Jitter trains the read-out to emit
  the same magnitude across gate sharpness, and the jitter arms run a higher mean focus
  (0.342 → 0.391). If the `|output|` distribution differs by arm, the same absolute threshold
  is not the same decision criterion in each, and cross-arm comparisons at a fixed 0.5 are
  confounded. Plot the distribution of `max|output_traj|` per arm before concluding anything
  cross-arm. Consider a per-session threshold set at a fixed quantile of that distribution as
  a robustness check.
- **H6 — noise interacts.** At `arrow_noise_std` 0.4 accuracy is near ceiling, so RT carries
  the signal; threshold choice may matter more there. Check whether sensitivity varies along
  the ladder.

## Deliverables

1. A new standalone script, e.g. `flanker_rt_threshold_sweep.py`, that does the above and
   writes results to a **new** folder (`exports/flanker_random/rt_threshold/`). Do not
   overwrite the figures in the existing run directories.
2. A figure: signature effect size vs threshold, and undecided fraction vs threshold.
3. A flip table: every (arm, noise, signature) whose verdict changes over the range, so the
   fragile conclusions are enumerated rather than described.
4. A one-paragraph recommendation: keep 0.5, move it, or report results with a robustness
   band — with the evidence for whichever it is.

## Guardrails

- **Do not retrain or resubmit anything.** This is pure re-analysis of existing pickles.
- **Do not change `RT_THRESHOLD` in `flanker_sweep_config.py`** as the mechanism. It is the
  global default for every figure and analysis script; editing it would silently restate all
  existing figures at a new threshold. Pass `rt_threshold=` explicitly instead — the
  collectors already accept it.
- Use `.venv/bin/python` (the repo venv). The system python has no torch/numpy.
- Report what you find, including "0.5 is fine" if that is the answer. A negative result here
  is a genuine result: it would let the project stop worrying about this parameter.
