# Trial-history regression on flanker behaviour

What the post-error and sequential measures mean, how the human analysis is specified,
and how our version maps onto it.

Code: `flanker_regression.py`. Interactively:

    import flanker_regression as reg
    reg.describe_runs()               # every sweep on disk, with the params that differ
    reg.RUN = 'sweep_noise'           # notes on each run are in reg.SWEEP_RUNS
    summaries = reg.group_report(variant='noise10')   # signatures + M2 vs M3
    fig = reg.fig_group_coefficients(summaries)

or `python flanker_regression.py --variant noise10 [--run sweep_noise]`.
Either way each session plays the role of one participant, as in the human analysis.
`run_flanker.py` (Result 6 cell) fits a single session, which is one synthetic subject:
useful for seeing shape and mechanism, not for evidence.

Human scale for comparison: Fischer et al. have **1088 trials per participant**; the
STA_BH replication package ships 10 participants and the full dataset has 998. Our
sessions are 5000 trials, so trials per subject are not the limitation — subjects are.
Human reference: `code_shared/STA_BH/Regression_on_Behavior.m` (Fischer et al. 2018).

---

## Glossary — the names, in one line each

These effects are usually referred to by the author who introduced them, which carries no
information on its own. Look them up here rather than trying to remember them.

| name | what it means |
|---|---|
| **Gratton** (1992) | the flanker cost is smaller right after a conflicting trial — the classic conflict-adaptation effect |
| **Mayr, Awh & Laurey** (2003) | that Gratton pattern can be pure repetition priming, so you must split by whether the response repeated before calling it control |
| **Yu & Cohen** (2008) | people track whether the target keeps repeating or keeps alternating, and get faster at whichever pattern is currently running |
| **Dutilh** (2012) | the fix for measuring post-error slowing without slow drift contaminating it: compare the trial *before* the error to the trial after, rather than post-error to post-correct |
| **Ridderinkhof** (2002) | the source of PERI as a control measure — errors should selectively reduce interference, not just slow everything |
| **Fischer et al.** (2018) | the human flanker dataset (N ≈ 1300) and the behavioural regression this analysis mirrors |
| **Eriksen & Eriksen** (1974) | the original flanker task |

---

## 1. The post-error signatures

Trial A is the trial that just happened; trial B is the one being measured. All three
compare B after an **error** on A against B after a **correct** response on A. The
premise: an error signals that control was too loose, so the system should tighten.

| | what it is | human direction |
|---|---|---|
| **PES** — post-error slowing | B is slower after an error | RT(B\|A error) − RT(B\|A correct) > 0 |
| **PIA** — post-error improvement in accuracy | B is more accurate after an error | acc(B\|A error) − acc(B\|A correct) > 0 |
| **PERI** — post-error reduction of interference | the congruency effect shrinks after an error | CE(B\|A correct) − CE(B\|A error) > 0 |

PES and PIA are about the **level** of performance; PERI is about the **slope** — how
much the flankers still get in. That difference is why all three are reported together:
generalised slowing raises RT everywhere and can come from orienting or freezing, whereas
a real control adjustment should selectively reduce the flanker cost. PES alone is
ambiguous; PES **with** PIA and PERI is the control interpretation.

As regression coefficients:

```
PES  = prev_error          on log RT     (positive)
PIA  = prev_error          on accuracy   (positive)
PERI = incong:prev_error   on accuracy   (positive)  and on log RT (negative)
```

## 2. The sequential effects

No errors involved — these are about the congruency and stimulus sequence.

- **Gratton / sequential congruency** — `incong:prev_incong`. The flanker cost is smaller
  after an incongruent trial.
- **Response repetition** — `resp_rep`. The Mayr, Awh & Laurey confound: congruency
  sequences are correlated with response repetitions, and repetition priming produces a
  Gratton-shaped pattern with no control adjustment at all. It has to be *in the model*,
  not just checked in a split.
- **Target repetition** — `target_rep`. The stimulus sequence (Yu & Cohen), which comes
  apart from the response sequence exactly on error trials.
- **Lag 2** — `prev2_incong`, `prev2_error`. Not in the human core model, added because
  the control state in this model integrates over roughly three trials, so lag-2
  congruency predicts as strongly as lag-1.

---

## 3. The human specification

`Regression_on_Behavior.m`, fit per participant with MATLAB `fitglm`:

```matlab
% recoding: Congruence −1 cong / +1 incong;  Error −1 correct / +1 error
%           Distance 0→−1, so −1 close / +1 far;  RSI −1 short / +1 long
%           ACC = 1 correct;  Trnr = trial index ÷ session length

Accuracy:  ACC   ~ Congruence + Distance + RSI + PrevError + Trnr
                   + PrevError:Congruence                              (binomial)
RT:        LogRT ~ Congruence + Error + Distance + RSI + PrevError + Trnr
                   + Congruence:PrevError                              (normal)
```

Conventions worth stating, because several are easy to get backwards:

1. **The accuracy DV is ACC (1 = correct)** — positive coefficient means *more accurate*.
   Hans's DBM package models `Error` instead, so its signs are flipped relative to these.
2. **The RT model keeps error trials** and controls for accuracy with an `Error`
   regressor. It does not restrict to correct responses, and there is no RT trimming.
3. Factors are ±1 but declared `'CategoricalVars'`, so `fitglm` dummy-codes them: each
   coefficient is a **level contrast**, and main effects are simple effects at the
   reference level.
4. Group inference (`helper/AGFHK_summary.m`) is a **one-sample t-test across
   participants** on each coefficient, Bonferroni-corrected by the number of regressors,
   with a degenerate-fit guard (|t| > 1500) and the design correlation matrix stored per
   subject.

The lab's **extended** specification appears in the header comment of
`helper/AGFHK_Eval_Model.m` and is what our M2 follows:

```
LogRT ~ Congruence + Distance + RSI + Error + Hand + PostError + PostIncom + PostFar
        + RSI_Prev + PrevRT + HandRepeat + CongRepeat + RSIRepeat
        + Distance:Congruence + PostError:Congruence + PostError:HandRepeat
```

## 4. Mapping to the model

| human | ours |
|---|---|
| `LogRT` | `log(rt_interp)` — timesteps; undecided trials sit at the trial end |
| `ACC` | `correct_at_decision` |
| `Congruence` | `incong` (0/1) |
| `Distance` | `far` (0/1); note **0 = close** in their raw data |
| `PrevError` | `prev_error` |
| `Trnr` | `trial_progress` = trial index ÷ n trials |
| `Hand`, `HandRepeat` | `resp_side`, `resp_rep` |
| `PostIncom`, `PostFar`, `PrevRT` | `prev_incong`, `prev_far`, `prev_logrt_c` |
| `Target` repetition | `target_rep` |
| `expectedConflict` (fitted RL latent) | `focus_in_z` — ours is measured, not inferred |
| `RSI`, `RSIRepeat`, block resets | **no analogue** — the model has no inter-trial interval and runs one continuous block |

Because there is no RSI factor, PES here is comparable to the human result in
**direction but not magnitude**.

## 5. The three specs

Reference level is congruent / near / post-correct.

- **M1** — the human core, minus RSI.
- **M2** — plus the lab's extended terms, lag-2 history and target repetition.
- **M3** — plus `focus_in_z`, the control state the trial inherited.

**A collinearity trap worth knowing.** `cong_rep` (their `CongRepeat`) is an exact linear
combination of `{1, incong, prev_incong, incong:prev_incong}`, so it cannot sit in the
same model as the Gratton interaction. Their extended model uses `CongRepeat` *instead
of* the interaction; we keep the interaction, since that is the term the conflict-
adaptation prediction is actually about. `build_design` still computes `cong_rep` for
anyone who wants the other parameterisation.

**What M3 can and cannot show.** `focus_in` is generated by the same trial history it
competes with, so shrinkage of the history coefficients is a mediation *description*, not
a causal test — it says the control state carries the history, not that it causes the
behaviour. The causal handle is the `Z_decay` intervention, which moves the control state
without touching the trial sequence. Print the VIFs (the `collinearity()` helper) before
reading a shrinkage result: above ~5 the two are too entangled to separate.

## 6. Reading the output

**Two levels of detail.** `run_flanker.py` has a `regression_detail` flag near the top,
and `report_session(reg, detail=...)` is the entry point behind it.

- `False` (default) — the ~6 rows that are results: each human signature with its
  accuracy and RT coefficient side by side and a match / null / wrong-way verdict, then
  the M2 → M3 mediation. M1 is not even fitted, since its only job is comparability with
  the human coefficients. The figure shows the same terms plus `focus_in`.
- `True` — every model's full coefficient table and the VIF check. Most of those rows are
  **nuisance regressors** (`resp_side`, `prev_far`, `trial_progress`, `prev_logrt_c`,
  `is_error`): they are in the design to be controlled for, not to be read. Turn this on
  when auditing a fit or comparing against `code_shared/STA_BH`, not on every run.



`signature_report()` returns one row per human signature with a verdict of MATCH, null,
or WRONG DIRECTION, based on the M2 coefficient and its sign relative to the human
prediction. For a single session the p-values are trial-level; across sessions use
`fit_sessions()`, which t-tests each coefficient over sessions the way `AGFHK_summary.m`
does over participants — that is the error bar that matters, since a single session is
one synthetic subject.

---

## 7. Why PIA and PERI fail — and what fixes them

Figure: `group_7_noise_series.pdf`, built by `flanker_sweep_figures.fig_noise_series`.

`arrow_noise_std` is the SD of the per-timestep noise on each arrow against a signal of
1.0. When it is large the *target slot's own samples* frequently point the wrong way, so
two very different things cause an error: the target misled (bad luck) or the flankers won
(too little control). The latent update minimises this trial's prediction error, so it
cannot tell them apart — on a bad-luck error, attending the target *less* genuinely does
lower the error. Locally correct, globally anti-adaptive.

The noise sweep is the test, everything else held fixed. The table below is a snapshot —
**regenerate it before quoting it**, with `python flanker_sweep_figures.py` (which writes
`group_7_noise_series.pdf`) or `flanker_sweep_analysis.py` per level. What matters is the
*pattern*, which has been stable: every row moves monotonically with noise, and the
mechanism row crosses zero where the behaviour does.

| measure | noise 1.3 | 1.0 | 0.7 | 0.4 |
|---|---|---|---|---|
| share of errors that are bad luck | 0.626 | 0.605 | 0.575 | 0.524 |
| Δ focus after a bad-luck error | **−0.035** | +0.037 | +0.282 | +0.565 |
| PIA | −0.066 | −0.064 | +0.008 | **+0.120** |
| PERI | −0.059 | +0.057 | +0.279 | **+0.630** |
| conflict adaptation (lag 1) | 0.035 | 0.062 | 0.127 | 0.161 |
| PES | +0.424 | +0.191 | −0.282 | **−0.833** |

Everything moves monotonically, and the mechanism crosses zero where the behaviour does:
below about noise 1.0 a bad-luck error stops loosening control, and PIA and PERI turn
positive. Conflict adaptation quadruples over the same range.

**But PES inverts.** At low noise the model responds *faster* after an error, 0/20 seeds in
the human direction. No single noise level gives all three post-error signatures at once:
high noise buys PES at the cost of PIA and PERI, low noise the reverse. That is a real
limitation of this architecture rather than a parameter to tune — the model has no error
monitor, only a prediction-error minimiser, and the two coincide only by accident.

Two caveats on the low-noise end: the congruency effect shrinks (0.258 → 0.191) and the
distance effect disappears (−0.136 → −0.006), so the fingerprint the model was built to
reproduce weakens as the post-error signatures appear. And the accuracy regression stops
converging in 3 of 20 sessions at noise 0.4 because the congruent cells approach ceiling —
`group_report` prints a warning when that happens, and those coefficients should not be
read.
