# Findings: does `rt_threshold` change the conclusions?

Answer to the brief in [`rt_threshold_experiment.md`](rt_threshold_experiment.md). Everything
below comes from re-analysing the 400 result pickles already on disk at seven absolute
thresholds and three per-session quantile thresholds. Nothing was retrained.

Reproduce with:

    .venv/bin/python flanker_rt_threshold_sweep.py all       # tables + figures, ~4 min
    .venv/bin/python flanker_rt_threshold_sweep.py figures   # the project's own figures,
                                                             # one set per threshold, ~6 min

Outputs in `exports/flanker_random/rt_threshold/`: `effects.csv` (4000 rows), `verdicts.csv`,
`flips.csv`, `matched_counts.csv`, and five figures.

A read-only version of this document with the figures inline:
<https://claude.ai/code/artifact/32b0ab50-2dbb-406d-909d-f1828cd5ebed>

---

## Recommendation

**Keep 0.5. Report three signatures with a robustness band, and amend two sentences in
`flanker_sweep_config.py` that are true at 0.5 but not across the range.**

0.5 survives the two tests that would have condemned it. It is **not** a maximum of the
scorecard — summed over all 20 (arm x noise) cells the matched count is 154, 154, 151, 152,
152, 155, 157 at thresholds 0.2 … 0.8, so 0.5 is joint-third-*lowest* and 0.8 is highest.
Whatever else it is, it is not a value tuned to flatter the model, and the project can say
so. It also puts the undecided rate at ~10%, which is a defensible non-response rate, and
the cross-arm conclusions that the factorial exists to support hold across the whole range
and survive a criterion-equalising control. What does not survive is a handful of
*within-arm* significance calls, all of them on measures already sitting near zero: `peri`,
`dist_effect_rt_incong` and `dist_effect_acc_cong` account for 26 of the 34 verdict changes.
Those three should be reported with the range, not with a single number and a star.

Two concrete changes are worth making beyond that. **Report `cong_effect_rt_decided` next to
`cong_effect_rt`** — it already exists in `rt_outcome_effects`, is currently unused by any
figure, and is the honest denominator for the single most threshold-sensitive number in the
project (see H2). And **prefer the regression's history terms to the cell contrasts** where
both are available: the regression's PERI is significant and human-signed at all seven
thresholds, so the one genuinely load-bearing fragility disappears under the operationalisation
the project already has (see below).

---

## Scale of the problem

Across 4 arms x 5 noise levels x 11 signatures = 220 cells, scored at 7 thresholds by
`fig_scorecard`'s own rule:

| | |
|---|---|
| cells whose verdict is constant over 0.2–0.8 | **186 / 220 (85%)** |
| cells that change verdict | 34 (15%) |
| changes that are PASS ↔ n.s. (a significance call softening) | 33 |
| changes that reverse sign, PASS ↔ OPP | **1** |

The single sign reversal is `nojit_pc58`, `noise10`, `peri` — a cell that is
`n.s. n.s. OPP OPP n.s. n.s. PASS` across the range, i.e. noise, not a finding.

The pilot in the brief is reproduced exactly, to four decimals, including `matched` =
7/8/9/9/8 and the three flipping signatures. The verdict rule used here imports
`flanker_sweep_figures._effect_size` rather than reimplementing it, so it is the scorecard's
rule by construction — which also confirms the pilot's reimplementation was correct.

---

## The hypotheses

### H1 — accuracy is robust, RT is not. **Confirmed, with one exception.**

Median relative swing, |value(0.8) − value(0.2)| / |value(0.5)|, over all (arm, noise) cells:

| kind | median swing |
|---|---|
| accuracy signatures | **0.19** |
| RT signatures | **1.08** |

A 5.7x difference. The exception is `dist_effect_acc_cong`, an accuracy measure with the
largest swing of all (1.62) — but only because it is tiny everywhere (0.007 at 0.5). It is
fragile for the ordinary reason a near-zero effect is fragile, not because of the threshold.

### H2 — the RT effects are carried by the undecided pile-up. **Confirmed, and this is the mechanism behind almost all of the fragility.**

Recomputing each RT signature on decided trials only, and comparing how much of its
0.2 → 0.8 swing survives (baseline arm, all noise levels):

| signature | all trials | decided only | share of the swing that is the pile-up |
|---|---|---|---|
| `cong_effect_rt` | 0.327 → 1.350 | 0.237 → 0.340 | **90%** |
| `dist_effect_rt_incong` | 0.016 → 0.153 | 0.028 → 0.070 | 69% |
| `peri` | 0.054 → 0.284 | 0.039 → 0.144 | 55% |
| `pes_BI` | −0.021 → −0.169 | −0.016 → −0.146 | 12% |

Nine tenths of the congruency effect's threshold dependence is trials that never responded,
pinned at `rt = arrows_duration`. On decided trials only, `cong_effect_rt`,
`dist_effect_rt_incong` and `peri` are **PASS at every threshold** — the instability is
manufactured by the convention, not present in the reaction times.

This is a diagnostic, not a proposed fix: `_interpolated_rt` documents that dropping those
trials was tried and rejected because it biases incongruent RT downward. The truth is
between the two columns, which is exactly why both should be reported.

`pes_BI` is the exception and deserves a note: only 12% of its swing is the pile-up, and on
decided trials it goes **OPP at thresholds ≥ 0.6**. That is a genuine RT effect changing
sign, not a convention artifact.

### H3 — PERI's verdict is threshold-dependent. **Confirmed for the baseline's own verdict; the cross-arm conclusion is robust.**

These are two different claims and they come apart:

- **"Jitter reverses PERI."** Robust. At `noise09` the sign pattern (baseline positive,
  jitter negative) holds at **7 of 7** thresholds. The between-arm difference
  (`nojit_pc52` − `jit_pc52`) is significant by Welch t at **every threshold at 4 of the 5
  noise levels**; the exception is `noise13`, where it is significant at none — there both
  arms are negative. The reported +0.073 → −0.188 at `noise09` is the middle of a band that
  runs +0.018/−0.067 at 0.2 to +0.219/−0.045 at 0.8, and never crosses.
- **"The baseline arm matches human PERI at `noise09`."** Not robust. `n.s.` at 0.2–0.4,
  `PASS` at 0.5–0.8. This is the claim the brief was right to flag.

PERI is the most flip-prone signature in the set (10 of 34 flips), so it earns a band
wherever it is reported.

### H4 — is 0.5 at a maximum? **No. Cleanly no.**

Summed matched counts over all 20 (arm, noise) cells:

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|
| matched | 154 | 154 | 151 | **152** | 152 | 155 | 157 |
| opposite | 31 | 30 | 31 | 31 | 31 | 30 | 29 |

The pilot's apparent peak at 0.5–0.65 was specific to the baseline arm at `noise09`. Across
the factorial the curve is flat to within noise and if anything tilts *up* toward 0.8. This
is a negative result and a useful one: the project can state that its analysis threshold is
not sitting on a maximum of its own scorecard.

### H5 — is a fixed absolute threshold fair across arms? **The confound is real, measurable, and too small to matter.**

The prediction was right in direction. The jitter arms run a larger decision variable, so an
absolute 0.5 is a slightly *laxer* criterion there:

| | undecided at abs. 0.5 | threshold needed for 10% undecided |
|---|---|---|
| `jit_pc52` | 0.092 | 0.521 |
| `jit_pc58` | 0.085 | 0.540 |
| `nojit_pc52` | **0.110** | 0.477 |
| `nojit_pc58` | 0.102 | 0.495 |

So cross-arm comparisons at a fixed 0.5 are scored about 2 percentage points of undecided
rate apart — worth roughly ±0.03 in threshold units.

Rescoring everything at a **per-session quantile threshold**, so every session is held at
the same undecided rate rather than the same absolute level, leaves the PERI pattern
completely unchanged: jitter arms OPP at `noise09/10/13` and PASS at `noise04/07`, baseline
PASS at `noise09`, at every quantile tried (5%, 10%, 20%). The criterion difference is not
what produces the jitter result.

### H6 — noise interacts. **Yes, and the sensitive region is the middle of the ladder, not the clean end.**

Verdict flips per noise level (out of 44 = 4 arms x 11 signatures):

| noise | 1.3 | 1.0 | 0.9 | 0.7 | 0.4 |
|---|---|---|---|---|---|
| flips | 3 | 4 | **10** | **10** | 7 |

The brief guessed that `noise04` would be most sensitive because accuracy is near ceiling
there and RT carries the signal. It is `noise09` and `noise07` instead — the mid-ladder,
which is where the post-error signatures turn over and effects sit closest to the
significance boundary. Threshold choice matters most exactly where the interesting
transition is, which is a reason for care rather than alarm.

---

## The flip table

Full version in `exports/flanker_random/rt_threshold/flips.csv`. Concentration by signature
(out of 20 = 4 arms x 5 noise levels):

| signature | flips |
|---|---|
| `peri` | 10 |
| `dist_effect_rt_incong` | 10 |
| `dist_effect_acc_cong` | 6 |
| `pes_BI` | 3 |
| `dist_effect_acc_incong` | 2 |
| `pia_BI`, `sce_acc_repeat`, `sce_acc_switch` | 1 each |
| `cong_effect_acc`, `cong_effect_rt`, `lag2_contrast_acc` | **0** |

The three headline signatures — the congruency effect on accuracy and on RT, and the lag-2
contrast — never change verdict anywhere in the factorial.

At the headline cell (`nojit_pc52`, `noise09`):

| signature | at 0.5 | range over 0.2–0.8 | verdict stable? |
|---|---|---|---|
| `cong_effect_acc` | 0.2020 | 0.190 – 0.210 | yes (PASS) |
| `cong_effect_rt` | 0.9666 | 0.321 – 1.432 | yes (PASS) |
| `dist_effect_acc_incong` | −0.1196 | −0.120 – −0.114 | yes (PASS) |
| `dist_effect_acc_cong` | 0.0110 | 0.005 – 0.029 | **no** |
| `dist_effect_rt_incong` | 0.1374 | 0.038 – 0.233 | **no** |
| `pes_BI` | 0.1080 | 0.013 – 0.109 | yes (n.s.) |
| `pia_BI` | −0.0678 | −0.082 – −0.049 | yes (OPP) |
| `peri` | 0.0734 | 0.018 – 0.219 | **no** |
| `sce_acc_repeat` | 0.1261 | 0.119 – 0.126 | yes (PASS) |
| `sce_acc_switch` | 0.1246 | 0.111 – 0.126 | yes (PASS) |
| `lag2_contrast_acc` | 0.0396 | 0.032 – 0.049 | yes (PASS) |

---

## Written claims that need amending

These are in the `WHAT THIS FACTORIAL FOUND` block of `flanker_sweep_config.py`. All were
checked at `noise09` unless stated.

| claim | status |
|---|---|
| "9 of 11 signatures matched without jitter, 8 with" | **holds only at 0.5–0.6.** The baseline runs 7, 8, 8, 9, 9, 9, 8 across 0.2 → 0.8; jitter runs 8, 8, 8, 8, 8, 9, 9. |
| "Across the whole ladder jitter never matches MORE signatures than the baseline at any level" | **violated in 3 of 35 (noise x threshold) cells** — `noise09` at 0.2 and 0.8, `noise07` at 0.8. Holds at all 5 noise levels at the default 0.5. |
| "It REVERSES PERI, +0.073 → −0.188" | **robust.** Sign pattern holds at 7/7 thresholds and survives the quantile control. |
| "pushes the incongruent RT distance effect out of significance, +0.137 → +0.097" | **holds only at 0.3–0.6.** At 0.7–0.8 the jitter arm passes too, so the contrast dissolves. |
| "Its one real gain is post-error slowing, pes_BI 0.108 (n.s.) → 0.375" | **robust.** Jitter PASS and baseline n.s. at every threshold. |
| "raises the control state, mean focus 0.342 → 0.391" | **robust by construction** — `focus_all` is computed from `z_act` and the threshold never touches it. |

The pattern is consistent: statements about *jitter's mechanism* survive; statements that
*count signatures* do not, because the count is a sum of eleven significance calls and three
of those calls sit near the boundary.

---

## The regression tells the same story, and is steadier

    .venv/bin/python flanker_rt_threshold_sweep.py figures

redraws the project's *own* figures once per threshold (see below). The regression forest is
the interesting one, because it is an independent operationalisation of the same signatures:
PERI is the `incong:prev_error` coefficient rather than a difference of four cell means, and
the model controls for congruency, previous congruency, response repetition and time on task
simultaneously.

`t` across the 20 sessions, baseline arm at `noise09`, spec M2. Human-consistent signs are
`+` on accuracy and `−` on log RT for PERI, `+` on both for post-error:

| term | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|
| `incong` (RT) | 18.4 | 33.6 | 43.1 | 45.8 | 47.8 | 44.1 | 40.8 |
| `incong:far` (RT) | −5.5 | −7.6 | −6.9 | −6.9 | −6.7 | −6.6 | −6.3 |
| `prev_error` (RT) | 2.2 | 3.5 | 3.9 | 4.6 | 5.3 | 5.6 | 5.0 |
| **`incong:prev_error` (RT)** | **−3.1** | **−5.2** | **−5.3** | **−6.6** | **−7.5** | **−7.9** | **−9.0** |
| `incong:prev_error` (acc) | −1.8 | −0.9 | −0.5 | 0.1 | 0.0 | 0.0 | 0.5 |

Three things worth noting:

- **PERI on RT is significant and human-signed at all seven thresholds.** The cell-contrast
  version is n.s. below 0.5. So "PERI's verdict is threshold-dependent" is specific to the
  cell-contrast operationalisation; the regression does not share the problem.
- The same holds for the other two flip-prone signatures. `incong:far` on RT is significant
  at every threshold, where the cell contrast `dist_effect_rt_incong` flips. And
  `prev_error` on RT (PES) is significant at every threshold, where the cell contrast
  `pes_BI` is n.s. at every threshold — a standing disagreement between the two methods that
  has nothing to do with the threshold, but is worth knowing about.
- `incong` on RT still moves 2.5x across the range (18 → 46 → 41), so the regression does
  **not** escape H2. The pile-up still inflates the congruency effect; what the regression
  buys is stability in the *history* terms, which are the ones the project's conclusions
  actually turn on.

Coefficients for every term, spec and threshold are in `regression_coefficients.csv`.

## Figures

`figures` mode writes the project's own figures at each threshold into
`exports/flanker_random/rt_threshold/by_threshold/`, one folder, the threshold in every
filename, so sorting by name gives a flip-through series:

| pattern | count | what |
|---|---|---|
| `scorecard_<arm>_<variant>_thr<X>.pdf` | 35 | `fig_scorecard`, all 5 noise levels x 7 thresholds |
| `noise_series_<arm>_thr<X>.pdf` | 7 | `fig_noise_series`, the 8-panel noise ladder |
| `regression_<arm>_<variant>_thr<X>.pdf` | 7 | `fig_group_coefficients`, the forest plot |

Defaults to the baseline arm; pass another (`figures jit_pc52`) to switch. The regression
runs for one noise level only (`noise09`) because fitting is ~0.9 s per session per spec —
the full ladder would be ~20 minutes against ~4 for one level.

Every figure carries the threshold in its title and a corner stamp, so one pulled out of the
series still says which one it is. The scorecard and noise-ladder figures are the project's
own functions called unmodified; only `collect_effects` is redirected to serve per-threshold
effects, and that substitution was checked to be bit-identical to the normal loader at 0.5
across all 132 measures and 20 seeds.

The threshold-on-an-axis figures, in `exports/flanker_random/rt_threshold/`:

| file | what it shows |
|---|---|
| `fig_1_undecided.pdf` | undecided fraction vs threshold, by noise level and by arm — the size of the pile-up each threshold manufactures |
| `fig_2_signatures.pdf` | all 11 signatures' effect size vs threshold, one line per arm, with the non-significance band shaded. The PERI panel is the one to look at: blue above zero and red below it across the whole range |
| `fig_3_matched.pdf` | scorecard count vs threshold — the H4 evidence |
| `fig_4_decided_only.pdf` | the H2 result, and the clearest panel in the set: solid (all trials) climbs, dashed (decided only) is flat |
| `fig_5_amplitude.pdf` | where 0.5 falls in each arm's `max|output|` distribution, and PERI rescored at a matched undecided rate |
