# Where the model stands against human flanker behaviour

Run `sweep_td03_n10`: SGD latent optimizer, `temporal_decay_factor` 0.3, `arrow_noise_std`
1.0, spatial gradient 0.75/0.52, p(congruent)=0.5, 10 seeds. Each seed is a synthetic
subject: every effect is computed within a session, then one-sample t-tested across the 10.

**How to read the columns.** *model* is the mean across seeds with its standard error.
*p* tests it against zero. *seeds* counts how many of the 10 point in the direction human
data shows — an effect carried by 9/10 seeds is a different animal from one carried by
5/10 with two outliers doing the work. *verdict* is **match** (significant, right
direction), **WRONG** (significant, wrong direction), or null.

Units: accuracy effects are proportions; RT effects are in timesteps, of which a trial has
four usable ones. Control-state effects are in units of the softmax gate, where 0 means
attention spread evenly over all five slots.

| measure | what it is | model (mean ± SEM) | p | seeds | verdict |
|---|---|---|---|---|---|
| **BEHAVIOURAL FINGERPRINT** | | | | | |
| Congruency effect (accuracy) | accuracy on congruent minus incongruent trials (proportion) | +0.211 ± 0.016 | 0.0000 | 10/10 | **match** |
| Congruency effect (RT) | RT on incongruent minus congruent trials (timesteps) | +1.232 ± 0.092 | 0.0000 | 10/10 | **match** |
|   … same, decided trials only | excludes trials that never crossed threshold (timesteps) | +0.402 ± 0.027 | 0.0000 | 10/10 | **match** |
| Distance effect, accuracy (incongruent) | near minus far flankers, within incongruent trials (proportion) | -0.067 ± 0.033 | 0.0717 | 8/10 | null |
| Distance effect, accuracy (congruent) | near minus far flankers, within congruent trials (proportion) | -0.001 ± 0.008 | 0.8719 | 4/10 | null |
| Distance effect, RT (incongruent) | near minus far flankers; near should interfere more, so be slower (timesteps) | +0.224 ± 0.105 | 0.0615 | 7/10 | null |
| Proportion-congruent modulation | congruency effect at p(cong)=0.8 minus at 0.2, same models (proportion) | +0.110 ± 0.011 | 0.0000 | 10/10 | **match** |
| **SEQUENTIAL EFFECTS** | | | | | |
| Conflict adaptation, lag 1 | accuracy on incongruent trials after incongruent minus after congruent (proportion) | +0.051 ± 0.010 | 0.0005 | 9/10 | **match** |
| Conflict adaptation, lag 2 | same, but keyed to the trial two back (proportion) | +0.034 ± 0.006 | 0.0003 | 10/10 | **match** |
| Gratton effect, response repeats | congruency-effect reduction after conflict, when the response repeated (proportion) | +0.095 ± 0.012 | 0.0000 | 10/10 | **match** |
| Gratton effect, response switches | same on switch trials — the control-not-priming test (proportion) | +0.065 ± 0.012 | 0.0003 | 10/10 | **match** |
| **POST-ERROR SIGNATURES** | | | | | |
| PES — post-error slowing | RT after an error minus after a correct trial (timesteps) | +0.280 ± 0.103 | 0.0236 | 7/10 | **match** |
| PIA — post-error accuracy gain | accuracy after an error minus after a correct trial (proportion) | -0.053 ± 0.012 | 0.0018 | 1/10 | **WRONG** |
| PERI — post-error interference drop | congruency effect after correct minus after error (timesteps) | +0.092 ± 0.047 | 0.0795 | 9/10 | null |
| **MECHANISM (model-only)** | | | | | |
| Control state (Z focus) | centre-slot gate weight minus mean flanker weight; 0 = uniform attention (0–1) | +0.297 ± 0.013 |  |  | — |
| Update after a flanker-driven error | change in focus when the target was clear but flankers won (0–1) | +0.155 ± 0.013 | 0.0000 | 10/10 | **match** |
| Update after a noise-driven error | change in focus when the target slot itself misled (0–1) | +0.040 ± 0.018 | 0.0497 | 7/10 | **match** |
| Share of errors that are noise-driven | median split on target-slot evidence; 0.5 = unrelated to errors (proportion) | +0.599 ± 0.007 |  |  | — |
| Trials that never cross threshold | their RT is extrapolated and capped at 10 (proportion) | +0.095 ± 0.005 |  |  | — |
| Overall accuracy | all trials (proportion) | +0.825 ± 0.008 |  |  | — |

## The one clear failure: PIA

Post-error accuracy is the only measure pointing reliably the wrong way (−0.053, 1/10
seeds). `pia_circularity.pdf` shows why, in four panels:

1. **The deficit precedes the error.** Control is already low two trials before an error
   (0.279 vs 0.304) and lowest on the error trial itself (0.236 vs 0.321). Errors happen
   *because* control had drifted down — they do not create the deficit, they reveal it.
2. **The error's own correction undershoots.** It updates the control state by +0.086,
   which is larger than the +0.053 a correct trial produces, but it starts 0.085 behind
   and so closes only 39% of the gap.
3. **The next trial therefore starts behind** (0.322 vs 0.373) and is less accurate
   (0.791 vs 0.838) — which is the negative PIA.
4. **The exchange rate**: on incongruent trials, accuracy rises steeply with inherited
   control over exactly this range, so a residual gap of 0.05 costs ~5 accuracy points.

The deficit is gone within two trials (lag +2: 0.312 vs 0.323), so this is a
recovery-rate problem, not a permanent one.

**What would fix it** — all of these make the post-error update larger or faster, and the
group should weigh them, since they differ in how much theory they import:

- a larger `Z_lr`, or more latent-update steps per trial — cheapest, no new mechanism
- an error-gated learning rate — makes the model's control adjustment explicitly
  error-driven, which is the conflict-monitoring assumption stated rather than emergent
- an explicit conflict representation that rises after an error and feeds the latent

Note the trade-off already visible: lowering `arrow_noise_std` from 1.3 to 1.0 fixed the
direction of the post-error update and moved PERI across zero, but weakened the
distance-on-accuracy effect from −0.086 (p=0.03) to −0.067 (p=0.07).
