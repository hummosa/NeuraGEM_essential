# Where the model stands against human flanker behaviour

This file says **how to read the scorecard and what the standing conclusions are**. It
deliberately holds no numbers: every effect here is re-estimated whenever a sweep is
re-run, and a table pasted into a document goes stale silently. Generate the current
state instead:

```bash
python flanker_sweep_analysis.py                    # across-seed tables
python flanker_sweep_figures.py --variant noise10   # group_6_scorecard.pdf and the rest
python flanker_regression.py --variant noise10      # the same signatures as GLM coefficients
```

`flanker_sweep.describe_runs()` lists what is on disk and the parameters each run actually
used. `flanker_metrics.SIGNATURES` is the registry of benchmark effects and the sign each
should take in human data — the single source of truth for any pass/fail table.

---

## How to read the numbers

Each seed is a synthetic subject: every effect is computed *within* a session, then
one-sample t-tested across seeds. Report four things together, never the mean alone:

| | |
|---|---|
| **mean ± SEM** | across seeds, not across trials |
| **p** | one-sample t-test against zero |
| **seeds with the predicted sign** | an effect carried by 9/10 seeds is a different animal from one carried by 5/10 with two outliers doing the work |
| **verdict** | match (significant, right direction), WRONG (significant, wrong direction), or null |

Units: accuracy effects are proportions; RT effects are in timesteps, of which a trial has
only a handful of usable ones; control-state effects are in units of the softmax gate,
where 0 means attention spread evenly over all five slots.

Always label which measure and which trials — accuracy or RT? congruent or incongruent?
Ambiguity there has caused real confusion in this project.

---

## What holds

**The behavioural fingerprint.** The congruency effect on both accuracy and RT is large
and carried by every seed, and the list-wide proportion-congruent modulation goes the
classic way — control relaxes when conflict is rare. These are solid.

**Conflict adaptation.** The sequential congruency effect is present at lag 1 and lag 2,
and survives the response-repetition control (Mayr, Awh & Laurey 2003), so it is not pure
feature-integration priming. The lag-2 contrast being comparable to lag-1 is a model
prediction worth testing in the human data: the control state integrates over roughly
three trials.

**Post-error slowing** appears, but see below — PES alone is ambiguous.

## What fails, and why

**The distance effect is weak.** Near flankers should hurt more than far ones on
incongruent trials. The only thing that teaches the model near ≠ far is the gap between
`p_corr_by_distance[1]` and `[2]` in Stage 1, so the size of that gap bounds the effect.
The effect is also conditional on the list: it strengthens as congruent trials become
common, because the controller *reallocates* weight onto the near slots rather than simply
becoming more or less focused. That conditionality is itself the finding.

**PIA and PERI go the wrong way at high stimulus noise.** The mechanism is a real property
of the architecture, not a tuning accident:

1. The control deficit **precedes** the error. Focus is already low before an error and
   lowest on the error trial. Errors do not create the deficit, they reveal it.
2. The error's own correction **undershoots**: it updates focus more than a correct trial
   does, but starts further behind, so it closes only part of the gap.
3. The next trial therefore starts behind and is less accurate — a negative PIA.
4. On incongruent trials accuracy rises steeply with inherited control over exactly this
   range, so a small residual gap costs several accuracy points.

The deficit is gone within about two trials, so this is a recovery-rate problem rather
than a permanent one.

**The deeper reason** is that the latent update minimises *this trial's* prediction error,
which cannot distinguish two very different errors: the flankers won (attend the target
more) versus the target slot's own samples misled (attending the target less genuinely
does lower this trial's error). The second kind is locally correct and globally
anti-adaptive. `arrow_noise_std` sets how often it happens, which is why the sweep is
built on that axis — see `flanker_regression.md` §7 for the series and its conclusion.

**No single noise level gives all three post-error signatures.** High noise buys PES at
the cost of PIA and PERI; low noise reverses that and PES inverts. Treat this as a
limitation of the architecture — the model has a prediction-error minimiser, not an error
monitor — rather than a parameter left untuned. Note the trade-off in the other direction
too: the congruency and distance effects weaken as the post-error signatures appear.

## What would change it

All of these make the post-error update larger or faster, and they differ in how much
theory they import, so the group should weigh them rather than pick the cheapest:

- a larger `Z_lr`, or more latent-update steps per trial — cheapest, no new mechanism
- an error-gated learning rate — makes the control adjustment explicitly error-driven,
  i.e. states the conflict-monitoring assumption rather than deriving it
- an explicit conflict representation that rises after an error and feeds the latent
