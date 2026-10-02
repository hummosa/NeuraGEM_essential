# Where the model stands against human flanker behaviour

This file says **how to read the scorecard and what the standing conclusions are**. It
deliberately holds no numbers: every effect here is re-estimated whenever a sweep is
re-run, and a table pasted into a document goes stale silently. Generate the current
state instead:

```bash
python flanker_sweep_analysis.py                    # across-seed tables, default variant (delay1)
python flanker_sweep_analysis.py noise19            # any other rung
python flanker_sweep_figures.py                     # group_7_scorecard.pdf and the rest
python flanker_regression.py --variant delay1       # the same signatures as GLM coefficients
```

`flanker_sweep.describe_runs()` lists what is on disk and the parameters each run actually
used. `flanker_metrics.SIGNATURES` is the registry of benchmark effects and the sign each
should take in human data — the single source of truth for any pass/fail table.

**Which model this describes.** The current sweep, `ad10_delay`: 10-timestep trials (9
response steps), a **raw gate** (`latent_activation = 'none'`), no background noise, and two
ladders — target-onset delay 0/1/2/4 at `arrow_noise_std` 1.35, and noise 1.9/1.35/1.0/0.6
at delay 0. Anything measured on the 400 `factorial_*` pickles (5-timestep trials, softmax
gate) is the *retired* model; its conclusions are kept, labelled, in the last section.

---

## How to read the numbers

Each seed is a synthetic subject: every effect is computed *within* a session, then
one-sample t-tested across seeds. Report four things together, never the mean alone:

| | |
|---|---|
| **mean ± SEM** | across seeds, not across trials |
| **p** | one-sample t-test against zero |
| **seeds with the predicted sign** | an effect carried by 18/20 seeds is a different animal from one carried by 11/20 with two outliers doing the work |
| **verdict** | match (significant, right direction), WRONG (significant, wrong direction), or null |

Units: accuracy effects are proportions; RT effects are in timesteps, of a 10-step trial
with 9 usable response steps; a trial that never crosses threshold sits at RT = 10.
Control-state effects are in units of the raw gate: **selectivity** (`focus`, centre minus
mean flanker weight — where the gate points) and **gain** (mean weight over all five slots
— how hard it gates).

Always label which measure and which trials — accuracy or RT? congruent or incongruent?
Ambiguity there has caused real confusion in this project.

---

## The change that matters: the raw gate

Matt Nassar's suggestion. Under a softmax the five gate weights must sum to one, so the
latent can only move attention *between* slots; gain is pinned at 1/5 on every trial. With
the softmax removed, the latent also sets the gate's overall magnitude, which speeds or
slows the network's evidence accumulation globally. The two axes are close to
uncorrelated and price behaviour differently (see `flanker_task.md`, "The gate has two
axes"): more selectivity is faster *and* more accurate; more gain is faster but less
accurate.

This is what turned the post-error signatures around. An error lowers both selectivity and
gain on the next trial. Lower gain makes the response slower but more accurate; lower
selectivity makes it slower and less accurate. On RT the two add; on accuracy they pull
against each other and the gain term wins, so the model is slower *and* more accurate after
an error — which a one-axis gate cannot do, since on the selectivity axis alone slower and
less accurate go together.

---

## What holds

**The behavioural fingerprint.** The congruency effect on accuracy and on RT is carried by
every seed at every rung of both ladders.

**The accuracy distance effects.** Near incongruent flankers cost accuracy relative to far
ones, and near congruent flankers help, at the default and across the delay ladder. Both
fade at the lowest noise, where accuracy approaches ceiling.

**Post-error adaptation — all three signatures together.** PES, PIA and PERI match at
delay 0 and delay 1 (and at noise 1.9), the first time one setting has given all three.
The inherited state after an error is lower on both axes; see the mechanism above.

**Fast errors.** Incongruent errors are faster than incongruent correct responses
(decided trials), and the gap grows with target delay: with the flankers on first, an
early flanker-driven commitment *is* the error.

**Conflict adaptation.** The Gratton effect on accuracy survives the response-repetition
control (Mayr, Awh & Laurey 2003) everywhere, and lag-2 congruency predicts almost as
strongly as lag-1 — the control state integrates over a few trials, a model prediction
worth testing in the human data.

**What the delay ladder does.** A later target amplifies every conflict-adaptation measure
and the fast-error effect, and costs overall accuracy and decisiveness (more non-responses).
At delay 4 post-error slowing reverses and PERI is lost, so the long-delay end trades the
post-error signatures for the sequential ones.

## What fails, or is conditional

**The RT distance effect on incongruent trials.** Near incongruent flankers should slow the
response more than far ones. At the default it is null: near flankers cost accuracy but
not time. It appears only at low noise (1.0 and 0.6), and turns negative (near *faster*)
as the delay grows. Untested guess: near flankers produce more fast wrong commitments, which
pull the near-trial mean RT down.

**Post-incongruent slowing on incongruent trials (`pcs_BI`).** Scored as a match only at
delay 4 and noise 1.9; null or wrong elsewhere. Read the split before quoting it —
`group_14_post_conflict.pdf` draws it. In short:

- After a correct incongruent trial the gate is more selective and lower in gain, in every
  condition and every seed.
- On a **congruent** next trial that is a robust slowing (every condition, nearly every
  seed), with lower accuracy — the cost a control adjustment should pay. On congruent
  trials gain has no trade-off — more gain is faster *and* more accurate
  (`group_13_control_axes_cong.pdf`) — so the post-conflict gain drop is a pure cost
  there. One reading, untested: gain amplifies whatever is on screen, which on a
  congruent trial all agrees with the target.
- On an **incongruent** next trial, *correct responses get faster* almost everywhere; the
  selectivity gain outweighs the gain drop. Where the pooled RT contrast does come out
  slower (delay 4, noise 1.9) it is a mixture effect — higher accuracy means fewer fast
  errors in the mean, and more trials fail to respond — not slower correct responses.

Whether that counts as a match depends on whether the human measure is computed on correct
trials only, which is not yet settled.

**Distance effects at low noise.** The congruent distance effect and the accuracy
congruency effect shrink toward zero at noise 0.6 because accuracy is near ceiling there,
and post-error slowing turns wrong-way. The low-noise end is not a usable working point.

---

## Earlier model (retired), for the record

On 5-timestep trials with a softmax gate, PIA and PERI went the wrong way at high stimulus
noise and PES inverted at low noise, so no noise level gave all three. The mechanism was
real for that architecture: the control deficit preceded the error, the error's correction
undershot, and with gain fixed the only lever was selectivity, on which slower and more
accurate cannot coexist. The candidate fixes then considered — a larger `Z_lr`, an
error-gated learning rate, an explicit conflict representation — were motivated by that
failure. The raw gate resolved it without any of them; the error-gated rate remains of
interest on its normative grounds (does it improve overall accuracy), not as a repair.
`flanker_regression.md` §7 keeps the noise series that documented it.
