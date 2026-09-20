# hier_switch — methods narrative and figure captions

Written for a reader who has not seen the code: what was run, what is on each axis, and
what it shows. The internal labels (P1…P7, A/B/C) are dropped here; they survive in
`docs/hier_switch_handoff.md` §5b and in `exports/hier_switch/group/group.json`, which
carries the per-seed numbers behind every sentence below. Definitions of every derived
quantity are in `docs/hier_switch_analyses.md`.

Figures: `exports/hier_switch/group/figures/*.pdf`, rebuilt by
`.venv/bin/python hier_switch/hier_switch_figures.py`.

---

## Methods

**Task.** On each trial the network hears 16 noisy pulses — 9 informative (high-pass or
low-pass) and 7 white noise — then sees a visual and an auditory target on opposite sides
and reports the side of the attended one. The dominant pulse type is the *cue*; the cue and
the current *context* together fix which modality to attend, so the correct side is the
three-way product vis × cue × context. **Cue conflict** is the ratio of non-dominant to
dominant informative pulses (9:0, 8:1, 7:2, 6:3, 5:4 → 0, .125, .29, .5, .8), drawn
uniformly per trial; Gaussian noise (σ = 0.5) on the pulse channels makes it a real sensory
uncertainty rather than a label. The context reverses without warning every 30–60 trials.
Because the answer is a three-way product, a reversal flips the correct response on every
trial, and nothing in the input announces it.

**Networks.** One architecture throughout: a 64-unit LSTM whose hidden state is gated
multiplicatively by a 2-unit latent Z (softmax, temperature 0.5) through a fixed sparse
mask. One trial is one 25-step sequence and the hidden state resets between trials, so **Z
is the only thing that crosses trials**. Two models differ in what may change:

- **NeuraGEM (NG).** Weights by Adam on the response error; **Z by plain SGD on the same
  error, one step per trial** (learning rate 1e4, L2 decay 3e-6, no momentum), so
  `z ← z − lr·(dL/dZ + decay·z)`. There is no context input, no context read-out and no
  context term in the loss: the only supervision is the correct side, which in a two-choice
  task carries what binary reward would.
- **RNN baseline.** Identical, with the latent update switched off, so Z stays at the
  uniform gate for good and the weights are the only adaptive variable; they stay plastic at
  test, since that is its only route to adaptation.

Training was fixed in phase 1: 2 × 2000 passive trials (weights only, Z held) to learn the
task at all, then 5000 trials on 200-trial blocks with the latent update on. 6 of 10 seeds
discover the two contexts; **those six are the group** for every NG number reported here.
All ten RNN seeds are used.

**Test sessions.** Weights are frozen, Z is restarted at the uniform gate and inferred
trial by trial, and the network runs 2000 fresh trials on the paper's 30–60-trial blocks
(≈ 45 reversals) drawn from a separate RNG stream. The first block is discarded as a
start-up transient. Each seed was run three times as a **paired triplet**: with the first
five trials of every block forced to low conflict (7:2), forced to high conflict (6:3), or
left as drawn. The forced level is drawn as usual and then overwritten, so the random
stream is untouched and the three sessions are identical trial by trial apart from those
five trials — the low/high comparison is therefore within seed *and* within trial stream.

**Two variants of the gate, for two different questions.**

- **Sigmoid at test.** The softmax has one degree of freedom: *which* of the two units is
  open. Its sum-to-one constraint removes any overall **gain**. To ask what the gain
  direction does, the same softmax-trained weights were tested with the softmax replaced by
  a per-unit sigmoid, which leaves both directions live (Z_lr 3e4; decay ladder
  Z_lr × decay = 0.09 / 0.3 / 0.9). This costs accuracy — 0.70 steady-state against 0.88 —
  and every figure using it says so; the point is the axis, not the performance.
- **Clamped Z.** Weights frozen *and* the latent update off, with Z held at a chosen raw
  value (m + d, m − d) for a whole session: **d is the contrast** (which context the gate
  selects) and **m the gain** (how open both units are). 5 contrast levels under the
  softmax, 4 gains × 5 contrasts under the sigmoid, every cell a full 1000-trial session,
  6 seeds.

  **The trial stream is left exactly as it is, so the context still reverses every 30–60
  trials and a fixed gate is "right" for about half the session and "wrong" for the other
  half.** That is deliberate, and it is not a confound, because *the context label never
  reaches the network*: the inputs are the pulses and the two targets, and the context only
  determines which side is scored correct. With the weights frozen and the latent update
  off, a clamped session is therefore literally one set of 1000 stimuli scored against two
  different answer keys in alternating runs. Consequences:
  - **Anything defined against the correct answer is always reported split** — both contexts'
    raw accuracies, plus accuracy on the context the gate selects and on the other one. For a
    committed softmax gate these are mirror images (0.90 / 0.10), which is itself the check
    that the gate is acting as the rule rather than as a general performance knob.
  - **Anything measured from the hidden state is pooled over both halves**, because the two
    halves are the same experiment. Pooling is also the *safer* estimate here: in a clamped
    session the network has no context signal, so a regression of the hidden state on
    [cue, rule] cannot separate the two — the fitted axes are 32° ± 10 apart with a committed
    gate and 39° ± 14 with a uniform one, against 83.4° ± 0.7 when Z is inferred. Splitting by
    context then hands each half ± that leak (cue velocity 0.194 vs 0.217 between the two
    halves of one session, with identical inputs), while pooling cancels it. The integration
    index is unchanged either way (identical to 0.01).
  - **Which context a gate selects is read off behaviour**, as the context it is more
    accurate in, with both raw accuracies printed. Geometry does not work: the sigmoid's gain
    direction takes a clamp off the context axis, and the raw middle gate is not neutral —
    each network has a default context it falls back on.
  - **Rule decoding is at chance in every clamped cell, by construction**: "rule" is
    cue × true context, and a clamped network has no access to the second factor. That is a
    result, not a gap — the rule code in the unclamped model is Z's doing.

**What is recorded.** Every test session saves, per trial: the response output at all 25
timesteps, the 16 noisy pulse frames the network saw, the 64-unit hidden state at all 25
timesteps, Z before and after the trial's own update, and the labels. Because the latent
update is a single plain-SGD step, the trial's error gradient is recovered exactly from the
saved Z (`g = −Δz/lr − decay·z_in`; it matches the logged gradient to 7e-12).

**Derived measures.**

- **Context axis.** The two contexts' mean Z over steady-state trials (≥ 11 trials into a
  block) define two *prototypes*; the axis joins them, the midpoint separates them.
  **Z evidence** is where a trial's incoming Z sits on that axis, signed toward the true
  context and scaled so ±1 are the prototypes. A trial is **aligned** if Z was on the true
  context's side and **stale** if it was on the other — "stale" is almost always the first
  few trials after a reversal, i.e. the trials where the network is still acting on the old
  rule. **Z uncertainty** is 1 − |Z evidence|: 0 on a prototype, 1 at the midpoint.
- **The trial's update.** Split Δz into the part weight decay would have produced anyway
  and the part this trial's error produced, then project the error part onto the context
  axis: **s** is the fraction of the distance between the two prototypes that one trial
  moved Z, signed positive toward the context Z was *not* holding. **Tipped** means the
  update carried Z across the midpoint. **Δgain** is the change in the mean of the two
  units (identically zero under the softmax; measured max 1.9e-7).
- **Response speed.** The decision variable is the response channel over the response
  window; **RT** is its threshold crossing (|output| > 0.5) interpolated between timesteps,
  and a trial that never crosses is **undecided** and scored at the trial end (25) rather
  than dropped. Undecided rate is always reported next to accuracy, because with Z at the
  middle the output sits near zero and the sign of a near-zero output is a coin flip that
  *looks* like chance performance.
- **Switching.** Three criteria per reversal: the **behavioural** one (the paper's: the
  first correct trial with another correct within the next two), the same on **decided**
  trials only, and the **latent** one (the first trial Z holds the true context).
- **Conflict-weighted error.** The paper's ACC quantity, `error / (1 − P(cue read
  correctly))` averaged over the previous five trials, with P estimated from the session's
  own steady-state accuracy at each conflict level. It averages 1.10 here, as it does in the
  paper.
- **Ideal observer.** A numpy Bayesian observer on the same trials: a frame-wise likelihood
  over the noisy pulses gives the cue posterior (it reads the cue correctly on 93 % of
  trials at σ = 0.5), and a hazard-1/45 filter over the feedback gives the context belief.
  It provides the accuracy ceiling, the normative size of every switching effect, and the
  normative log-odds update each trial deserves.

**Figure conventions.** Hue is the model (NeuraGEM, RNN, ideal observer;
`plot_style.get_model_color`). **Outcome gets a hue and a marker of its own** — correct is
teal circles on a solid line, error crimson triangles on a dashed one
(`plot_style.outcome_line`) — because a dashed line alone is unreadable in a paper-sized
legend. The early-reversal conflict is a **shade** of the model's hue: lighter = low, full =
high, dashed for high. Every mean carries its SEM across seeds and one dot or faint line per
seed. Sizes come from `plot_style.FigSize` presets only.

**Statistics.** The seed is the unit of analysis. Every effect is computed within seed and
reported as the mean across the six (ten for the RNN) with its standard error, one dot per
seed on the panel, and the count of seeds carrying the predicted sign. No trial-level
significance tests are used.

---

## Figure captions

Nine figures, in `exports/hier_switch/group/figures/`. An earlier `z_updates.pdf` (the
per-trial update decomposed cell by cell) was retired as unreadable; the numbers behind it
are still computed and live in `docs/hier_switch_handoff.md` §5b, and the builders
(`spec_z_update`, `spec_tipped`, `spec_normative`) are still in `hier_switch_figures.py` if
a panel is ever wanted again.

### Fig. 1 — Behaviour (`behaviour.pdf`)

**Ran.** The unforced test sessions: 6 NG seeds and 10 RNN seeds, frozen weights, 2000
trials each, Z inferred (NG) or fixed (RNN).

**Plotted.** *(a)* Accuracy on steady-state trials (≥ 11 into a block) against cue conflict,
with the ideal observer's probability of reading the cue correctly as the ceiling. *(b)* RT
against conflict. *(c)* RT against trials since the reversal. Lines are means over seeds ±
SEM, thin lines individual seeds.

**See.** NG falls from 0.98 to 0.66 as conflict rises, tracking the observer's ceiling
(1.00 → 0.70) a little below it — the errors are mostly the task's sensory noise, not the
model. RT rises monotonically with conflict (18.8 → 20.5 of a 25-step trial) and the
undecided rate rises with it (0.05 → 0.28): ambiguous cues are answered later and more often
not at all. Reversal-aligned RT is **non-monotonic**: trial 1 is *fast* (19.4) because the
network is confidently applying the old rule, RT peaks two to three trials later (20.5)
where Z is most uncertain, then recovers by trial 6. The RNN sits at 0.50 at every conflict
level.

### Fig. 2 — What a low- or high-conflict start does to a reversal (`switching.pdf`)

**Ran.** The paired forced-conflict triplet per seed. The only difference between the two
sessions plotted is the conflict of the first five trials of each block; everything else,
including the random stream, is identical.

**Plotted.** *(a)* Accuracy and *(b)* the fraction of trials on which Z holds the true
context, from 5 trials before a reversal to 15 after. Light solid = low-conflict start, dark
dashed = high.

**See.** After a low-conflict start, Z is back on the true context by trial 4 (0.49, then
0.75, 0.88); after a high-conflict start the same curve lags by about one trial (0.38, 0.61,
0.73), and accuracy follows it. Note the first post-reversal trial: accuracy 0.04 (low)
against 0.15 (high). High conflict *raises* raw early accuracy, simply because an ambiguous
cue makes the perseverative response less reliable — which is why switch latency, not early
accuracy, is the measure in Fig. 3.

### Fig. 3 — How much slower an ambiguous start makes it (`switch_latency.pdf`)

**Ran.** The same paired sessions. Three ways of asking "when did it switch": the paper's
**behavioural** criterion (first correct trial with another correct within the next two), the
same on **decided** trials only (|decision| crossed threshold, so a coin-flip on a near-zero
output cannot count), and the **latent** one (first trial Z is back on the true context). The
**ideal observer** is the same criterion applied to its own choices on the same trials.

**Plotted.** *(a)* Trials to switch, one low/high pair per criterion; shade is the early
conflict. *(b)* The same thing as one number: the extra trials a high-conflict start costs,
paired within seed, dots = seeds, zero line = no cost.

**See.** Every criterion pays a cost, and it is larger the less the criterion is
contaminated by hedging: +0.21 ± 0.13 trials behaviourally (4/6 seeds), +0.61 ± 0.08 on the
latent criterion (6/6), +0.66 ± 0.09 on decided trials (6/6). **An ambiguous start makes the
network slower to abandon the old rule** — the paper's central behavioural result,
reproduced with nothing in the model that is told about conflict. The ideal observer pays
+1.07 ± 0.06, so the model shows about 60 % of the normative effect.

### Fig. 4 — The latent, against the ideal observer (`latent.pdf`)

**Ran.** Unforced sessions for (a) and (c); the forced pair for (b).

**Plotted.** *(a)* Z on the context axis (+1 = the true context's prototype) and the
observer's posterior belief in the true context, aligned on the reversal. *(b)* Z uncertainty
(1 = the midpoint of the axis) after low- and high-conflict starts. *(c)* The size of the
trial's error gradient on Z, |dL/dZ| along the context axis, against cue conflict, for
correct trials (teal) and errors (crimson).

**See.** Z drops to the wrong prototype on the first trial after a reversal and recovers over
4–6 trials, tracking the observer's belief at r = 0.63 with no lag, slightly slower than the
observer. Uncertainty spikes at trial 3 in every seed; after a high-conflict start the peak
is no higher (0.37 vs 0.40) but it is **wider** (3.8 vs 2.7 trials at half height, 5/6
seeds) — the model dwells in the uncertain state longer when the evidence for switching was
ambiguous. Panel (c) is the ACC analogue: on error trials |dL/dZ| falls with conflict
(3.3e-5 → 2.0e-5), on correct trials it is flat and roughly ten times smaller. **The gradient
is conflict-weighted by construction** — the ∂y/∂Z factor shrinks when the cue is ambiguous —
and its five-trial running mean tracks the paper's hand-built conflict-weighted error at
r = 0.57.

### Fig. 5 — The gain axis, and why the softmax removes it (`gain.pdf`)

**Ran.** The gate has two directions: the **contrast** between its two units, which selects
the context, and their **mean — the gain**, which is how far open the gate is overall. Under
the softmax the two units sum to one, so the gain cannot move at all. To see what that
direction would do, the same frozen weights were tested with the softmax replaced by a
per-unit sigmoid (Z_lr 3e4) at three weight-decay strengths (Z_lr × Z_decay = 0.09, 0.3,
0.9), latent update on. Each trial's update is decomposed exactly as in the latent analyses,
but projected onto the mean of the two units instead of their difference.

**Plotted.** *(a)* Δ gain per trial against cue conflict, correct vs error, for the sigmoid
(solid/dashed with markers) and for the softmax (dotted, as the flat reference). *(b)* The
leak as one number per setting: how much further down an error pushes the gain than a correct
trial does, at the most ambiguous cue, for each decay strength and for the softmax. *(c)*
What it costs: steady-state accuracy of the same weights under each gate.

**See.** **An error pushes the gain down and a correct trial pushes it back up, and the
down-steps are bigger** — −0.15 to −0.24 per error at every conflict level above zero
(−0.02 at zero conflict, where such errors barely happen) against +0.02 to +0.06 per correct
trial. The asymmetry is the "gain leak": after every reversal the burst of errors shrinks the
gate and the correct trials that follow do not fully restore it. It grows with the decay
strength (−0.21 ± 0.08, −0.34 ± 0.11, −0.38 ± 0.10; 5/6, 5/6, 6/6 seeds) and it costs
accuracy: 0.88 ± 0.01 steady-state under the softmax against 0.70 ± 0.02, 0.70 ± 0.02 and
0.69 ± 0.03 on the sigmoid ladder, with the *same weights*. Under the softmax the same
quantity is identically zero (largest |Δ gain| 1.9e-7 over every cell and seed). **The
sum-to-one constraint is what keeps the latent a context code rather than a volume knob** —
this panel is the reason the model's latent works at all.

### Fig. 6 — The hidden state (`hidden.pdf`)

**Ran.** Unforced sessions, hidden states recorded at all 25 timesteps. Decoders are 5-fold
cross-validated logistic regressions, fitted separately at each timestep, on steady-state
trials and on the first five trials after a reversal. Cue and rule are perfectly correlated
inside a block, so every fit pools trials from both contexts.

**Plotted.** *(a)* Cue and *(b)* rule decoding accuracy against timestep (grey = the pulse
period), steady state vs the first five trials after a reversal. *(c)* The paper's
integration index — activity late in the cue period over activity early in it, along the
sustained cue-and-rule dimension — by condition. *(d)* The fraction of tuned units in the
paper's three classes, NG against the RNN.

**See.** In the steady state the cue and the rule are both read out of the hidden state by
the end of the pulses (0.92 and 0.91). In the first five trials after a reversal **the cue
code is untouched (0.90) while the rule code is at chance (0.52)** — the network still hears
the cue perfectly well and has simply lost what to do with it, which is the paper's
"exploration" signature and, here, a direct consequence of Z sitting near the midpoint. Class
composition differs from PFC: NG's tuned units are 64 % sustained cue-like and 33 %
rule-like, with almost no transient class (3 %), whereas the RNN has the transient class
(41 %) and essentially no rule units (1 %). The integration index varies little across
conditions (1.80 steady vs 1.77 early) — the paper's regime shift is at best weakly present.

### Fig. 7 — Holding the gate still: contrast (`clamp_softmax.pdf`)

**Ran.** Weights frozen *and* the latent update off, Z clamped at a fixed contrast for a
whole 1000-trial session, 5 levels × 6 seeds. The context still reverses every 30–60 trials,
so each session tests the same gate against both contexts; accuracy is reported for the
context the gate selects (read off behaviour) and for the other one, while the hidden-state
measures pool both halves, which see identical inputs (Methods).

**Plotted.** Integration index, cue velocity (the fastest rise along the cue axis in the
first half of the pulse period) and accuracy on the selected context, against the clamped
contrast.

**See.** At any committed gate the network is normal on the context that gate selects
(0.89–0.91) and inverted on the other (0.10) — the gate *is* the rule. At the uniform gate
the output collapses: |decision| 0.18, undecided on 85 % of trials, RT 24.3 of a 25-step
trial, and what sign remains follows a seed-specific default context rather than the block's.
The integration index falls with it (1.58 → 1.33) while **the cue velocity does not move at
all (0.188–0.193 across the whole ladder)**. The gate scales the rule-driven part of
integration and the decision; the cue is read at the same speed whatever the gate does.

### Fig. 8 — Holding the gate still: the full gain × contrast grid (`clamp_sigmoid.pdf`)

**Ran.** The same clamp experiment with the sigmoid gate, where both directions exist: 4 gain
levels × 5 contrast levels × 6 seeds, 120 sessions of 1000 trials.

**Plotted.** Integration index, cue velocity and accuracy on the selected context against the
clamped contrast, one line per clamped gain.

**See.** A clean double dissociation. **Gain sets how fast the hidden state integrates the
cue**: cue velocity rises 0.126 → 0.257 from gain −1 to +2, monotonically in 6 of 6 seeds,
and the integration index rises with it (1.09 → 1.61). **Contrast sets which context is
applied and how decisive the answer is**: accuracy on the selected context rises 0.62 → 0.83
and the undecided rate falls 0.79 → 0.25 across the contrast ladder, while cue velocity stays
flat to three decimals (0.200–0.201) at every level.

### Fig. 9 — The same result in one line (`clamp_sigmoid_gain.pdf`)

**Ran.** One cut through Fig. 8: the contrast held at +1 (a committed gate) and only the gain
varied, 4 levels × 6 seeds.

**Plotted.** Cue velocity, integration index and accuracy on the selected context against the
clamped gain; dots are seeds.

**See.** Opening the gate speeds the cue up and nothing else does: cue velocity 0.126 ± 0.007
→ 0.188 → 0.232 → 0.258 ± 0.019 from gain −1 to +2, and the integration index 1.06 → 1.44 →
1.56 → 1.51. Accuracy is **non-monotonic** and peaks at gain 0 (0.89 ± 0.02, against 0.61 at
−1 and 0.72 at +2): too little gain and the network is undecided on every trial (undecided
rate 1.00 at gain −1), too much and both units saturate so the contrast between them stops
meaning anything. "Turn the gain up" and "commit to a context" are separate controls, and
only the first one changes integration speed.
