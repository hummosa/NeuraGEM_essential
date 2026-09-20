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

**Which context is which.** The two contexts are not interchangeable from the network's
point of view: it comes out of training with its weights and its latent sitting in the
context of the last block it saw, and when the gate is held at the middle it falls back on
a context of its own rather than sitting between the two. Every per-context split is
therefore reported **relative to training** — "the last trained context" against "the
other" — read per seed from the final trial of the active phase. Across the six networks
this is context 1 for three of them and context 0 for the other three, so pooling on the
raw label would have averaged two different things.

**What is encoded where.** The paper's representational claim is that the cortex mixes task
variables while the thalamus demixes them. We ask the same question of this model's own
signals, without assuming its answer, and with the one control the comparison needs. Six
sources are tested: the 64 hidden units at the end of the cue period and at the end of the
trial, the latent before the trial, the latent's own update, the error gradient on the
latent, and — the control — the hidden state reduced to its top two principal components.
That last one matters because a 64-dimensional signal will out-decode a 2-dimensional one
on almost anything simply by having more dimensions to do it with, so any difference
between the hidden state and the latent that does not survive against the 2-component
version is a dimensionality effect and is reported as such. Each source is tested against
cue, rule, context, cue conflict, outcome, and the ideal observer's rule and cue
uncertainty, in two ways: cross-validated decoding with a shuffled-label null computed for
every cell, and a drop-one variance decomposition that fits all the variables jointly and
credits each one only with the variance it uniquely explains, leaving what two variables
share and what nothing explains as their own segments.

**Manipulating the switch.** The paper's causal experiments act on a handful of trials just
after a reversal and then stop: the ACC→MD terminals are silenced during the feedback of
the first four trials, and the thalamus is driven during the feedback of the first five
trials after a high-conflict reversal. The model's equivalent is a hook on the latent
update, off by default, that applies one change inside a window defined in trials since the
reversal. Three are used. **The latent update is scaled**, to zero (the silencing analogue)
or up by 3 or 10 — the upward version has no counterpart in the paper, which never
stimulated ACC, and is labelled as ours throughout. Scaling to zero stops the latent from
moving but not the gradient from being computed, which is the point: in the paper the error
signal survives the silencing of its output. **Both latent units are driven to 1** at the
first post-reversal feedback and the gradient takes over from there. Under the softmax this
is a reset to the uniform gate, because the softmax is shift-invariant; under the sigmoid
both gates open to 0.73 and it is a genuine gain boost, so it is run on both and each panel
says which. **The latent update is given momentum** for the window. This last one is a
question rather than a control: the model's latent is persistent where the paper's thalamic
switch response is transient, and the paper's cortical error signal builds up over
consecutive errors in a way a memoryless gradient cannot, so momentum is the smallest change
that would let it build up the same way. Every manipulation runs on the forced low- and
high-conflict sessions, so each is compared with its own unperturbed partner within seed and
within trial stream.

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

## The story figure (`story.pdf`)

One figure, six rows of four panels, lettered a–x, with each row also written on its own
(`story_1_behaviour.pdf` … `story_6_manipulations.pdf`) so a row can be reworked without
rebuilding the rest. It is larger than a single panel preset on purpose: it is the whole argument, and
`docs/figure_style.md` allows that for a figure that summarises this much, provided the
reason is stated. Six networks throughout; the seed is the unit, every mean carries its SEM
across seeds and one faint line or dot per seed. A legend appears only where it says
something the caption cannot — elsewhere, light solid is a low-conflict start and dark
dashed a high-conflict one, in every panel that makes the split.

### Row 1 — Behaviour (`story_1_behaviour.pdf`), against Fig 1e–f

**Ran.** Unforced test sessions for a and b; the paired forced-conflict triplet for c and d.

**Plotted.** *(a)* Steady-state accuracy against cue conflict, one curve per context, with
the ideal observer's cue accuracy as the ceiling and the RNN baseline in green.
*(b)* Trials to switch against the conflict of the block's first five trials, reversals
split into three equal-count bins per seed. *(c)* Trials to switch on the forced low/high
pairs, under the decided-only and latent criteria, with the observer. *(d)* Reversal-aligned
accuracy.

**See.** Accuracy falls from 0.98 to 0.68 as the cue becomes ambiguous, tracking the
observer's ceiling (1.00 → 0.70) just below it. **The two contexts are close but not
identical**: the context the network saw last in training is better at every conflict level,
by 0.042 on average — small, and in the same direction in every seed, so it is a real
residue of training rather than noise, and it is why every per-context split here is
relative to training rather than to the raw label. Panel b is **the paper's Fig 1f on
reversals nobody controlled**: the more ambiguous the first five trials happened to be, the
longer the network stays on the old rule — 4.6, 5.3 and 6.1 trials across the three bins,
with every one of the six networks slower in the most ambiguous bin than in the least. The
ideal observer pays the same cost on the same trials (2.6, 2.9, 3.5). The raw behavioural
criterion does *not* show it (4.1, 4.6, 4.3), which is the hedging problem: the sign of a
near-zero output is a coin flip that can satisfy a "first correct trial" rule by luck, and
it is why the decided-only criterion is the one plotted.

### Row 2 — What is encoded where (`story_2_encoding.pdf`), against Fig 2k–l and 4b

**Ran.** Unforced sessions, all trials of the test phase. Decoders are 5-fold
cross-validated, each with its own shuffled-label null; the variance decomposition fits all
seven variables jointly and credits each only with what it uniquely explains.

**Plotted.** *(e)* Decoding accuracy for cue, rule, context and conflict from the hidden
state, from the hidden state cut to two principal components (hollow), from the latent Z,
and from the error gradient on Z. *(f)* Cue and rule decoding from the hidden state per
timestep, steady state against the first five trials after a reversal. *(g)* The size of the
gradient on Z against cue conflict, correct trials against errors. *(h)* The variance
decomposition, one stacked bar per signal.

**See.** **The hidden state mixes and the latent does not.** The hidden state decodes all
four variables well above its null (cue +0.42, rule +0.33, context +0.40, conflict +0.35;
6/6 networks each), and a hidden unit is tuned to 4.5 of the seven variables on average.
The latent decodes **context and nothing else** — +0.41 above null for context in every
network, and flat on cue, rule, conflict and outcome. In panel h that is one tall segment
against several: of Z's variance, context uniquely explains 0.17 and no other variable
reaches 0.01, while the hidden state's is split across cue (0.21), rule (0.08) and context
(0.06). This is the paper's cortex-mixed / thalamus-demixed dissociation, arrived at
without being asked for.

**The dimension-matched control.** A 64-unit signal will out-decode a 2-unit one by having
more dimensions, so the hidden state reduced to its own top two principal components is
plotted beside it (hollow). Those two components carry cue and rule as well as the full
hidden state does, and carry context **less than half as strongly** (+0.18 against +0.40).
So context is present in the hidden state but not in the directions that dominate its
variance — which is what arriving through the gate looks like, rather than being computed.

**The baseline is the other half of that argument.** The RNN, which has the same
architecture with the latent update switched off, decodes cue +0.29 from its hidden state
(10/10 networks) but rule +0.01 and context +0.04, and its units are tuned to 1.45 variables
each against NeuraGEM's 4.5. **The mixing is therefore not something an LSTM does on this
task**; it appears only when a latent is gating the state.

One column has to be read carefully: the decoders are linear, and the two uncertainties
are magnitudes, which a signed two-unit signal cannot produce under a linear map. Z scoring
near zero on rule uncertainty is therefore a fact about the read-out and not about Z — the
same quantity measured directly, as the latent's distance from the midpoint of its context
axis, does rise after a reversal and stays high longer when the start was ambiguous. The
substantive comparison is the four sign-carrying variables.

The gradient is the error signal. It carries the outcome (+0.16) and which context was
wrong (+0.14), both 6/6, and its size falls with conflict on error trials while staying flat
and roughly ten times smaller on correct ones (g) — conflict-weighting with nothing in the
model that is told about conflict. It carries the cue and the rule only weakly (+0.07,
+0.06, and not consistently across networks); along the context axis the gradient's sign is
fixed by which context was wrong, so there is little room in it for the cue. Panel f is the
reversal seen from the hidden state: the cue code is untouched in the first five trials
after a reversal while the rule code drops to chance — the network still hears the cue and
has lost what to do with it.

### Row 3 — Holding the gate still (`story_3_gate.pdf`)

**Ran.** Weights frozen *and* the latent update off, Z clamped at a fixed (gain, contrast)
for a whole 1000-trial session: 4 gains × 5 contrasts × 6 networks, 120 sessions, sigmoid
gate (the softmax has no gain direction to clamp). Accuracy is on the context the gate
selects, read off behaviour; hidden-state measures pool both halves of the session, which
see identical inputs.

**Plotted.** *(i)* Accuracy, *(j)* RT, *(k)* integration index and *(l)* cue velocity
against the clamped contrast, one line per clamped gain.

**See.** **Gain and contrast do different jobs.** Opening the gate speeds the hidden state's
integration of the cue — cue velocity 0.126 → 0.188 → 0.231 → 0.257 from gain −1 to +2,
monotonically — and raises the integration index (1.09 → 1.61). Moving the contrast does not
touch the cue velocity at all: 0.200 to 0.201 across the whole ladder. What the contrast
sets is which context is applied and how decisive the answer is (accuracy 0.61 → 0.83,
undecided 0.79 → 0.25). Accuracy is non-monotonic in gain and peaks near 0 to +1: too little
and the network is undecided on nearly every trial, too much and both units saturate so the
contrast between them stops meaning anything. **RT is the one measure both move** (24.5 →
20.6 across gain, 23.9 → 20.2 across contrast), which is what it should do — a decision needs
both a rule to apply and enough gain to apply it with.

### Row 4 — Around a reversal (`story_4_reversal.pdf`), against Fig 3c

**Ran.** The forced pair for m, unforced sessions for n–p. The integration index and cue
velocity are population measures over a set of trials, so each point pools the trials at
that offset over every reversal of the session.

**Plotted.** *(m)* Undecided rate, *(n)* RT, *(o)* integration index and *(p)* cue velocity,
from 5 trials before a reversal to 15 after.

**See.** The behavioural signature of the transition is clear. The undecided rate rises
from 0.12 at baseline to 0.26 after a low-conflict start and 0.34 after a high-conflict one,
peaking at trial 3 and back to baseline by trial 6 or 7 — the network passes through a
period of not committing, and dwells there longer when the evidence for switching was
ambiguous. RT is **non-monotonic**: trial 1 is *fast* (19.4 of a 25-step trial), because the
network is confidently applying the old rule, then RT peaks at trial 3 (20.5) where the
latent is most uncertain, and recovers by trial 6 (19.5). **The paper's population signature is at best weakly present.** The integration
index is lower in the first five trials than in the steady state, but only by 0.079 ± 0.025
(5/6 networks), and the cue velocity does not move at all (+0.001 ± 0.011, 4/6) where the
paper has it rise. Row 3 says why that is coherent rather than contradictory: cue velocity
is set by the *gain*, and under the softmax the gain cannot move, so the one measure the
paper uses to define the exploratory regime is the one measure this model's latent has no
way to change. The RNN baseline sits at the trial end in n because it is undecided on
99.9 % of trials, so its RT is not a response time.

### Row 5 — The three latent signals (`story_5_latent.pdf`)

**Ran.** Unforced sessions; t uses the sigmoid-at-test condition, the only one where the
gain exists as an axis, and is labelled accordingly. It holds the context less well than the
softmax (0.70 against 0.88 steady-state accuracy) — the point of the panel is the axis, not
the performance.

**Plotted.** All aligned on the reversal. *(q)* The latent's position on the context axis
(+1 = the true context's prototype) with the ideal observer's belief. *(r)* The size of the
trial's own latent update along that axis. *(s)* The size of the raw error gradient.
*(t)* The gain, the mean of the two latent units.

**See.** **The state is persistent and its update is transient**, which is this model's
answer to a tension in the paper: there, the thalamic context signal is a brief switch
response, while here the latent is the one thing that crosses trials and so cannot be brief.
Panel q shows the state — sitting at +1.0, thrown to −0.99 on the first post-reversal trial,
and climbing back over four to six trials, tracking the observer a little more slowly.
Panels r and s show what moves it, and both are sharp: the update is 0.10 of the distance
between the prototypes in the steady state, rises to 0.56 at trial 2 and is back near
baseline by trial 8; the gradient does the same, 7.5e-6 → 4.4e-5 at trial 2 → 1.7e-5 by
trial 6. **So the transient and the persistent signal are both here, as the derivative and
the integral of one another** — not a correspondence the paper draws, and available only
because the update and the state are separately measurable in a model. Panel t is the cost
of the gain axis when it exists: after a reversal the burst of errors pushes the gain down
(+0.03 before, −0.33 by trial 5) and the correct trials that follow do not restore it. Under
the softmax this panel would be a flat line at zero, and that is the reason the softmax is
used.

### Row 6 — Manipulating the switch (`story_6_manipulations.pdf`), against Fig 4h and 5d

**Ran.** Each manipulation acts on the first few trials after a reversal and then stops, as
the paper's optogenetics does, and each runs on the forced low- and high-conflict sessions
so it is compared with its own unperturbed partner within network and within trial stream.
15 conditions × 6 networks, 90 sessions.

**Plotted.** *(u)* Extra trials to switch against that same session unperturbed, one
low/high pair per manipulation; positive is slower. *(v)* What the latent itself does, on
the low-conflict reversals: its position on the context axis under no manipulation, under
the update being switched off, and under being driven to (1, 1). *(w)* The size of the
latent's update and *(x)* of the raw gradient, with momentum at 0.5 and 0.9 against none.

**See.** **Switching off the latent update for four trials nearly doubles the switch
latency** — 9.07 trials against 4.61, +4.46 ± 0.19, in 6 of 6 networks — which is the
paper's ACC→MD silencing result (Fig 4h) in a model where nothing else was touched. Panel v
shows why: the latent simply sits on the old context for the four silenced trials and only
then begins to move. Driving the update the other way gives the converse and does so
**monotonically**: ×3 and ×10 reach 3.68 and 2.80 trials, so across 0×, 1×, 3×, 10× the
latency runs 9.07, 4.61, 3.68, 2.80. Steady-state accuracy is 0.88 in every arm, so none of
this is a manipulation breaking the network. The graded version has no counterpart in the
paper, which never stimulated ACC, and is ours.

**Driving the latent to (1, 1) means different things under the two gates, and the sizes
say so.** Under the softmax it is a reset to the uniform gate, because the softmax is
shift-invariant, and it is worth −1.37 trials. Under the sigmoid both units open and it is a
real gain boost, worth −7.85 — though that arm's control is genuinely slow (13.90 trials at
0.69 steady accuracy), so the raw pair belongs beside the difference.

**Momentum does not make the error signal ramp.** At 0.5 the peak update is unchanged and
switching is marginally faster (4.86 against 5.35); at 0.9 the peak update is *lower* and
switching is *slower* (6.17). The raw gradient (x) is unchanged by construction — momentum
changes the step, not the gradient — and its peak falls if anything. The reason is
structural: **this error signal is self-limiting.** It exists because the latent is in the
wrong place and it drives the latent to the right place, so anything that makes the latent
move faster removes the very errors that would have made the signal grow. An accumulation
over consecutive errors, of the kind the paper's cortical signal shows, needs something that
integrates *without* acting on what it is integrating.

---

## Figure captions


The supplementary figures, in `exports/hier_switch/group/figures/`. These are the panels
the story figure does not carry, kept because each says something it leaves out: all three
switch criteria rather than two, the latent's uncertainty, the gain leak measured per trial,
the unit classes, the softmax clamp ladder and the single-gain cut.

**Two of them have been folded into the story figure.** `switching.pdf` is gone entirely —
its reversal-aligned accuracy is story panel d and its Z-side curve is story panel q — and
`behaviour.pdf` has lost its reversal-aligned RT panel, which is story panel n. Neither
*builder* was deleted; the figures below describe what is still written. An earlier
`z_updates.pdf` (the per-trial update decomposed cell by cell) was retired before that as
unreadable; the numbers behind it are still computed and live in
`docs/hier_switch_handoff.md` §5b, and its builders (`spec_z_update`, `spec_tipped`,
`spec_normative`) remain in `hier_switch_figures.py` if a panel is ever wanted again.

### Fig. 1 — Behaviour (`behaviour.pdf`)

*Its third panel, reversal-aligned RT, is now story panel n and is no longer written here.*

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

### Fig. 2 — What a low- or high-conflict start does to a reversal (`switching.pdf`, retired)

*Both panels are now in the story figure (d and q); this figure is no longer written. The
caption is kept because the numbers in it are still the ones to quote.*

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
