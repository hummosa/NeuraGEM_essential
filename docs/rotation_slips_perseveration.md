# Perseveration & Context Slips on the Rotating-Targets Task

`rotation_slips_perseveration_config.py` · `_sweep.py` · `_analysis.py`

The hypothesis going in: a model that fails to form an appropriate latent state representation —
a plain RNN, `no_of_steps_in_latent_space=0` — **perseverates** on the previous context after an
unsignalled switch, **slips** back to it under observation noise once it has adapted, and its
implicit decision boundary drifts as weights keep training on changing blocks.

The first two hold. The third does not, on this evidence: the RNN's belief converges
monotonically through each block and never settles anywhere biased, so there is no stable
boundary to have moved. What replaces it is simpler — the RNN's context updating is roughly an
order of magnitude slower, so it spends most of every block in transit, and a belief in transit
sits close enough to the midpoint for noise to push it across. See
[What F5 settled, and what it ruled out](#what-f5-settled-and-what-it-ruled-out).

This is the [mean-prediction](task_mean_prediction.md) perseveration experiment ported to the
[rotating-targets task](task_rotating_targets.md), where context is a continuous rotation angle
rather than a binary side.

---

## The belief head

Same augmented-input trick `MeanPredictionConfig` uses, via
`RotatingTargetsConfig.enable_context_output`. The observation vector gains dims carrying the
rotation; `input_feed_mask` zeroes them before the forward pass so the model cannot read them,
and `output_loss_mask` re-opens them so the model is trained to *emit* them.

```
              [ color_onehot(n_colors) | ctx_cos, ctx_sin | attack_x, attack_y ]
index                 0 .. 4                 5, 6                7, 8
input_feed_mask       1 1 1 1 1               0    0              1    1
output_loss_mask      0 0 0 0 0               w    w              1    1
```

`input_size = output_size = n_colors + 2 + C`. Note the xy loss stays on — unlike mean
prediction, where the belief *is* the task, here the model must still do the rotating-targets
task **and** report its belief.

> **Context dims sit before `x, y`, not at the end.** Every rotating-targets analysis reads the
> attack as `[-2:]` (`rotating_targets_analysis._analyze_adaptation`, `analyze_rotation_sweep`,
> `run_2D_predicitve_task.plot_arena_trials`) and the cue frame as `ii[t, :nc].sum() > 0.5`.
> This layout leaves both correct; appending at the end would silently redirect all of them at
> the context dims.

### Encodings

| `encoding` | `C` | Target | Decode | Notes |
|---|---|---|---|---|
| `'circular'` | 2 | `target_radius · [cos θ, sin θ]` | `atan2(out[nc+1], out[nc])` | Default. Same scale as the coordinates, so the two loss terms are balanced; no wraparound seam; valid for any angle |
| `'one_hot'` | `len(train_rotations)` | one slot per trained rotation | `argmax` | Maximally separated, but no metric between contexts and no valid target for an unseen angle |

`loss_weight` becomes the `output_loss_mask` entry for the context dims. `_mask_loss` applies the
mask as a plain elementwise multiply, so a float works as a weight with no extra machinery;
`0.0` keeps the dims present but unsupervised.

**Call `enable_context_output` after `train_rotations` is final** — `'one_hot'` sizes itself
from it.

### Reading the belief

`logger.context_ids` is `context_ids[:, 1:]`, the context of the frame being *predicted*, and
`logger.predicted_outputs[t]` is that same frame's prediction. Belief and ground truth therefore
land at the **same index** — no off-by-one:

```python
belief_rad = np.arctan2(oi[:, nc + 1], oi[:, nc])   # reported context
true_rad   = ll                                      # logger.context_ids
correct    = nearest_trained_rotation(belief_rad) == nearest_trained_rotation(true_rad)
```

Two judgements per trial (cue frame and outcome frame). Default scoring uses the **outcome
frame** — the post-evidence belief, one judgement per trial; `AnalysisParams.frames` accepts
`'all' | 'cue' | 'outcome'`.

### Why the head makes the test stronger, not weaker

The belief head supervises the network to encode rotation, which is the very thing an RNN is
hypothesised to be bad at. That is the point: if the RNN *still* perseverates and slips even
when explicitly trained to report context, the failure is in the **dynamics** of context
inference — holding and updating a state — not in whether rotation is representable at all.

Two guards make that argument checkable rather than assumed:

- **No-head control** (`context_output_encoding=None`, run by the pilot;
  `compare_head_vs_no_head` scores both arms with the existing normalized-state-error metric,
  which reads only the xy dims and is therefore comparable across arms).

  > **The control is not a null, and should not be reported as one.** Measured on the pilot
  > (normalized state error by trial position after a switch, 1 seed):
  >
  > | condition | noise | head | t1 | t2 | t3 |
  > |---|---|---|---|---|---|
  > | RNN | 0.04 | on | 0.916 | 0.483 | 0.489 |
  > | RNN | 0.04 | off | 0.792 | 0.718 | 0.711 |
  > | NG α_z=0.4 | 0.04 | on | 0.532 | 0.154 | 0.097 |
  > | NG α_z=0.4 | 0.04 | off | 0.876 | 0.160 | 0.143 |
  > | Oracle Z | 0.04 | on | 0.059 | 0.055 | 0.067 |
  > | Oracle Z | 0.04 | off | 0.058 | 0.049 | 0.066 |
  >
  > The head changes xy behaviour for **both** non-oracle models and leaves the Oracle
  > untouched (as expected — it already has the context). It helps NG most at trial 1 (0.876 →
  > 0.532), and helps the RNN from trial 2 on (0.718 → 0.483) while making its trial-1
  > perseveration slightly *worse* (0.792 → 0.916). The effect shrinks as noise rises.
  >
  > So the head is a supervised gradient, not an inert readout. It cuts in the safe direction
  > for the hypothesis — it is the RNN's easier case from trial 2 on, so surviving perseveration
  > is a conservative estimate — but it belongs in the writeup rather than being asserted away.

- **Belief–behaviour agreement**, free from the same runs. `oi[t]` is the prediction *of* frame
  `ii[t]`, so the predicted attack sits on the outcome frame and the colour that cued it on the
  preceding cue frame; the attack's polar angle is `2πc/n_colors + rotation`, giving
  `θ̂ = atan2(pred_y, pred_x) − 2πc/n_colors` as a second, purely behavioural estimate of the
  same quantity — on the outcome frames, which is where judgements are scored by default. If a
  model reports one context while its predictions sit at another, that decoupling is itself a
  result.

---

## Making the context ambiguous

**At the task's shipped settings, slips are impossible by construction.** For the cued colour
the two candidate targets are separated by the chord `D = 2·target_radius·sin(sep/2)`, and each
observation is isotropic Gaussian with std `noise_std`. The optimal single-observation
classifier is their perpendicular bisector, so one trial misclassifies the context with
probability `Φ(−D / 2σ)` — `p_context_error_single_obs` in the analysis module:

| `noise_std` | sep = 60° (D = 0.500) | sep = 120° (D = 0.866) |
|---|---|---|
| 0.04 | 0.0000 | 0.0000 |
| 0.12 | 0.0186 | 0.0002 |
| 0.20 | 0.1056 | 0.0152 |
| 0.30 | 0.2023 | 0.0745 |

> **Use the chord, not the arc.** The observation lives in the plane, so the arc length
> `(sep/2)·target_radius` overstates the separation and understates the error — by ~3× at
> sep = 120°, σ = 0.20 (0.0044 vs 0.0152).

The design therefore **narrows the separation** (`train_rotations = [0.0, 60.0]`) rather than
just cranking the noise. Matching the same ambiguity at 120° would need `noise_std ≈ 0.34–0.49`
— scatter comparable to the arena itself, which confounds "cannot hold context" with "cannot
learn the targets". At 60° the targets stay crisp and easy to learn, because the colour is
*cued* and the network never has to discriminate colours from position; only the **context**
becomes uncertain.

`noise_std` is swept over `{0.04, 0.12, 0.20, 0.30}`, spanning 0 → 19% per-observation
ambiguity. That axis is what turns "slips happen" into "slips are caused by noise".

Neither 60° nor 120° is a multiple of `360/n_colors = 72°`, so no rotation maps one colour's
target onto another's.

---

## Removing the schedule confounds

The task ships with fixed-length blocks in a hard-cycled, seed-independent rotation order
(`datasets.py`, `rotations[block_idx % len(rotations)]`). A recurrent network could in principle
learn *when* the switch lands and *which* angle is next, which would undercut a perseveration
claim. Two opt-in flags remove that, both defaulting to the original behaviour:

| Flag | Default | Effect |
|---|---|---|
| `block_duration_distribution` | `'fixed'` | `'geometric'` varies block length (≈ 93–280 timesteps at `block_size=140`), so switch timing is unpredictable |
| `rotation_block_order` | `'cyclic'` | `'random_no_repeat'` redraws the angle each block, never repeating the previous one |

> **With two rotations `'random_no_repeat'` *is* alternation**, so for this headline design the
> confound actually removed is switch *timing*. The order flag earns its keep in the
> three-rotation variant. (The mean-prediction dataset likewise degenerates to strict alternation
> with two means, so this matches precedent.)

`'random_no_repeat'` never repeats an angle across a boundary, which also keeps
`get_block_switches` correct — it keys on the rotation *value*, so two consecutive same-angle
blocks would be silently merged.

---

## Metrics

Per block (blocks after the first, Phase `'Learning and inference'` only). Definitions match
`mean_prediction_analysis.extract_block_criterion_metrics`, with "side of the midpoint" replaced
by "nearest trained rotation"; `_find_criterion` is reused verbatim.

| Metric | Definition |
|---|---|
| **Perseveration errors** | Wrong judgements before the criterion run of `criterion_n` consecutive correct. If criterion is never reached, every error counts as perseverative |
| **Context slips** | Wrong judgements after criterion, same block. `NaN` when criterion was never reached — the model never adapted, so there is no "after" |
| **Slip rate** | slips / post-criterion judgements. Block lengths are geometric here, so raw counts are not comparable across blocks |
| **Trials to criterion** | Judgements elapsed until the criterion run ends. `NaN` when never reached |
| **`no_criterion`** | Fraction of blocks that never reached criterion. Guard: such blocks leave the slip average entirely. Measured 0.000 everywhere on the pilot |
| **`belief_norm`** | Where the belief sits between the two contexts: `d_current / (d_current + d_previous)` on the context-output vectors. **0 = current context, 0.5 = hedge, 1 = previous context.** One scalar for both "which context" and "how committed", no circular statistics, and 0.5 is the hedge point by construction |
| **`\|belief\|`** | Validity guard only, never plotted — a context vector collapsing toward the origin would make its angle meaningless. Reported in `summarize()` |

> **`belief_norm` is the repo's existing convention, not a new one.** It is the same
> normalized-state-error construction as `trial_norm_errors` in
> `rotating_targets_analysis._analyze_adaptation` (0 = correct, 0.5 = chance, 1 = the other
> context), applied to the belief head instead of the xy prediction. Two earlier attempts at
> this — a circular bias signed toward the previous context, and `|belief| / target_radius`
> against an 0.866 "hedge" level — were dropped: both needed a derivation before the number
> meant anything, and the first silently cancelled to ~0 when pooled across alternating blocks.
| **Belief–behaviour agreement** | Does the reported belief imply the same context as the xy prediction? |

> **Perseveration and slips partition a block's errors at the criterion, so read them together.**
> As trials-to-criterion grows, more of the block sits *before* criterion and errors are booked
> as perseveration rather than slips — a model can therefore show a *falling* slip rate while
> getting monotonically worse. On the pilot the RNN does exactly this between `noise_std` 0.20
> and 0.30 (slip rate 0.054 → 0.048) while its perseveration climbs 11.2 → 17.9 and its TTC
> climbs 18.2 → 26.1. That is why F4 plots trials-to-criterion beneath the slip rate, and why
> **perseveration errors per block is the monotone headline number**, not the slip rate.

### Ideal observer

A discrete Bayes filter over the trained rotations, run on the attacks the model **actually
saw** (reconstructed from the logged inputs):
`log p(a | θ_k) = −‖a − R(θ_k)·base_c‖² / 2σ²`, with a hazard-rate transition step derived from
the realised block structure, then the identical criterion analysis. Pure numpy, no training.

> **It reports the *predictive* MAP** — the belief after the transition step but *before* the
> current trial's attack is seen. The network is a predictor: its belief about frame `t` is
> formed from data through `t−1`. A filter scored on its posterior would be getting one free
> observation per trial, and would show 0 perseveration errors, which is not a floor any
> predictor could reach. Scored predictively it shows exactly **1.00** perseveration error per
> block at low noise — one trial of detection delay, which is the real normative floor.

Its slips are then the irreducible rate at that noise level.

---

## Sweep

13 conditions × 4 noise levels × 10 seeds = **520 runs**, ~3 min each single-threaded.

| Condition family | Purpose |
|---|---|
| `RNN`, `RNN (WU_lr=0.005)`, `RNN (WU_lr=0.01)` | Weights are the RNN's only route to adaptation, so a single learning rate would leave the result open to "you crippled the baseline". See the dissociation below — the answer is not the one the belief columns suggest |

> **The faster RNNs game the readout, and the belief metrics alone would say the opposite.**
> At `noise_std = 0.20`, `WU_lr = 0.005` and `0.01` look like the *best* models in the table —
> slips of 0.04 and 0.03 per block, beating NeuraGEM at almost every `Z_lr`. Their attack
> predictions tell a different story (normalized state error after a switch, 0 = new context,
> 1 = old):
>
> | condition | t1 | t2 | t3 | t4 | t5 | belief | agree |
> |---|---|---|---|---|---|---|---|
> | RNN (WU_lr=0.001) | 0.792 | 0.612 | 0.611 | 0.617 | 0.622 | 0.231 | 0.870 |
> | RNN (WU_lr=0.005) | 0.737 | 0.744 | 0.759 | 0.745 | 0.752 | 0.064 | **0.746** |
> | RNN (WU_lr=0.01) | 0.728 | 0.731 | 0.727 | 0.724 | 0.736 | 0.053 | **0.746** |
> | NG α_z=0.2 | 0.815 | 0.579 | 0.277 | 0.138 | **0.136** | 0.080 | 0.982 |
> | Oracle Z | 0.107 | 0.109 | 0.111 | 0.106 | 0.108 | 0.012 | 0.998 |
>
> The fast-weight RNNs sit flat at 0.73–0.75 for the entire block — still nearer the *old*
> context than the new one, and never adapting at all, which is **worse** than the baseline RNN
> (0.79 → 0.62). Raising `WU_lr` does not make the RNN better at the task; it makes it better at
> satisfying the auxiliary context output, which is a constant-per-block 2-vector and far easier
> to fit than the 5-colour × 2-context attack mapping.
>
> This is what the belief–behaviour agreement diagnostic exists for, and it is the reason it is
> not optional. `summarize()` marks any condition below `AGREEMENT_FLOOR` (0.85) with `*` and
> prints a warning; the flagged rows' belief columns must not be read as performance.
>
> For the writeup this strengthens rather than weakens the control: a faster baseline is not a
> better baseline, so "you crippled the RNN" is answered — and answered by a mechanism worth
> reporting on its own.
| `NG$\alpha_z=0.01$` … `0.9` | Z_lr grid, log-spaced (`0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9`), `Z_decay = 1e-3·Z_lr²` — same coupling as the mean-prediction sweep, so the two Z_lr axes are comparable |

> **The grid is log-spaced and starts at 0.01, not 0.1.** NeuraGEM has already converged by
> `Z_lr = 0.1`, so a linear 0.1–0.9 grid spends all nine points on models that behave
> identically; the structure that distinguishes them from the RNN is below it. The values are
> the classic 1-2-3-5-per-decade sequence, within 20% of true logspace and matching its
> geometric ratio (1.755) exactly. `CONDITION_INFO` shades them by **log** `Z_lr`
> (`_z_lr_shade`) — colouring by the raw value would give five of the nine conditions nearly
> identical dark purples.
>
> `HEADLINE_Z_LR` names the single NG condition used by the pilot and the per-model figures,
> and `ng_label()` builds every condition name, so neither can drift out of `_Z_LRS` the way a
> hard-coded label can. There is an assert for it.
| `Oracle Z (one-hot)` | Ceiling. The true rotation is handed in as Z, so the head only has to read it out — which also validates the head itself |

Only Phase 2 is trained (`run_test_phase=False`); generalization to novel rotations is
`run_rotating_targets_comparison.py`'s job.

`compact_logger` shrinks each pickle before saving. The logger appends one tiny `(1,1,D)` array
per timestep per field, so per-array pickle overhead dominates (15–20 MB/run raw). Each kept
field is collapsed to a single concatenated float32 array **wrapped in a one-element list**, so
`np.concatenate(field, axis=0).reshape(-1, D)` — what every consumer does — is unchanged.
`phases` and `others` are left alone.

```bash
# Stage 0 — calibration pilot: set PILOT = True in the config, then
python rotation_slips_perseveration_sweep.py       # 24 runs incl. the no-head control arm

# Stage 1 — full sweep
./submit_job.sh 519 rotation_slips

# Stage 2 — analysis
python rotation_slips_perseveration_analysis.py
```

Running the analysis module directly first executes a synthetic self-test — a perfect belief
must decode to the true rotation exactly, a stale belief must score as fully perseverative —
independent of any trained model.

### Stage 0 pilot result (1 seed)

Perseveration errors per block, asymptotic (last 3 block groups):

| `noise_std` | RNN | NG α_z=0.4 | Oracle Z | Ideal observer |
|---|---|---|---|---|
| 0.04 | 1.33 | 0.67 | 0.00 | 1.00 |
| 0.12 | 4.67 | 0.89 | 0.00 | 1.00 |
| 0.20 | 11.22 | 1.22 | 0.00 | 1.56 |
| 0.30 | 17.89 | 1.22 | 0.00 | 3.00 |

The RNN climbs 13× with noise; NeuraGEM stays flat and at or below the ideal observer's
detection-delay floor. Trials-to-criterion tells the same story (RNN 4.8 → 26.1, NG 3.7 → 4.4).
Belief–behaviour agreement also decouples for the RNN (0.977 → 0.809) while NG holds (0.999 →
0.937): under noise the RNN's *reported* context increasingly disagrees with the context its own
predictions imply.

Criteria all passed: Oracle at ceiling (1.000 correct at every noise level), NG learns under
geometric blocks (≥ 0.985), belief magnitude ≥ 0.895 everywhere so the angle readout is valid.

Note the tail context-correct rate saturates near 1.000 for every condition — it asks only
"did you get there eventually", and everyone does. The signal lives in perseveration, TTC and
slips, not in that number.

---

## Figures

All paper-panel sized; see [figure_style.md](figure_style.md).

| | Content |
|---|---|
| F1 | Reported belief per trial over the last few blocks, against the true rotation. One panel per model — the exception the style guide allows, since per-trial point clouds cannot overlay |
| F2 | Context-correct rate over training + asymptotic summary |
| F3 | Perseveration errors and **context slips per block** over training + asymptotic summaries |
| F4 | Asymptotic **slips per block** vs `noise_std`, with the ideal observer, and trials-to-criterion beneath. **The causal figure** |
| F5 | Belief vs trial-within-block, on the 0/0.5/1 scale. **The mechanistic figure** |
| D | Belief–behaviour agreement over training |

Counts per block, not rates: "2.9 slips per block" is a number you can hold; "a slip rate of
0.054" is not. F4 drops the memoryless reference for the same reason — it is an error *rate*,
and converting it to slips-per-block needs each model's own post-criterion window length, so it
would be a different number per curve rather than a shared reference.
`p_context_error_single_obs` remains for the design table above.

### What F5 settled, and what it ruled out

F5 is the panel that decides the interpretation, and it killed two framings I had put in
earlier drafts of this document:

1. **"The RNN settles at a shifted decision boundary."** Not supported. The belief decays
   monotonically through the block and never plateaus above 0 — at `noise_std=0.30` the RNN goes
   49° → 45° → 37° → 27° → 19° → 13° → 5° across trial positions 0-5 … 70+. It is not settling
   anywhere biased; it is still in transit when the block ends. Establishing an actual boundary
   shift would need a frozen-network probe at successive points within a block.
2. **"The RNN carries a graded estimate while NeuraGEM has discrete context states."** Also not
   supported. The RNN's belief distribution looks smeared only because it is sampled mid-transit.
   Tighten the cut and the difference evaporates:

   | trial position ≥ | RNN mean | NG mean | RNN fraction in [0, 0.1) | NG fraction |
   |---|---|---|---|---|
   | 20 | 0.268 | 0.129 | 0.226 | 0.518 |
   | 40 | 0.190 | 0.122 | 0.328 | 0.530 |
   | 60 | **0.127** | **0.119** | **0.505** | **0.527** |

   By trial 60 the two are nearly identical. Both models reach the same asymptotic context
   representation.

What survives is a single mechanism, and it is simpler than either: **the RNN updates context
roughly an order of magnitude more slowly** (trials-to-criterion 18–26 vs 4). Blocks are short
enough that it spends most of every one en route, and a belief in transit is close enough to the
midpoint that noise can push it across. That one time constant accounts for the perseveration
counts, the trials-to-criterion, the slips, and the belief distribution — no boundary movement
and no representational difference required.

An untested prediction that would strengthen this: the **integration kernel**, regressing the
belief at trial *t* on the implied angles of observations at *t−1, t−2, …*. The account above
predicts a short, shallow kernel for the RNN and a long one for NeuraGEM. Not implemented.

---

## Caveats

- `blocked_phase_length` is converted to a *block count* by `int(len / block_size)`, and
  geometric blocks then average ~1.16× nominal — the realised Phase-2 length exceeds the nominal
  16800 timesteps. The analysis works per block, so this only matters if you quote a length.
- Every phase rebuilds its dataset with `default_rng(env_seed)`, so Phase 1 and Phase 2 replay
  the same noise/permutation stream prefix. Harmless here — metrics are Phase-2 only.
- At outcome frames the model is trained to predict the *next cue* frame, whose xy target is
  `(0, 0)`. That is pre-existing behaviour under `output_loss_mask=[0,0,0,0,0,1,1]` and is
  unchanged; the context dims are supervised on every frame regardless.

---

## See Also

- [task_rotating_targets.md](task_rotating_targets.md) — the task, block structure, context IDs
- [task_mean_prediction.md](task_mean_prediction.md) — the experiment this ports
- [rotation_decoding.md](rotation_decoding.md) — decoding rotation from Z vs hidden activity
- [figure_style.md](figure_style.md) — `FigSize` presets and colour conventions
