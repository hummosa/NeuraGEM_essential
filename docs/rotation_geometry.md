# What Kind of Code Is It? — RSA on the Rotation Representation

`rotation_geometry_analysis.py`

[rotation_decoding.md](rotation_decoding.md) answers *how much* rotation information a
representation carries. This answers *what shape* that code has: does the model hold
rotation as a ring, as a set of learned categories, as a blend of learned contexts, or as
something else?

The method is **representational similarity analysis (RSA)**, chosen because it transfers
to neural data unchanged — every step after "average the state within each rotation"
operates on a matrix of pairwise distances, which you can compute from Z, from RNN hidden
activity, or from voxels/spikes.

---

## The idea in one paragraph

For each rotation angle, average the model's internal state over all the timesteps at that
angle. That gives one pattern per angle. Then ask, for every *pair* of angles, how
different those two patterns are — that matrix is the **RDM** (representational
dissimilarity matrix). Different theories about the code predict visibly different RDMs.
Rank-correlate the measured RDM against each prediction; the best match wins.

---

## The hypotheses

Reported by default (`CANDIDATES`):

| Name | Plain reading | Brain analogue |
|---|---|---|
| `circular` | State turns smoothly with angle and **wraps**: 10° and 350° are neighbours. | head-direction cells |
| `blend` | Like mixing paint: 45° = 60% of the 0° state + 40% of the 120° state. Smooth like `circular`, but its geometry comes from the trained set, not the circle — no wraparound, and angles far from any trained context get squashed together. | interpolation over learned exemplars |
| `nearest` | State reports only *which trained context is closest*. Flat within a trained angle's basin, sharp jump at the boundary. | categorical perception (phoneme, colour boundaries) |

Available via `candidates=ALL_CANDIDATES` (`OPTIONAL_CANDIDATES`):

| Name | Plain reading | Why it is off by default |
|---|---|---|
| `periodic` | Codes the target **constellation**, not the colour→location mapping. Cannot resolve θ from θ + 360/n_colors. | This is the analysis's **negative control** — it lands at ~0 whenever the model tracks the colour→location mapping, which is what shows the method can return "no". Run it once per new task configuration, not on every figure. |
| `seamed` | Smooth in angle but **cut** somewhere, like a number line whose ends sit far apart. | Correlates ~0.72 with `circular` by construction, so its bar largely tracks circular's and rarely adds independent evidence. Reach for it when wrap-vs-no-wrap is the actual question. |

`plot_candidate_rdms` draws them before any data is involved — start there. Their
signatures are distinctive: a smooth wrapping gradient (`circular`), blocky plateaus
(`nearest`), a repeating fine grid (`periodic`), one corner-to-corner ramp (`seamed`).

```python
# Include the optional hypotheses for a one-off control run:
from rotation_geometry_analysis import ALL_CANDIDATES
geometry_across_models(results, candidates=ALL_CANDIDATES)
```

> **The `periodic` period is set by `n_colors`, not by `train_rotations`.** With
> `n_colors=5`, the five evenly-spaced targets mean a 72° rotation maps the *set* of target
> positions onto itself — only which colour sits where changes. Changing which rotations
> you train on will not remove this hypothesis; changing `n_colors` would.

---

## Read the confusability table first

The hypotheses are **not orthogonal**. Under `train_rotations=[0, 120, 215]` with 24 test
angles, `blend` and `nearest` correlate at **0.79** — so a model scoring 0.60 on one and
0.55 on the other has not distinguished them.

```python
print_confusability(test_angles, cfg.train_rotations, cfg.n_colors)
```

This is also a **design lever**: which rotations you train on changes how separable the
hypotheses are. Measured maximum off-diagonal correlation for a few choices:

| `train_rotations` | max correlation | worst pair |
|---|---|---|
| `[0, 90, 180, 270]` | **0.71** | circular vs blend |
| `[0, 120, 210]` | 0.78 | blend vs nearest |
| `[0, 120, 215]` | 0.79 | blend vs nearest |
| `[0, 40, 80]` | 0.82 | blend vs nearest |
| `[0, 180]` | 0.84 | blend vs nearest |

More trained rotations, spread evenly, separate the hypotheses better. If distinguishing
`blend` from `nearest` is the goal, four evenly-spaced training angles is a sharper
experiment than three clustered ones.

---

## Noise ceiling

`noise_ceiling` is the split-half reliability of the measured RDM: the most any hypothesis
could possibly score given the data. Halves are split **by state-block** (falling back to
mini-block parity when an angle has only one block) — splitting at random would put
temporally adjacent, highly correlated timesteps on both sides and inflate the ceiling.

A candidate near the ceiling explains everything the data supports. A candidate far below
it is genuinely the wrong shape, rather than the data being too noisy to tell.

---

## API

```python
# One model or many; one logger or a list of seeds (same `as_runs` convention).
geometry_results = geometry_across_models(results, keys=('Z', 'H'),
                                          phase='phase3a', frames='all')
summarize_geometry(geometry_results, key='Z')

plot_candidate_rdms(angles, cfg.train_rotations, cfg.n_colors)   # what each predicts
plot_rdm_fits(geometry_results, key='Z')                         # which one wins
plot_geometry_embedding(geometry_results, key='Z')               # 2-D map of the code
```

Lower-level pieces, if you want to bring your own patterns (e.g. real neural data):

```python
patterns, angles = mean_pattern_by_angle(samples, key='Z')
rdm   = build_rdm(patterns, metric='euclidean')   # features z-scored first
cand  = candidate_rdms(angles, train_rotations, n_colors)
rho   = rdm_fit(rdm, cand['circular'])
xy    = classical_mds(rdm)                        # deterministic, for the 2-D map
```

`build_rdm` z-scores features before computing distances so that a few high-variance units
cannot dominate — this matters when comparing a 3-dim Z against a 64-dim hidden state.

Adding a hypothesis is one more entry in `candidate_rdms` — a function of the angle list
and task parameters. Nothing downstream changes.

---

## Reading `plot_geometry_embedding`

Each point is one rotation, positioned so distances match the measured RDM, coloured by
angle on a circular colour wheel; trained rotations are ringed in black.

- A **ring** running smoothly through the colour wheel → circular code.
- A **triangle** with points bunched at the corners → blend over trained contexts.
- Tight **separate clumps** → categorical code.
- A ring that is *uneven* — dense near trained angles, stretched between them → a circular
  code with categorical warping, which is a real possibility none of the five pure
  hypotheses captures on its own.

---

## The oracles do NOT calibrate this in Phase 3a

It is tempting to treat `Oracle Z (one-hot)` as a known-categorical code and
`Oracle Z (cont.)` as a known-circular one, and read their RSA fits as ground truth. **That
is wrong for Phase 3a.** `reconfigure_for_prediction` sets `what_latent_to_use='self'` for
every condition ([configs.py:165](../configs.py#L165)), so in the test phase no oracle uses
its designed encoding — all conditions self-infer Z by gradient descent and differ only in
the weights they learned. Measured consequence: `Oracle Z (one-hot)` returns a near-perfect
*circular* fit in Phase 3a, not a categorical one.

The oracle encodings are only in force during Phases 1–2, where there are just
`len(train_rotations)` distinct angles — too few for a meaningful RDM. So there is no
oracle calibration available for this analysis. Interpret the fits on their own terms, and
lean on the `constellation` hypothesis as the working negative control: it should sit near
zero, and if it does not, something is wrong with the pipeline.

## A ring may be partly structural, not learned

θ is a circular variable, the loss is smooth in θ, and Z is found by gradient descent to
minimise that loss — so a continuous closed manifold in Z-space is close to inevitable
regardless of what the model "chose" to represent. Empirically, four conditions trained
under quite different latent encodings all converge on a ring in Phase 3a, which is
consistent with the ring being largely forced by the task.

The claim that would carry more weight is about the ring's **metric**: uniform arc length
per degree means a genuine metric code, whereas arc bunched near the trained angles is the
categorical-perception signature and implies something learned. `plot_geometry_embedding`
hints at this (uneven spacing around the loop) but does not quantify it.

---

## Self-test

```bash
.venv/bin/python rotation_geometry_analysis.py
```

Builds synthetic representations with a known code (ring, 3-slot categorical, blend) and
checks RSA recovers the right hypothesis for each, with a high noise ceiling on clean data.
If a synthetic ring does not score highest on `circular`, nothing downstream is
trustworthy.

---

## See Also

- [rotation_decoding.md](rotation_decoding.md) — how much rotation information is present
- [task_rotating_targets.md](task_rotating_targets.md) — the task and its context IDs
- [figure_style.md](figure_style.md) — `FigSize` presets used by the plots
