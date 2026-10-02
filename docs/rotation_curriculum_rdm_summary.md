# RDM analysis — where things stand

One page. Detail in [rotation_curriculum_rdm.md](rotation_curriculum_rdm.md), code in
`rotation_curriculum_rdm.py`.

## The question

For the grant: show, in the model, the analysis we would run on human fMRI — **does a context
representation appear, and does it develop with training?**

Method: take one pattern per trial, sample trials across the whole of Stage 1, and build a
trial × trial dissimilarity matrix (euclidean). The task alternates two rotations block by block,
so if the model holds context the matrix should show a **checkerboard**. Trials run
chronologically, so the diagonal is training time.

## What was run

`RECORD_HIDDEN = True`, `PILOT = False`, then `./submit_job.sh 79 curriculum` — 80 trees
(8 `Z_lr` × 10 seeds), 1.2 GB, no errors. Behaviour reproduces the previous run to 0.0005, so it
is a clean superset of the existing results, not a replacement.

Three representations come out of it, one pattern per trial at the **cue frame** (where the
rotation is not in the input, so it has to come from held context):

| | |
|---|---|
| `Z` | the latent itself, 2-D |
| `H` | hidden state **after** the Z gate multiplies it — what the readout sees |
| `H_pre` | the same timestep **before** the gate — what the recurrence holds on its own |

`H_pre` exists because the config is `post_gating=True, pre_gating=False`: Z never enters the
recurrence, so these two are different quantities and only one of them can carry the gate.

## The figures

In `exports/rotation_curriculum/sep60_2000-4000-3000_head-off_decay-grad_zgrid1_hid/figures/`.
Two sets of three — the same plot under three trial selections. Columns are four `Z_lr` arms,
each panel a trial × trial dissimilarity matrix, seed-averaged over 10 seeds. Red/yellow strips
on two edges mark which context each block was.

| file | rows | trials in the matrix |
|---|---|---|
| **`rdm_matrices.pdf`** | `Z`, `H` | **both** windows per block — 10 trials, early and late interleaved, so the checkerboard has a 10-row period. 130 × 130 |
| **`rdm_matrices_early.pdf`** | `Z`, `H` | the **early** window only — 2nd mini-block, 5 trials just after the switch. 65 × 65 |
| **`rdm_matrices_late.pdf`** | `Z`, `H` | the **late** window only — last complete mini-block, 5 trials once settled. 65 × 65 |
| `rdm_matrices_pregate{,_early,_late}.pdf` | `H_pre` | same three selections, for the pre-gate hidden state |

`Z` is the latent; `H` the hidden state **after** the Z gate multiplies it; `H_pre` the same
timestep **before** the gate. `H_pre` is in its own file because it is the null case — Z never
enters the recurrence under `post_gating`, so keeping it as a third row only shrank the two rows
that carry signal.

Two things about the colour scale:

- **Per representation.** A 2-D latent and a 64-unit population live on different distance
  scales, so absolute dissimilarity is not comparable between rows.
- **Shared across the three window views**, taken from the 2nd–98th percentile of the full
  matrix. That is deliberate: it is what makes `early` and `late` comparable in *level* as well
  as in pattern. The cost is that whichever window has the narrower spread looks washed out —
  and that washing out is itself the result, not an artifact.

`main(extras=True)` still produces the derived-index panels (context modulation against training
block, the directional "which context" index, the mini-block map, the behaviour scatter, the
cue/outcome contrast). They are not computed unless asked for.

Arms shown are `RNN` / 0.01 / 0.2 / 0.9 — no latent, too slow, balanced, too fast. Change with
`plot_rdm_grid(..., arms=(...))`; `Z_lr = 0.05` is worth a look on the late window specifically,
where it has the strongest separation in the grid.

### What the window split shows

The window matters, and it matters *differently* depending on `Z_lr`:

| `Z_lr` | early | late |
|---|---|---|
| 0.05 | 0.65 | **1.16** |
| 0.2 | **1.07** | 0.76 |

A well-tuned latent (`0.2`) separates the contexts immediately after a switch and loosens
slightly by the end of a block. A slow one (`0.05`) has barely moved in the early window but is
the most separated of any arm once settled. Same direction in post-gate `H` (0.22 early vs 0.14
late at `Z_lr = 0.2`). Pre-gate `H` is featureless in both windows, every arm.

## What they show

1. **The checkerboard is there, and it depends on `Z_lr` as behaviour does.** Strongest around
   `Z_lr = 0.1`–`0.2`, which is where the behavioural dose-response also peaks. `Z_lr = 0.01`
   shows a smooth diagonal *gradient* instead of a checkerboard — that is drift, not context.
   The `RNN` arm is flat (its Z is constant).

2. **Post-gate hidden activity represents context; pre-gate does not.** At `Z_lr = 0.2`, 61 of 64
   units separate the contexts in `H` against 7 of 64 in `H_pre`, where the no-context floor is
   0 of 64. `H_pre` looks like uniform noise in every panel. So the context signal in the hidden
   state is **entirely imposed by the gate** — the recurrent state does not distinguish the two
   contexts. This was the suspicion about `post_gating`, and it holds.

3. **34 of 64 hidden units are identically zero after the gate.** With
   `P_gates_bernoulli_prob = 0.3` and `Z_dim = 2`, about half the units draw an all-zero mask and
   are silenced. The effective post-gate population is ~30 units, not 64. Probably worth knowing
   before quoting a population size.

## Two things to distrust

- **Comparing the "context modulation" number across rows is invalid.** Its scale shrinks as the
  effect spreads over more dimensions: the same per-unit effect gives ~1.1 in 2-D `Z` and ~0.20
  in 64-unit `H`. Use per-unit discriminability across representations, that number only within
  one.
- **On `rdm_which_context_*`, 0.5 is not the neutral point.** The two reference patterns come from
  different points in training, so drift alone offsets it — the `RNN` arm, which has no context
  representation at all, reads 0.55 rather than 0.50. Read every arm against the `RNN` line.

## Open

- **`pre_gating=True`** is the obvious next run: one line in the config, and the analysis reads it
  with no further work. It is the direct test of whether a context representation can live in the
  recurrence at all.
- S1 was deliberately **not** lengthened — the well-tuned arms are already saturated in the first
  few blocks, so more blocks would extend a plateau.
- The fMRI translation needs the cue and outcome periods to be separable in the design, or the
  matrix carries the stimulus rather than held context.
