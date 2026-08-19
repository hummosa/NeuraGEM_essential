# Progress Report — Computational Mechanisms of Control Allocation Across Timescales

**Award:** CRCNS — *A task-general theory of cognitive control through inference over latent abstractions*
**PIs:** M. Nassar, M. Ullsperger, H. Kirschner, A. Hummos
**Reporting period:** first project period · **Aim addressed:** Aim 1 (computational modeling)

## Background, question, and plan

Our central hypothesis is that cognitive control is implemented through *inference over a small set of latent embedding units* in a recurrent neural network (RNN): weights capture slow, generalizable structure, while rapid gradient-based updates to the latent units reconfigure information flow to meet the current trial's demand. For the flanker task at the center of Aim 1, this raises the question of *which* latent variable a controller must operate over for human-like control to emerge. Our plan is to train a single RNN, then freeze its weights and let candidate embeddings be inferred online — updated by feedback on every trial — comparing the resulting synthetic behavior and within-trial dynamics against the human flanker data to see which latent best reproduces the fingerprint of control.

## Results so far

We evaluated the first candidate: **a latent that encodes the spatial location of the target**. We trained a recurrent neural network (RNN) to report the direction — left or right — of one or more arrows presented across a set of spatial slots. Critically, the identity of the task-relevant slot was supplied to the network as a vector through a latent embedding. After training, this embedding functions as a controller that directs the RNN toward the slots it should attend to; we then update it through error feedback on every trial. This single mechanism already reproduces a substantial portion of the human control fingerprint:

- **Congruency effect** — incongruent trials show lower accuracy and slower, right-shifted reaction times.
- **Distance × congruency interaction** — near flankers interfere more than far flankers.
- **Within-trial conflict dynamics** — the decision variable is first pulled toward the flanker-favored response, then overridden late by a target-driven signal on correct trials, mirroring the human beta-power-lateralization (BPL) signal.
- **Post-error adaptation (in progress)** — we are quantifying the predicted post-error slowing, post-error increase in accuracy, and reduced congruency effect following errors.

The surprise here is one of *parsimony*: we did not need an embedding that explicitly represents "conflict" for conflict-like control to emerge. An attentional latent — arguably the simplest controller in our candidate set — when placed under feedback from trial outcomes, already captures congruency effects, the near/far interaction, and the within-trial dynamics that map onto the neural signal. This validates the core claim that inference over latents behaves as a task-general controller, and sharpens the next comparison: whether an explicit conflict embedding or a data-driven one adds explanatory power over this spatial-attention baseline, particularly for across-trial, history-dependent effects.

## Suggested figures (two)

1. **Behavioral fingerprint — congruency × distance.** Accuracy and reaction-time distributions across the four near/far × congruent/incongruent conditions — the headline behavioral result. *(`flanker_stage3_results.pdf`)*
2. **Within-trial evidence-accumulation dynamics.** The sign-normalized decision variable over time, showing the early flanker-driven dip and its late reversal — linking the model's internal dynamics to the human BPL signature used to validate the models. *(`flanker_stage4_accumulation.pdf`)*
