"""Shared configuration for the rotation perseveration / context-slip experiment.

Ports the mean-prediction perseveration experiment onto the rotating-targets task. The model
is asked to report its context belief directly — the same augmented-input trick
MeanPredictionConfig uses, here via RotatingTargetsConfig.enable_context_output — so
perseveration after a switch and slips under noise can be read straight off a per-trial
"which rotation do you think you are in" judgement.

Sweeps NeuraGEM over Z_lr, plain-RNN baselines over WU_lr, and an oracle ceiling, crossed with
observation noise. Structure mirrors mean_prediction_sweep_config.py; both
rotation_slips_perseveration_sweep.py and rotation_slips_perseveration_analysis.py import
exclusively from here.

See docs/rotation_slips_perseveration.md for the design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

import plot_style
from configs import RotatingTargetsConfig

COLOR_SCHEME = plot_style.Color_scheme()


# ── Task design ───────────────────────────────────────────────────────────────

# Two contexts 60 deg apart. The separation, not the noise, is what creates context ambiguity
# here. The cued colour's two candidate targets are a chord D = 2*target_radius*sin(sep/2)
# apart, so one observation confuses the contexts with probability Phi(-D / (2*noise_std)) —
# see p_context_error_single_obs in the analysis module (use the chord, not the arc: the
# observation is in the plane, and the arc understates the error by ~3x at sep=120):
#
#   noise_std   sep=60 (D=0.500)   sep=120 (D=0.866)
#     0.04           0.0000              0.0000
#     0.12           0.0186              0.0002
#     0.20           0.1056              0.0152
#     0.30           0.2023              0.0745
#
# At sep=120 deg (the setting run_rotating_targets_comparison.py uses) you would need
# noise_std ~= 0.34-0.49 for the same ambiguity — scatter as large as the arena, which confounds
# "cannot hold context" with "cannot learn the targets". At sep=60 deg the targets stay crisp
# and easy to learn (the colour is *cued*, so the network never discriminates colours from
# position); only the context becomes uncertain. That is the manipulation we want.
TRAIN_ROTATIONS = [0.0, 60.0]

# Spans 0 -> 19% per-observation ambiguity, which the slips-vs-noise figure needs for range.
NOISE_LEVELS = [0.04, 0.12, 0.20, 0.30]

CONTEXT_OUTPUT_ENCODING = 'circular'   # target_radius * [cos theta, sin theta]
CONTEXT_LOSS_WEIGHT     = 1.0          # output_loss_mask entry for the context dims


# ── Sweep conditions ──────────────────────────────────────────────────────────

# Z_decay scales as Z_lr^2 (power-law fit to hand-tuned values; rough approximation). Same
# coupling as the mean-prediction sweep, so the two experiments' Z_lr axes are comparable.
# Note the extrapolation to the new low end: at Z_lr=0.01 this gives Z_decay=1e-7, effectively
# no regularisation. That is the intended reading — a latent that barely moves and is barely
# pulled back is the continuous approach to the RNN limit, not a separate regime.
_z_decay_for_lr = lambda lr: 1e-3 * lr ** 2

# Sampled by response, not by parameter. A uniform log grid is the right *prior* when you do not
# know where the response changes; it is the wrong posterior once you do, because uniform spacing
# across a saturated region is wasted sampling. The first sweep measured this (perseveration
# errors and slips per block, mean over training, noise 0.20, 10 seeds):
#
#   Z_lr    0.01  0.02  0.03  0.05   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9
#   persev 17.0  18.2  18.7  14.6   5.3   2.5   1.8   1.6   1.8   2.4   3.5   5.0   8.7
#   slips   1.71  1.24  1.03  0.06  0.00  0.01  0.10  0.32  0.52  0.71  0.97  1.25  1.69
#
# Three regions, and the grid now spends points in proportion to how fast each moves:
#   0.01-0.03  saturated at the RNN level, flat -> two points suffice (0.02 dropped)
#   0.04-0.15  the transition, and much the steepest part: 0.05 -> 0.1 alone drops 9.3, more
#              than the whole rest of the curve. This is where the grid was thinnest.
#   0.2-0.9    the U-shaped optimum and its degradation limb, already dense on disk
#
# Perseveration bottoms out at Z_lr=0.4 while slips bottom out at 0.1-0.2: a latent that adapts
# faster escapes the old context sooner but then chases observation noise. The U is that
# speed/stability tradeoff, which is why the upper limb is worth resolving rather than trimming.
_Z_LRS = [0.01, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15,
          0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# The NG condition used wherever a single representative is needed (pilot, per-model figures).
# Derived rather than written out, so it cannot drift out of _Z_LRS the way a literal would.
#   0.2 is chosen off the table above: slips are effectively zero (0.01) and perseveration is
#   within 40% of its minimum. It is selected on the same data it gets shown against, which is
#   fine as "NeuraGEM tuned" when the whole sweep is reported beside it, but it is not an
#   independent estimate of the best Z_lr.
HEADLINE_Z_LR = 0.2
assert HEADLINE_Z_LR in _Z_LRS, f'HEADLINE_Z_LR={HEADLINE_Z_LR} is not one of _Z_LRS={_Z_LRS}'


def ng_label(lr) -> str:
    """Condition name for one NeuraGEM Z_lr. Also the export directory name."""
    return rf"NG$\alpha_z={lr}$"


def _z_lr_shade(lr) -> float:
    """Colormap position for a Z_lr, spaced by log so the low end does not collapse to one hue.

    Colouring by the raw value would give 0.01-0.1 — five of the nine conditions — nearly
    identical dark purples.
    """
    import numpy as _np
    lo, hi = _np.log10(min(_Z_LRS)), _np.log10(max(_Z_LRS))
    return float((_np.log10(lr) - lo) / (hi - lo))


# The RNN's only route to adaptation is its weights, so a single WU_lr would leave the result
# open to "you crippled the baseline". Three rates bracket it.
_RNN_WU_LRS = [0.001, 0.005, 0.01]

CONDITIONS: Dict[str, Dict[str, Any]] = {
    "RNN": dict(no_of_steps_in_latent_space=0),
    **{
        f"RNN (WU_lr={lr})": dict(no_of_steps_in_latent_space=0, WU_lr=lr)
        for lr in _RNN_WU_LRS[1:]
    },
    **{
        ng_label(lr): dict(
            no_of_steps_in_latent_space=1,
            Z_lr=lr,
            Z_decay=_z_decay_for_lr(lr),
        )
        for lr in _Z_LRS
    },
    # Ceiling: the true rotation is handed in as Z, so the belief head only has to read it out.
    # Doubles as a check that the head itself works.
    "Oracle Z (one-hot)": dict(
        no_of_steps_in_latent_space=0,
        what_latent_to_use='context_ids',
        oracle_context_encoding='one_hot',
    ),
}

DEFAULT_SEEDS: int = 10

# ── Pilot (Stage 0) ───────────────────────────────────────────────────────────
# Flip PILOT to run the calibration subset instead of the full sweep: does the Oracle reach
# ceiling at every noise level (is the operating point learnable at all), does NG learn under
# geometric blocks, and does the belief head leave the primary xy task unchanged?
PILOT: bool = False
PILOT_CONDITIONS = ["RNN", ng_label(HEADLINE_Z_LR), "Oracle Z (one-hot)"]
PILOT_SEEDS: int = 1
# Control arm: context head off entirely. Its xy-adaptation curve must match the head-on runs,
# which is what shows the head is a readout rather than a change to the task.
PILOT_INCLUDE_NO_HEAD_CONTROL: bool = True

# A condition's directory name encodes its Z_lr and the noise level, and Z_decay is a pure
# function of Z_lr, so an existing pickle at a given path was produced by the same settings the
# job would use now — safe to reuse when the grid changes shape (as it did when _Z_LRS went
# log-spaced: five of the nine NG conditions carried over untouched).
#   Set this back to False after changing anything NOT in the path — blocked_phase_length,
#   block structure, the gating setup, TRAIN_ROTATIONS — or stale runs will be silently kept.
SKIP_EXISTING: bool = True


# ── Base config ───────────────────────────────────────────────────────────────

def make_base_config(noise_std: float = 0.20,
                     context_output_encoding: str | None = CONTEXT_OUTPUT_ENCODING):
    """Config shared by every condition.

    Adapted from run_rotating_targets_comparison.make_base_config (lines 108-155) rather than
    imported: that module trains at import time, so importing it would run an experiment.

    context_output_encoding=None gives the no-head control arm — identical task, no belief
    output, so the xy behaviour can be compared with and without the head.
    """
    cfg = RotatingTargetsConfig()
    cfg.train_rotations = list(TRAIN_ROTATIONS)
    cfg.noise_std       = noise_std

    # Must come after train_rotations: the encoding sizes itself from it, and it rewrites
    # input_size / output_size / input_feed_mask / output_loss_mask.
    if context_output_encoding is not None:
        cfg.enable_context_output(context_output_encoding, loss_weight=CONTEXT_LOSS_WEIGHT)

    # Oracle Z one-hots each trained rotation into its own slot, so Z_dim must cover them all.
    # Kept identical across conditions so every model has the same latent capacity.
    cfg.latent_dims = [max(2, len(cfg.train_rotations))]

    # block_size must be recomputed with the x2 that two-timesteps-per-trial requires; the
    # config's own formula omits it, so the field would otherwise mean half what it says.
    cfg.n_miniblocks_per_state_block = 14
    cfg.block_size = cfg.n_miniblocks_per_state_block * cfg.n_colors * 2   # 140

    cfg.add_passive_learning_phase = True
    cfg.passive_phase_length = 500
    # -> int(16800 / 140) = 120 blocks. Geometric block lengths then average ~1.16x nominal, so
    # the realised phase is longer than 16800 timesteps; the analysis works per block, so this
    # only matters if you want to quote a training length.
    cfg.blocked_phase_length = 16800

    # Remove the schedule confounds: with fixed-length blocks in a hard-cycled rotation order,
    # a recurrent net could in principle learn *when* the switch lands and *which* angle is
    # next, which would undercut a perseveration claim.
    #   NOTE with only two rotations 'random_no_repeat' *is* alternation, so for this headline
    #   design the confound actually removed is switch timing, via the geometric durations. The
    #   order flag earns its keep in the three-rotation variant.
    cfg.block_duration_distribution = 'geometric'
    cfg.rotation_block_order        = 'random_no_repeat'

    cfg.WU_lr = 0.001
    cfg.pass_previous_latent = True

    cfg.use_add_gating = False
    cfg.use_mul_gating = True
    cfg.pre_gating     = False
    cfg.post_gating    = True
    # Post-gating defaults; per-condition overrides replace these. run_rotating_targets_
    # comparison.py hand-tuned this pair to (0.5, 1e-4) — the power law above gives 2.5e-4 at
    # Z_lr=0.5, which the pilot confirms still learns.
    cfg.Z_lr    = 0.5
    cfg.Z_decay = _z_decay_for_lr(0.5)

    cfg.log_hidden_states = False   # behavioural analysis only; keeps the pickles small
    cfg.run_no_updates    = False

    return cfg


# ── Export paths ──────────────────────────────────────────────────────────────

_SEP = int(round(max(TRAIN_ROTATIONS) - min(TRAIN_ROTATIONS)))
RUN_NAME    = f"sep{_SEP}_blocked_{16800}_geometric"
EXPORT_ROOT = Path(f"./exports/rotation_slips/{RUN_NAME}")


# ── Condition display metadata ────────────────────────────────────────────────

@dataclass(frozen=True)
class ConditionInfo:
    label: str
    color: Any


CONDITION_INFO: Dict[str, ConditionInfo] = {
    "RNN": ConditionInfo("RNN", plot_style.get_model_color("RNN")),
    **{
        f"RNN (WU_lr={lr})": ConditionInfo(
            rf"RNN $\alpha_w={lr}$",
            plt.cm.Greens(0.45 + 0.25 * i),
        )
        for i, lr in enumerate(_RNN_WU_LRS[1:])
    },
    **{
        # plasma, not viridis: viridis spends its middle third on greens, which is exactly the
        # RNN's registered colour, so the baseline was reading as another point in the sweep.
        # plasma runs purple -> magenta -> orange -> yellow and never touches green, leaving the
        # RNN unambiguous without overriding its project-wide colour. Clamped short of the
        # washed-out top end so alpha_z=0.9 stays visible on white.
        ng_label(lr): ConditionInfo(
            rf"NG $\alpha_z={lr}$",
            plt.cm.plasma(0.04 + 0.82 * _z_lr_shade(lr)),
        )
        for lr in _Z_LRS
    },
    "Oracle Z (one-hot)": ConditionInfo(
        "Oracle Z", plot_style.get_model_color("Oracle Z (one-hot)")),
}

# Model-class subset used by the figures that would be unreadable with 13 lines
# (the belief-trajectory panels and the slips-vs-noise curve).
HEADLINE_CONDITIONS = ["RNN", ng_label(HEADLINE_Z_LR), "Oracle Z (one-hot)"]

# ── Which conditions appear in the perseveration/slips figure (F3) ────────────
#
# Deliberately fewer than CONDITIONS. Two omissions, both reported in the text instead:
#
#   RNN (WU_lr=0.005 / 0.01) — their belief-behaviour agreement is 0.75, below
#       AGREEMENT_FLOOR. The belief head decoupled from the attack prediction, so their slip
#       counts (0.03-0.04/block, apparently the best in the sweep) describe the readout rather
#       than the behaviour: the xy prediction sits flat at 0.73-0.75 all block and never adapts.
#       Plotting them would put the sweep's most misleading numbers in its most prominent panel.
#   Oracle Z — pinned at 0 on both metrics by construction. A flat line at zero adds a colour
#       to the legend and nothing to the comparison; quote it in the caption.
#
# The RNN baseline stays, and is drawn apart from the NeuraGEM family: it is a different model,
# not the alpha_z -> 0 end of the sweep, so it gets its own x slot, its own colour outside the
# viridis ramp, and no connecting line to the family.
F3_CONDITIONS = ["RNN"] + [ng_label(lr) for lr in _Z_LRS]

# Time-course panels stay legible at ~6 lines; the dose-response panel beside them carries every
# condition, so nothing is hidden by thinning these.
F3_CURVE_CONDITIONS = ["RNN"] + [ng_label(lr) for lr in (0.01, 0.05, 0.1, 0.2, 0.4, 0.9)]


def active_conditions() -> Dict[str, Dict[str, Any]]:
    """CONDITIONS, or the pilot subset when PILOT is set."""
    if not PILOT:
        return CONDITIONS
    return {k: v for k, v in CONDITIONS.items() if k in PILOT_CONDITIONS}


def active_seeds() -> int:
    return PILOT_SEEDS if PILOT else DEFAULT_SEEDS
