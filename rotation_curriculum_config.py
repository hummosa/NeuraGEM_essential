"""Shared configuration for the three-stage cued-context curriculum on rotating targets.

Ports the Stage-1 / Stage-A / Stage-B protocol from
`../NeuraGEM_rl_objective/run_probabilistic_reversal.py` onto the rotating-targets task:

    S1  standard task, uncued switches      what_latent_to_use='self', WU + LU
    S2  CUED context                        what_latent_to_use='context_ids', LU off, WU on
    S3  uncued again, WEIGHTS FROZEN        what_latent_to_use='self', LU on, WU off

The cue is not in the observation — it is the ground-truth rotation injected as a one-hot into
Z, which multiplicatively gates the hidden state. S2 exists to force a Z -> behaviour mapping
into the weights; freezing the weights in S3 then makes recovery speed a readout of Z_lr alone.

Task design (rotations, block structure, noise) is imported from
rotation_slips_perseveration_config so the two experiments cannot drift apart. What differs here:
no context-belief head, and the three-stage schedule.

See docs/rotation_curriculum.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

import plot_style
from rotation_slips_perseveration_config import (
    NOISE_LEVELS, TRAIN_ROTATIONS, make_base_config as _slips_base_config,
)


# ── Task design ───────────────────────────────────────────────────────────────

HEADLINE_NOISE = 0.20   # the operating point the slips experiment settled on

# NO context-belief head. `enable_context_output` supervises the network to *report* the
# rotation, which is an intervention that actively encourages context representations. That was
# the measurement in the slips/perseveration experiment; here it would confound the thing being
# studied — and per docs/rotation_slips_perseveration.md it measurably changed xy behaviour for
# both non-oracle models. Context belief is inferred from behaviour instead (see the analysis
# module). Set to 'circular' to run the belief-head variation arm.
CONTEXT_OUTPUT_ENCODING: str | None = None


# ── Z decay ───────────────────────────────────────────────────────────────────

# Z_decay used to be applied twice — once as the Z optimizer's weight_decay and once as a
# gradient term in RNN_with_latent._apply_chunk_lr_and_decay — so every tuned value on disk had
# double its nominal effect. configs.Config.Z_decay_mode now selects the path; 'grad' applies it
# once, honours chunk_l2_losses, and means the same thing under Adam / AdamW / SGD.
Z_DECAY_MODE = 'grad'

# The *total* decay the existing sweep actually ran under: its `1e-3 * lr**2` went through both
# paths. Expressing the calibrated total and dividing by the mode keeps this experiment's Z_lr
# axis comparable with rotation_slips_perseveration's, which is the whole point of reusing it.
_effective_z_decay = lambda lr: 2e-3 * lr ** 2


def z_decay_for(lr: float, mode: str = Z_DECAY_MODE) -> float:
    """Z_decay field value giving `_effective_z_decay(lr)` of realised decay under `mode`."""
    return _effective_z_decay(lr) / (2.0 if mode == 'both' else 1.0)


# ── Grids ─────────────────────────────────────────────────────────────────────

# One axis: the Z_lr an individual has all along, carried unchanged through S1, S2 and S3. This
# is the actual experiment — everyone has their own latent-inference speed from the start, S2
# cues them, and S3 (weights frozen, cue withdrawn) is where that speed is recovered from
# behaviour. There is no separate "train" vs "test" Z_lr any more: S3 continues each individual
# at the same Z_lr it trained and was cued with.
#   'RNN' -> no_of_steps_in_latent_space=0 through S1 and S2 (LU only switches on for the
#   S3_PINNED_Z_SANITY branch below, never for the main S3 continuation of this arm): a model
#   that never developed latent inference at all. Kept as a qualitative baseline, not a point on
#   the numeric dial.
# The seven numeric values are the fine dose-response grid this experiment used to reserve for
# forking S3 alone; using it to drive S1 itself is what makes the individuals genuine.
Z_LR: List[Any] = ['RNN', 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.9]

# Sanity-check-only fork, not a Z_LR value: rerun S3 off the *same* S2 checkpoint as the main S3
# continuation, but with Z pinned (no_of_steps_in_latent_space=0) instead of carrying the
# individual's Z_lr forward. This is the load-bearing check from the old Z_lr_test=None arm: if
# a model with Z pinned still tracks context, something other than Z carries context across
# blocks and nothing downstream is interpretable. It stays a small, explicit fork off S2 — rather
# than a 9th Z_LR value — because "pinned" is not a point on the dial, it is LU switched off.
S3_PINNED_Z_SANITY: bool = True

# 'none' is the control, not a cue variant: S2 continues exactly as S1, so the cued arm is
# compared against the same amount of extra training. S1 does not depend on cue_mode, so both
# arms fork from identical S1 weights — a paired control.
CUE_MODES: List[str] = ['none', 'oracle_z']


# ── Stage schedule ────────────────────────────────────────────────────────────

S1_LENGTH      = 2000   # 
S2_LENGTH      = 4000
S3_LENGTH      = 3000    # ~60 nominal blocks -> ~60 switches per cell (the reference had 5)
PASSIVE_LENGTH = 500     # S1 only; S2 and S3 warm-start from a trained model

# The reference shortened its cued stage's blocks (200 -> 10 trials) "so frequent reversals keep
# context/Z relevant" — i.e. so within-block inference cannot substitute for the gate. Keeping
# block structure identical across stages instead makes per-block metrics directly comparable.
# If pilot acceptance criterion (a) fails, drop this to ~0.3 and rerun; nothing else changes.
S2_BLOCK_SCALE = 1.0

# Where Z starts when S3 hands control back to self-inference. The oracle one-hot passes through
# latent_activation_function, so S2 only ever showed the weights gates at softmax([1,0]) =
# [0.731, 0.269]; a uniform [0.5, 0.5] start is out of that distribution.
#   'last_cued' — pre-softmax one-hot of the LAST S2 block's rotation. In-distribution, and
#                 uncorrelated with S3's first block under 'random_no_repeat', so half the runs
#                 start wrong: the first switch is a genuine perseveration test.
#   'uniform'   — 1/Z_dim everywhere (what the reference used).
#   'zeros'     — leave Z at zeros; softmax makes that uniform anyway, but skips the rebuild.
S3_Z_INIT = 'uniform' # 'last_cued' I think this is overblown I doubt it'll have impact. The model leaves the region quickly.

# Each phase rebuilds its dataset with default_rng(config.env_seed), so without distinct offsets
# S2 and S3 would replay S1's exact block and noise stream.
STAGE_SEED_OFFSETS = {'S1': 0, 'S2': 1000, 'S3': 2000}


# ── Sweep scope ───────────────────────────────────────────────────────────────

DEFAULT_SEEDS: int = 10
DEFAULT_NOISE: List[float] = [HEADLINE_NOISE]
# Set DEFAULT_NOISE = ALL_NOISE to add the noise axis, which would give this experiment the
# slips-vs-noise causal figure's counterpart. Costs 4x the compute, so it is not the default.
ALL_NOISE: List[float] = list(NOISE_LEVELS)

PILOT: bool = True
PILOT_SEEDS: int = 1
PILOT_NOISE: List[float] = [HEADLINE_NOISE]

# The export path encodes noise and the training arm, but NOT stage lengths, S2_BLOCK_SCALE,
# S3_Z_INIT or the grids — those live in RUN_NAME or nowhere. Set this False after changing any
# of them, or stale trees are silently kept.
SKIP_EXISTING: bool = False


def active_seeds() -> int:
    return PILOT_SEEDS if PILOT else DEFAULT_SEEDS


def active_noise() -> List[float]:
    return list(PILOT_NOISE if PILOT else DEFAULT_NOISE)


# ── Base config ───────────────────────────────────────────────────────────────

def make_base_config(noise_std: float = HEADLINE_NOISE,
                     context_output_encoding: str | None = CONTEXT_OUTPUT_ENCODING):
    """Config shared by all three stages. Stage-specific fields are applied by the sweep.

    Delegates the task design to rotation_slips_perseveration_config.make_base_config so the
    rotations, block structure, gating and noise cannot drift between the two experiments, then
    overrides only what this experiment changes.
    """
    cfg = _slips_base_config(noise_std=noise_std,
                             context_output_encoding=context_output_encoding)

    cfg.Z_decay_mode         = Z_DECAY_MODE
    cfg.blocked_phase_length = S1_LENGTH
    cfg.passive_phase_length = PASSIVE_LENGTH

    # The oracle plumbing must be right on the config that BUILDS the model: RNN_with_latent
    # caches oracle_context_encoding / oracle_context_values at __init__ (models.py:55-62), so
    # flipping what_latent_to_use at the S2 boundary reads whatever was set here. S1 runs with
    # what_latent_to_use='self', which is exactly why this is worth asserting rather than
    # assuming — nothing would fail until S2.
    assert cfg.oracle_context_encoding == 'one_hot', (
        f"S2 injects the cue as a one-hot Z; got oracle_context_encoding="
        f"{cfg.oracle_context_encoding!r}.")
    assert np.allclose(np.asarray(cfg.oracle_context_values, dtype=float),
                       np.deg2rad(TRAIN_ROTATIONS)), (
        'oracle_context_values must be the trained rotations in radians, in slot order.')
    assert int(np.prod(cfg.latent_dims)) >= len(TRAIN_ROTATIONS), (
        f'one-hot oracle Z needs one slot per rotation: latent_dims={cfg.latent_dims} '
        f'for {len(TRAIN_ROTATIONS)} rotations.')

    return cfg


# ── Export paths ──────────────────────────────────────────────────────────────

_SEP = int(round(max(TRAIN_ROTATIONS) - min(TRAIN_ROTATIONS)))
_HEAD = 'head-off' if CONTEXT_OUTPUT_ENCODING is None else f'head-{CONTEXT_OUTPUT_ENCODING}'

# Bump this whenever the stage-tree/grid shape changes incompatibly. RUN_NAME never encoded the
# grid itself (only stage lengths, head setting and decay mode), and several numeric Z_LR values
# repeat from the old Z_lr_train/Z_lr_test split — without a version tag, this rework would write
# new-shape payload['S3'] dicts (keyed by cue_mode only) to the exact paths old-shape ones
# (keyed by (cue_mode, z_lr_test)) live at, silently clobbering them with an incompatible shape.
GRID_VERSION = 'zgrid1'

RUN_NAME    = (f"sep{_SEP}_{S1_LENGTH}-{S2_LENGTH}-{S3_LENGTH}"
               f"_{_HEAD}_decay-{Z_DECAY_MODE}_{GRID_VERSION}")
EXPORT_ROOT = Path(f"./exports/rotation_curriculum/{RUN_NAME}")


def train_key(z_lr: Any) -> str:
    """Directory name for one Z_lr arm."""
    return 'RNN' if z_lr == 'RNN' else f'Zlr-{z_lr}'


def result_path(z_lr: Any, noise_std: float, seed: int) -> Path:
    """One pickle per (Z_lr, noise, seed) — the whole S2/S3 tree lives inside it."""
    return (EXPORT_ROOT / train_key(z_lr) / f'noise_std-{noise_std}'
            / f'results_seed-{seed}.pkl')


# ── Labels and colours ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConditionInfo:
    label: str
    color: Any


def zlr_label(z_lr: Any) -> str:
    return 'RNN' if z_lr == 'RNN' else rf'$\alpha_z={z_lr}$'


def _log_shade(lr: float, values: Sequence[float]) -> float:
    """Colormap position for a Z_lr, spaced by log.

    Same reasoning as rotation_slips_perseveration_config._z_lr_shade: on a linear scale the low
    end of the grid collapses to a handful of near-identical dark hues.
    """
    finite = [v for v in values if v]
    lo, hi = np.log10(min(finite)), np.log10(max(finite))
    return float((np.log10(lr) - lo) / (hi - lo)) if hi > lo else 0.5


# One grid now, one colour scheme: plasma for the numeric dial (matching alpha_z in the slips
# figures; purple -> magenta -> orange -> yellow, never touches green) plus the RNN's registered
# project colour for the qualitative no-latent arm.
_NUMERIC_ZLR = [z for z in Z_LR if z != 'RNN']
ZLR_INFO: Dict[Any, ConditionInfo] = {
    'RNN': ConditionInfo(zlr_label('RNN'), plot_style.get_model_color('RNN')),
    **{lr: ConditionInfo(zlr_label(lr),
                         plt.cm.plasma(0.04 + 0.82 * _log_shade(lr, _NUMERIC_ZLR)))
       for lr in _NUMERIC_ZLR},
}

# The one sanity-check fork (S3 with Z pinned): not part of the Z_LR dial, so it gets its own
# entry rather than a fake spot on the plasma ramp.
PINNED_Z_INFO = ConditionInfo('Z pinned', '0.55')

CUE_INFO: Dict[str, ConditionInfo] = {
    'oracle_z': ConditionInfo('Cued (oracle Z)', plot_style.get_model_color('Oracle Z (one-hot)')),
    'none':     ConditionInfo('Uncued control', '0.45'),
}

# The Z_lr the single-arm figures use when one representative is needed.
HEADLINE_ZLR: Any = 0.2 if 0.2 in Z_LR else Z_LR[-1]


# ── Pilot acceptance criteria ─────────────────────────────────────────────────
#
# Checked by rotation_curriculum_analysis.check_acceptance() before committing SLURM time to the
# full sweep. Each is a failure mode that would make the headline result uninterpretable, not a
# nice-to-have.
ACCEPTANCE = dict(
    # (a) The cue is actually being used, judged two ways. Failure on either => the weights are
    #     inferring context internally rather than reading the gate; drop S2_BLOCK_SCALE to ~0.3
    #     (what the reference did) and rerun.
    #
    #     Perseveration errors per block in S2, cued arm. The threshold is the ideal observer's
    #     one-trial detection-delay floor: a cue that names the *next* block's rotation removes
    #     detection delay entirely, so a model reading the gate must score below what the best
    #     possible inference-from-evidence can. Measured on the pilot: 0.00-0.63.
    s2_cued_max_persev       = 1.0,
    #     Normalized state error in the later half of S2, cued arm. This one has an irreducible
    #     floor set by residual prediction error at this noise level, NOT 0 — the slips
    #     experiment measures Oracle Z at 0.107 for the same metric at noise 0.20, and the pilot's
    #     cued arm lands at 0.095-0.127, i.e. at that ceiling. 0.15 sits clear of the floor and
    #     well below the uncued control (0.16-0.39). An earlier 0.10 was set below the achievable
    #     floor and failed a model that was demonstrably at ceiling on every other measure.
    s2_cued_max_norm_error   = 0.15,
    # (b) The dial works. Ratio of worst to best perseveration errors/block across Z_LR in the
    #     cued arm's S3. This is the "did the finicky part work" test.
    s3_min_zlr_spread        = 3.0,
    # (c) Z is the only context channel. The S3_PINNED_Z_SANITY branch must fail badly; if it
    #     does not, something else carries context across blocks and nothing downstream is
    #     interpretable.
    s3_frozen_min_norm_error = 0.40,
    # (d) The behavioural readout is valid — a prediction collapsing toward the origin would make
    #     its angle meaningless. ||pred_xy|| / target_radius; 0.866 is exact hedging at sep 60.
    min_belief_mag           = 0.50,
)
