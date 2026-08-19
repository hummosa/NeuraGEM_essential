"""Perseveration and context slips on the rotating-targets task.

Loads pickles produced by rotation_slips_perseveration_sweep.py and asks whether a model that
never forms a usable latent state (plain RNN) perseverates on the previous rotation after an
unsignalled switch, and slips back to it under observation noise once it has adapted.

The context belief is read straight off the network's own output. RotatingTargetsConfig.
enable_context_output appends dims carrying target_radius * [cos theta, sin theta], hidden from
the model input by input_feed_mask and supervised by output_loss_mask — the same augmented-input
trick MeanPredictionConfig uses to make the network emit the latent mean. So:

    belief_rad = arctan2(oi[:, nc+1], oi[:, nc])

and because logger.context_ids is the context of the frame being *predicted*, belief and ground
truth land at the same index with no off-by-one. Everything downstream — criterion detection,
perseveration/slip counting, seed aggregation — is then the mean-prediction analysis with
"side of the midpoint" replaced by "nearest trained rotation".

Usage:
    python rotation_slips_perseveration_analysis.py
"""

from __future__ import annotations

try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic('load_ext', 'autoreload')
        _ip.run_line_magic('autoreload', '2')
except Exception:
    pass

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import plot_style
from plot_style import FigSize

from rotating_targets_analysis import flatten_logger, get_target_positions
from mean_prediction_analysis import (
    _aggregate_per_block_curves, _asymptote_panel, _find_criterion, _prune_legend_handles,
)
from rotation_decoding_analysis import _angular_error_deg
from rotation_slips_perseveration_config import (
    CONDITION_INFO, CONTEXT_OUTPUT_ENCODING, ConditionInfo, EXPORT_ROOT,
    F3_CONDITIONS, F3_CURVE_CONDITIONS,
    HEADLINE_CONDITIONS, NOISE_LEVELS, TRAIN_ROTATIONS,
)

plot_style.set_plot_style()

# Below this belief-behaviour agreement, a condition's belief metrics no longer describe what
# the network actually does, and summarize() marks them. 0.85 sits well clear of the models that
# track (0.91-1.00 measured) and well above the ones that do not (0.75).
AGREEMENT_FLOOR = 0.85


# ---------------------------------------------------------------------------
# Global analysis parameters
# ---------------------------------------------------------------------------

@dataclass
class AnalysisParams:
    n_seeds:                  int   = 10
    headline_noise:           float = 0.20   # noise level used by F1/F2/F3/F5
    phases_to_include:        str   = 'Learning and inference'
    frames:                   str   = 'outcome'  # 'outcome' | 'cue' | 'all'
    last_ts_in_a_block:       int   = 15     # tail judgements per block for the correct-rate
    aggregate_blocks:  int | None   = 3      # None = one point per block
    skip_first_blocks:        int   = 0
    block_group_max:   int | None   = 40
    asymptotic_n_last_groups: int   = 3
    criterion_n:              int   = 3      # consecutive correct to declare the context acquired
    dpi:                      int   = 160
    show_plots:               bool  = True
    save_plots:               bool  = True
    # Which conditions get a line on the left-hand learning-curve panels. None = all of them.
    # The right-hand dose-response panels always show every condition — they are a scatter over
    # conditions, so they stay readable at 19; the time-course panels do not.
    curve_conditions: Sequence[str] | None = None
    linewidth:                float = 1.75
    # Line alpha. With the full 13-condition sweep the learning curves overlap heavily, and
    # partial transparency lets crossings show through instead of the last-drawn line winning.
    alpha:                    float = 0.70
    band_alpha:               float = 0.15   # SEM ribbons; 13 stacked ribbons grey out fast


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _cell_dir(noise_std: float, encoding: str | None = CONTEXT_OUTPUT_ENCODING) -> str:
    """Directory name the sweep's combo_key() produces for one (noise, encoding) cell."""
    return f"context_encoding-{encoding}_noise_std-{noise_std}"


def load_runs(condition_name: str, noise_std: float, n_seeds: int,
              encoding: str | None = CONTEXT_OUTPUT_ENCODING) -> List[Tuple[Any, Any]]:
    """Load (logger, config) pairs for all seeds of one condition x noise cell."""
    folder = EXPORT_ROOT / condition_name / _cell_dir(noise_std, encoding)
    runs, missing = [], []
    for seed in range(n_seeds):
        filepath = folder / f"results_seed-{seed}.pkl"
        if filepath.exists():
            with filepath.open("rb") as f:
                payload = pickle.load(f)
            runs.append((payload["train_logger"], payload["config"]))
        else:
            missing.append(seed)
    if missing:
        print(f"  [{condition_name} @ noise {noise_std}] missing seeds: {missing}")
    return runs


def cache_is_stale(cache, params: AnalysisParams,
                   noise_levels: Sequence[float] = tuple(NOISE_LEVELS)) -> bool:
    """True if the cache is missing runs that are now on disk, or conditions now in the config.

    Worth checking rather than assuming: in an interactive session the cache outlives both the
    sweep (jobs land after it was built) and the config (conditions get added to _Z_LRS), and a
    silently stale cache looks exactly like a sweep that did not finish.
    """
    if not cache:
        return True
    for noise in noise_levels:
        for cond in CONDITION_INFO:
            folder = EXPORT_ROOT / cond / _cell_dir(noise)
            on_disk = sum(1 for s in range(params.n_seeds)
                          if (folder / f"results_seed-{s}.pkl").exists())
            if on_disk > len(cache.get((cond, noise), [])):
                return True
    return False


def load_all_runs(params: AnalysisParams,
                  noise_levels: Sequence[float] = tuple(NOISE_LEVELS),
                  conditions: Sequence[str] | None = None) -> Dict[Tuple[str, float], List]:
    """{(condition, noise_std): [(logger, config), ...]} for every requested cell."""
    conditions = list(CONDITION_INFO) if conditions is None else list(conditions)
    print("Loading runs...")
    cache = {}
    for noise in noise_levels:
        for name in conditions:
            runs = load_runs(name, noise, params.n_seeds)
            if runs:
                cache[(name, noise)] = runs
    n_cells = len(cache)
    n_runs  = sum(len(v) for v in cache.values())
    print(f"  loaded {n_runs} runs across {n_cells} condition x noise cells")
    return cache


# ---------------------------------------------------------------------------
# Belief extraction
# ---------------------------------------------------------------------------

def _wrap(a):
    """Wrap angles to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def _circ_dist(a, b):
    """Absolute circular distance, radians."""
    return np.abs(_wrap(np.asarray(a) - np.asarray(b)))


def _nearest_slot(theta_rad, rotations_rad):
    """Index of the circularly nearest entry of rotations_rad, elementwise."""
    d = _circ_dist(np.asarray(theta_rad)[:, None], np.asarray(rotations_rad)[None, :])
    return np.argmin(d, axis=1)


def _phase_window(logger, config, phases_to_include) -> np.ndarray:
    """Boolean mask over flat timesteps selecting the requested phase(s)."""
    n = sum(len(np.asarray(e).reshape(-1)) for e in logger.context_ids)
    if phases_to_include is None:
        return np.ones(n, dtype=bool)
    include = ([phases_to_include] if isinstance(phases_to_include, str)
               else list(phases_to_include))
    mask = np.zeros(n, dtype=bool)
    for i, (name, ts) in enumerate(logger.phases):
        end = logger.phases[i + 1][1] if i + 1 < len(logger.phases) else n
        if name in include:
            mask[ts:end] = True
    return mask if mask.any() else np.ones(n, dtype=bool)


def extract_belief_trials(logger, config, params: AnalysisParams) -> Dict[str, np.ndarray]:
    """One row per scored judgement, restricted to the requested phase and frame type.

    Returns arrays of equal length:
        belief_rad   reported context belief, decoded from the network's context output
        belief_mag   ||context output|| / target_radius — the guard against a collapsed readout
        true_rad     ground-truth rotation of the frame being predicted (logger.context_ids)
        belief_slot  index of the circularly nearest trained rotation to belief_rad
        true_slot    same for true_rad
        correct      belief_slot == true_slot
        behav_rad    implicit belief read from the xy prediction (cue frames only, else NaN)
        block_id     contiguous block index within the selected phase
        t_index      flat timestep, for cross-referencing back into the logger
    """
    ii, oi, ll, _ = flatten_logger(logger, config)
    nc = config.n_colors
    C  = getattr(config, 'context_output_dims', 0)
    if not C:
        raise ValueError('This run has no context output head; nothing to read a belief from. '
                         'It is a no-head control — analyse its xy behaviour instead.')

    ctx_pred   = oi[:, nc:nc + C]
    belief_rad = np.arctan2(ctx_pred[:, 1], ctx_pred[:, 0])
    belief_mag = np.linalg.norm(ctx_pred, axis=1) / config.target_radius

    # Implicit readout, free from the same arrays. oi[t] is the prediction *of* frame ii[t], so
    # the predicted attack lives at the outcome frame; the colour that cued it is on the
    # preceding cue frame. The attack's polar angle is (2*pi*colour/n_colors + rotation), giving
    # a second, purely behavioural estimate of the same quantity — and it lands on the outcome
    # frames, which is where judgements are scored by default.
    is_cue = ii[:, :nc].sum(axis=1) > 0.5
    behav_rad = np.full(len(ll), np.nan)
    cue_idx = np.flatnonzero(is_cue[:-1])
    colour  = np.argmax(ii[cue_idx, :nc], axis=1)
    pred_xy = oi[cue_idx + 1, -2:]
    behav_rad[cue_idx + 1] = _wrap(np.arctan2(pred_xy[:, 1], pred_xy[:, 0])
                                   - 2 * np.pi * colour / nc)

    phase_mask = _phase_window(logger, config, params.phases_to_include)
    if params.frames == 'cue':
        frame_mask = is_cue
    elif params.frames == 'outcome':
        frame_mask = ~is_cue
    elif params.frames == 'all':
        frame_mask = np.ones(len(ll), dtype=bool)
    else:
        raise ValueError(f"frames must be 'all', 'cue' or 'outcome'; got {params.frames!r}")

    keep = phase_mask & frame_mask
    # Blocks are defined on the phase window (before the frame filter) so that a switch is not
    # missed when it lands on a frame type we are not scoring.
    ll_phase = ll[phase_mask]
    block_of_phase = np.concatenate([[0], np.cumsum(np.diff(ll_phase) != 0)])
    block_id_full = np.full(len(ll), -1)
    block_id_full[phase_mask] = block_of_phase

    rots_rad    = np.deg2rad(np.asarray(config.train_rotations, dtype=float))
    belief_slot = _nearest_slot(belief_rad[keep], rots_rad)
    true_slot   = _nearest_slot(ll[keep],         rots_rad)

    # Each block's rotation and the one before it, so every trial knows which context it is
    # supposed to be in and which one it might be perseverating on.
    n_blocks    = int(block_of_phase.max()) + 1
    rot_of_blk  = np.array([np.median(ll_phase[block_of_phase == b]) for b in range(n_blocks)])
    prev_of_blk = np.concatenate([[np.nan], rot_of_blk[:-1]])

    bid      = block_id_full[keep]
    prev_rot = prev_of_blk[bid]
    true_k   = ll[keep]

    # Belief on the repo's standard normalized-state-error scale (the same convention as
    # rotating_targets_analysis._analyze_adaptation's trial_norm_errors):
    #     0.0 = on the current context, 0.5 = midway between the two, 1.0 = on the previous one.
    # This is one scalar for both "which context" and "how committed", it needs no circular
    # statistics, and 0.5 is the hedge point by construction rather than by derivation.
    r   = config.target_radius
    vec = lambda a: r * np.stack([np.cos(a), np.sin(a)], axis=1)
    bel_v, cur_v, alt_v = vec(belief_rad[keep]), vec(true_k), vec(prev_rot)
    d_cur = np.linalg.norm(bel_v - cur_v, axis=1)
    d_alt = np.linalg.norm(bel_v - alt_v, axis=1)
    belief_norm = np.where(np.isnan(prev_rot), np.nan, d_cur / (d_cur + d_alt + 1e-12))

    # Ordinal position of each scored trial inside its block, for within-block time courses.
    pos_in_block = np.concatenate([np.arange(int((bid == b).sum()))
                                   for b in np.unique(bid)]) if len(bid) else np.array([])

    return dict(
        belief_rad   = belief_rad[keep],
        belief_mag   = belief_mag[keep],
        belief_norm  = belief_norm,
        true_rad     = true_k,
        prev_rot     = prev_rot,
        belief_slot  = belief_slot,
        true_slot    = true_slot,
        correct      = belief_slot == true_slot,
        behav_rad    = behav_rad[keep],
        block_id     = bid,
        pos_in_block = pos_in_block,
        t_index      = np.flatnonzero(keep),
    )


# ---------------------------------------------------------------------------
# Per-block criterion metrics
# ---------------------------------------------------------------------------

_BLOCK_FIELDS = ('persev', 'slips', 'slip_rate', 'ttc', 'no_criterion', 'correct_tail',
                 'belief_mag', 'belief_norm', 'agreement')


def block_criterion_metrics(trials: Dict[str, np.ndarray], params: AnalysisParams,
                            ) -> Dict[str, np.ndarray]:
    """Per-block perseveration, slips and belief statistics.

    Same definitions as mean_prediction_analysis.extract_block_criterion_metrics, on the
    rotating-targets belief:
      perseveration errors — wrong judgements before the criterion run of criterion_n
        consecutive correct. If criterion is never reached, every error counts as perseverative.
      context slips        — wrong judgements after criterion, in the same block. NaN when
        criterion was never reached (the model never adapted, so there is no "after").
      slip rate            — slips / post-criterion judgements. Block lengths are geometric
        here, so raw counts are not comparable across blocks; this is the headline metric.

    The first block of the phase is skipped: it follows no switch, so there is nothing to
    perseverate from.

    Returns arrays of shape (n_blocks - 1,), one entry per block.
    """
    out = {k: [] for k in _BLOCK_FIELDS}
    block_ids = np.unique(trials['block_id'])
    block_ids = block_ids[block_ids >= 0]

    prev_rot = None
    for bi in block_ids:
        sel = trials['block_id'] == bi
        curr_rot = float(np.median(trials['true_rad'][sel])) if sel.any() else None
        if bi == block_ids[0]:
            prev_rot = curr_rot
            continue
        correct = trials['correct'][sel]
        if len(correct) == 0:
            continue

        t_crit = _find_criterion(correct, params.criterion_n)
        if t_crit is not None:
            n_post = len(correct) - t_crit
            slips  = float(np.sum(~correct[t_crit:]))
            out['persev'].append(float(np.sum(~correct[:t_crit])))
            out['slips'].append(slips)
            out['slip_rate'].append(slips / n_post if n_post > 0 else np.nan)
            out['ttc'].append(float(t_crit))
            out['no_criterion'].append(0.0)
        else:
            out['persev'].append(float(np.sum(~correct)))
            out['slips'].append(np.nan)
            out['slip_rate'].append(np.nan)
            out['ttc'].append(np.nan)
            # Read this alongside slip_rate. Blocks that never adapt drop out of the slip
            # average by definition (there is no "after criterion"), so a model failing more
            # often can show a *falling* slip rate while getting worse overall — the surviving
            # blocks are its easy ones. This fraction is what makes that visible.
            # Guard, reported in summarize(): a block that never adapts drops out of the slip
            # average entirely (there is no "after criterion"). Measured at 0.000 everywhere on
            # the pilot, so it is not driving any result — but if it ever rises, the slip rate
            # for that cell is computed over that model's easy blocks only.
            out['no_criterion'].append(1.0)

        tail = correct[-params.last_ts_in_a_block:]
        out['correct_tail'].append(float(tail.mean()))
        out['belief_mag'].append(float(np.mean(trials['belief_mag'][sel])))

        # Where the settled belief sits on the 0=current / 0.5=hedge / 1=previous scale, after
        # criterion. Values above 0 mean the belief is still pulled toward the context just
        # left; 0.5 means it is committed to neither.
        post = slice(t_crit, None) if t_crit is not None else slice(0, 0)
        bn = trials['belief_norm'][sel][post]
        out['belief_norm'].append(float(np.nanmean(bn)) if len(bn) else np.nan)

        # Belief-vs-behaviour agreement: do the reported belief and the xy prediction imply the
        # same context? Only defined where behav_rad exists (cue frames).
        bh = trials['behav_rad'][sel]
        ok = ~np.isnan(bh)
        if ok.any():
            rots_rad = np.deg2rad(np.asarray(TRAIN_ROTATIONS, dtype=float))
            agree = (_nearest_slot(bh[ok], rots_rad)
                     == _nearest_slot(trials['belief_rad'][sel][ok], rots_rad))
            out['agreement'].append(float(agree.mean()))
        else:
            out['agreement'].append(np.nan)

        prev_rot = curr_rot

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def metrics_for_runs(runs: Sequence[Tuple[Any, Any]], params: AnalysisParams,
                     ) -> Dict[str, List[np.ndarray]]:
    """block_criterion_metrics for every seed; {field: [per-seed array, ...]}."""
    per_seed = {k: [] for k in _BLOCK_FIELDS}
    for logger, config in runs:
        trials = extract_belief_trials(logger, config, params)
        m = block_criterion_metrics(trials, params)
        for k in _BLOCK_FIELDS:
            if len(m[k]) > 0:
                per_seed[k].append(m[k])
    return per_seed


def _curve(per_seed_arrays, params: AnalysisParams):
    """Seed arrays -> (x, mean, sem) block-group curve, NaN-safe."""
    return _aggregate_per_block_curves(
        per_seed_arrays, params.aggregate_blocks,
        params.skip_first_blocks, params.block_group_max,
    )


def _asymptote(mean_c, sem_c, params: AnalysisParams) -> Tuple[float, float]:
    """Tail summary: the mean of the last asymptotic_n_last_groups block-groups."""
    n_last = min(params.asymptotic_n_last_groups, len(mean_c))
    return float(np.nanmean(mean_c[-n_last:])), float(np.nanmean(sem_c[-n_last:]))


def _whole_curve(mean_c, sem_c, params: AnalysisParams) -> Tuple[float, float]:
    """Mean over every block-group, i.e. over the whole of training.

    A tail summary answers "where did it end up"; this answers "how did it do overall", which
    for a per-block error count is the more stable and more interpretable number — it does not
    hinge on how many groups happen to be called the asymptote.
    """
    return float(np.nanmean(mean_c)), float(np.nanmean(sem_c))


def p_context_error_single_obs(noise_std: float, target_radius: float = 0.5,
                               rotations: Sequence[float] = tuple(TRAIN_ROTATIONS)) -> float:
    """Error rate of a memoryless observer that classifies context from ONE observation.

    Two candidate targets for the cued colour, separated by the chord
    D = 2*target_radius*sin(sep/2), each observed under isotropic Gaussian noise of std
    noise_std. The optimal classifier is the perpendicular bisector, so the error rate is
    Phi(-D / (2*noise_std)).

    Use the chord, not the arc: the observation lives in the plane, and the arc length
    (sep/2)*target_radius overstates the separation — at sep=120 deg it understates the error
    by a factor of ~3.
    """
    from math import erf, sqrt
    sep = np.deg2rad(np.ptp(np.asarray(rotations, dtype=float)))
    D   = 2 * target_radius * np.sin(sep / 2)
    z   = -D / (2 * noise_std)
    return float(0.5 * (1 + erf(z / sqrt(2))))


# ---------------------------------------------------------------------------
# Ideal observer
# ---------------------------------------------------------------------------

def ideal_observer_trials(logger, config, params: AnalysisParams) -> Dict[str, np.ndarray]:
    """Bayes-optimal context tracking on the attacks the model actually saw.

    A discrete filter over the trained rotations. On each trial the cued colour and the observed
    attack give a likelihood over rotations; a hazard-rate transition step lets the belief switch
    blocks. Reconstructed from the logged inputs, so the observer sees exactly the model's data.

    This is the normative reference: its perseveration is pure detection delay (you cannot know
    the context changed before evidence arrives), and its slips are the irreducible rate at this
    noise level. Returned in the same shape as extract_belief_trials so the identical
    block_criterion_metrics runs on it.
    """
    ii, _, ll, _ = flatten_logger(logger, config)
    nc    = config.n_colors
    sigma = config.noise_std
    rots  = np.asarray(config.train_rotations, dtype=float)
    K     = len(rots)
    targets = np.stack([get_target_positions(config, deg) for deg in rots])  # (K, nc, 2)

    phase_mask = _phase_window(logger, config, params.phases_to_include)
    is_cue = ii[:, :nc].sum(axis=1) > 0.5
    # One update per trial, at the cue frame whose outcome frame carries the attack.
    idx = np.flatnonzero(phase_mask & is_cue)
    idx = idx[idx + 1 < len(ll)]

    ll_phase = ll[phase_mask]
    block_of_phase = np.concatenate([[0], np.cumsum(np.diff(ll_phase) != 0)])
    block_id_full = np.full(len(ll), -1)
    block_id_full[phase_mask] = block_of_phase

    # Hazard from the realised block structure: switches per trial in this phase.
    n_trials = max(1, len(idx))
    n_blocks = max(1, len(np.unique(block_of_phase)))
    hazard   = min(0.5, n_blocks / n_trials)

    log_post = np.full(K, -np.log(K))
    slots, beliefs = [], []
    for t in idx:
        colour = int(np.argmax(ii[t, :nc]))
        attack = ii[t + 1, -2:]
        # transition step
        post = np.exp(log_post - log_post.max())
        post /= post.sum()
        post = (1 - hazard) * post + hazard * (1 - post) / max(1, K - 1)
        # Score the *predictive* belief — the MAP before this trial's attack is seen. The network
        # is a predictor (its belief about frame t is formed from data through t-1), so a filter
        # that had already seen trial t's evidence would be getting a free observation. This is
        # what makes the observer's perseveration a real detection-delay floor rather than 0.
        slot = int(np.argmax(post))
        slots.append(slot)
        beliefs.append(np.deg2rad(rots[slot]))
        # observation step
        d2 = np.sum((attack[None, :] - targets[:, colour, :]) ** 2, axis=1)
        log_post = np.log(post + 1e-300) - d2 / (2 * sigma ** 2)

    slots     = np.asarray(slots)
    beliefs   = np.asarray(beliefs)
    rots_rad  = np.deg2rad(rots)
    true_slot = _nearest_slot(ll[idx], rots_rad)

    bid = block_id_full[idx]
    n_blocks_p  = int(block_of_phase.max()) + 1
    rot_of_blk  = np.array([np.median(ll_phase[block_of_phase == b]) for b in range(n_blocks_p)])
    prev_of_blk = np.concatenate([[np.nan], rot_of_blk[:-1]])
    prev_rot    = prev_of_blk[bid]

    # The observer's belief is always exactly one of the trained rotations, so on the
    # 0 = current / 1 = previous scale it is binary — but computing it the same way as the
    # models keeps block_criterion_metrics a single code path.
    r   = config.target_radius
    vec = lambda a: r * np.stack([np.cos(a), np.sin(a)], axis=1)
    bel_v, cur_v, alt_v = vec(beliefs), vec(ll[idx]), vec(prev_rot)
    d_cur = np.linalg.norm(bel_v - cur_v, axis=1)
    d_alt = np.linalg.norm(bel_v - alt_v, axis=1)

    return dict(
        belief_rad   = beliefs,
        belief_mag   = np.ones(len(idx)),
        belief_norm  = np.where(np.isnan(prev_rot), np.nan, d_cur / (d_cur + d_alt + 1e-12)),
        true_rad     = ll[idx],
        prev_rot     = prev_rot,
        belief_slot  = slots,
        true_slot    = true_slot,
        correct      = slots == true_slot,
        behav_rad    = np.full(len(idx), np.nan),
        block_id     = bid,
        pos_in_block = np.concatenate([np.arange(int((bid == b).sum()))
                                       for b in np.unique(bid)]) if len(bid) else np.array([]),
        t_index      = idx,
    )


def ideal_observer_metrics(runs, params: AnalysisParams) -> Dict[str, List[np.ndarray]]:
    per_seed = {k: [] for k in _BLOCK_FIELDS}
    for logger, config in runs:
        m = block_criterion_metrics(ideal_observer_trials(logger, config, params), params)
        for k in _BLOCK_FIELDS:
            if len(m[k]) > 0:
                per_seed[k].append(m[k])
    return per_seed


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _info(name: str) -> ConditionInfo:
    """Label and colour as registered — NeuraGEM conditions on the plasma alpha_z ramp.

    For the sweep panels, where the ramp is the point.
    """
    return CONDITION_INFO.get(name, ConditionInfo(label=name, color='grey'))


def _solo_info(name: str) -> ConditionInfo:
    """Same label, but a lone NeuraGEM condition reverts to NeuraGEM's project colour.

    The plasma ramp exists to encode alpha_z. A figure showing one representative NG condition
    has no ramp to read, so the ramp colour carries no information — and at HEADLINE_Z_LR it
    lands on a salmon that reads as F1's wrong-context red and as the Oracle's orange in F4/F5.
    Outside the sweep panels NG is NeuraGEM blue, which is also what it is everywhere else in
    the project (plot_style.MODEL_COLORS).
    """
    info = _info(name)
    if name.startswith('NG'):
        return ConditionInfo(label=info.label, color=plot_style.get_model_color('NeuraGEM'))
    return info


def _dose_response_panel(ax, points, ylabel: str, params: AnalysisParams,
                         gap: float = 2.2, label_every: int = 2) -> None:
    """Summary-per-condition panel: a reference model, a gap, then the swept NeuraGEM family.

    The RNN is a different architecture, not the alpha_z -> 0 end of the sweep, so joining it to
    the family with a line would assert a continuum that does not exist. It gets its own slot,
    its own colour (from plot_style's registry, outside the viridis ramp), and no connector.

    Family ticks are the bare alpha_z values with the name in the axis label — spelling out
    "NG $\\alpha_z=0.15$" 15 times is what makes this panel unreadable at paper size.

    NOTE the family axis is *categorical*: conditions are evenly spaced in plot order, not
    positioned by alpha_z. It has to be, because _Z_LRS is denser at the top (0.5-0.9 spans only
    13% of the log range but holds a third of the points) and a true log axis piles those five
    on top of each other. The cost is that equal widths do not mean equal ratios — read the tick
    values, not the spacing.
    """
    if not points:
        return
    fam = [p for p in points if p[0].startswith('NG')]
    ref = [p for p in points if not p[0].startswith('NG')]

    xs_ref = list(np.arange(len(ref)))
    xs_fam = list(np.arange(len(fam)) + len(ref) - 1 + gap)

    # Connector through the swept family only.
    if len(fam) > 1:
        ax.plot(xs_fam, [p[1] for p in fam], color='0.7', linewidth=0.8, zorder=1)

    for xs, pts in ((xs_ref, ref), (xs_fam, fam)):
        for x, (cond, mean_val, sem_val) in zip(xs, pts):
            ax.errorbar(x, mean_val, yerr=sem_val, fmt='o', color=_info(cond).color,
                        capsize=1.5, elinewidth=0.8, markersize=3.2,
                        markeredgewidth=0, zorder=3)

    ticks  = list(xs_ref) + list(xs_fam[::label_every])
    labels = [_info(c).label for c, _, _ in ref]
    # ".01" rather than "0.01": at paper size the leading zero is the difference between the
    # low-end labels touching and not, and dropping it for values below 1 is conventional.
    labels += [c.split('=')[-1].rstrip('$').lstrip('0') for c, _, _ in fam[::label_every]]
    ax.set_xticks(ticks)
    # Bare numbers are short enough to sit horizontally, which buys back the vertical space a
    # rotated tick block costs — the reason the family label lives in the axis label.
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlim(-0.9, xs_fam[-1] + 0.9)
    ax.set_ylabel(f'{ylabel}\n(mean over training)')
    ax.set_xlabel(r'NeuraGEM  $\alpha_z$', labelpad=1)


def _draws_curve(cond: str, params: AnalysisParams) -> bool:
    """Whether this condition gets a line on a time-course panel. See curve_conditions."""
    return params.curve_conditions is None or cond in params.curve_conditions


def _warn_if_crowded(n_drawn: int, params: AnalysisParams) -> None:
    if params.curve_conditions is None and n_drawn > 8:
        print(f'  note: {n_drawn} curves on one panel. Set AnalysisParams.curve_conditions '
              f'(e.g. HEADLINE_CONDITIONS) to thin the time-course panels; the dose-response '
              f'panels keep all conditions either way.')


def _save(fig, export_dir: Path, name: str, params: AnalysisParams):
    if params.save_plots:
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / name
        fig.savefig(out, bbox_inches='tight')
        print(f"  Saved → {out}")
    if params.show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_belief_trajectory(cache, params: AnalysisParams, export_dir: Path,
                           n_blocks: int = 8, seed_idx: int = 0) -> plt.Figure:
    """F1 — the reported belief, block by block, late in training.

    One panel per model is the exception the style guide allows: these are per-trial point
    clouds, which cannot be overlaid on shared axes without becoming unreadable.
    """
    conds = [c for c in HEADLINE_CONDITIONS if (c, params.headline_noise) in cache]
    fig, axes = plt.subplots(1, len(conds), figsize=FigSize.row(len(conds), FigSize.small),
                             dpi=params.dpi, sharey=True, layout='constrained')
    axes = np.atleast_1d(axes)

    rots = np.asarray(TRAIN_ROTATIONS, dtype=float)
    boundary = float(np.mean(rots))       # 2 contexts: the midpoint is the decision boundary

    for ax, cond in zip(axes, conds):
        runs   = cache[(cond, params.headline_noise)]
        logger, config = runs[min(seed_idx, len(runs) - 1)]
        trials = extract_belief_trials(logger, config, params)

        blocks = np.unique(trials['block_id'])
        blocks = blocks[blocks >= 0][-n_blocks:]
        sel    = np.isin(trials['block_id'], blocks)
        x      = np.arange(int(sel.sum()))
        belief = np.degrees(_wrap(trials['belief_rad'][sel]))
        true   = np.degrees(_wrap(trials['true_rad'][sel]))
        ok     = trials['correct'][sel]

        ax.step(x, true, where='post', color='0.75', linewidth=1.0, zorder=1)
        ax.scatter(x[ok],  belief[ok],  s=1.5, color=_solo_info(cond).color, alpha=0.7,
                   linewidths=0, zorder=3)
        ax.scatter(x[~ok], belief[~ok], s=3.0, color='tab:red', alpha=0.9,
                   linewidths=0, zorder=4)
        ax.axhline(boundary, color='k', linewidth=0.6, linestyle=':', alpha=0.4, zorder=2)
        for r in rots:
            ax.axhline(r, color='0.6', linewidth=0.5, linestyle='-', alpha=0.4, zorder=0)
        ax.set_xlabel('Trial')
        ax.set_title(_solo_info(cond).label)

    axes[0].set_ylabel('Reported belief (deg)')
    axes[0].set_ylim(rots.min() - 45, rots.max() + 45)
    _save(fig, export_dir, 'F1_belief_trajectory.pdf', params)
    return fig


def plot_context_correct(cache, params: AnalysisParams, export_dir: Path) -> plt.Figure:
    """F2 — context-correct rate over training, plus the asymptotic summary."""
    pw, ph = FigSize.wide
    sw = FigSize.small[0]
    fig = plt.figure(figsize=(pw + sw, ph), dpi=params.dpi, layout='constrained')
    gs = gridspec.GridSpec(1, 2, width_ratios=[pw, sw], figure=fig)
    ax_main, ax_asy = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    asym = []
    for cond in CONDITION_INFO:
        runs = cache.get((cond, params.headline_noise), [])
        if not runs:
            continue
        per_seed = metrics_for_runs(runs, params)
        x, mean_c, sem_c = _curve(per_seed['correct_tail'], params)
        if x is None:
            continue
        info = _info(cond)
        asym.append((cond, *_asymptote(mean_c, sem_c, params)))
        if not _draws_curve(cond, params):
            continue
        ax_main.plot(x, mean_c, color=info.color, linewidth=params.linewidth,
                     alpha=params.alpha, label=info.label)
        ax_main.fill_between(x, mean_c - sem_c, mean_c + sem_c,
                             color=info.color, alpha=params.band_alpha, linewidth=0)

    chance = 1.0 / len(TRAIN_ROTATIONS)
    ax_main.axhline(chance, color='k', linewidth=0.6, linestyle=':', alpha=0.3)
    ax_main.set_xlabel(f'Block group (x{params.aggregate_blocks})')
    ax_main.set_ylabel(f'Context-correct\n(tail {params.last_ts_in_a_block} trials)')
    ax_main.set_ylim(chance - 0.12, 1.03)

    _asymptote_panel(ax_asy, asym, 'Context-correct',
                     params.asymptotic_n_last_groups, condition_info=CONDITION_INFO)
    ax_asy.axhline(chance, color='k', linewidth=0.6, linestyle=':', alpha=0.3)

    handles, labels = ax_main.get_legend_handles_labels()
    handles, labels = _prune_legend_handles(handles, labels)
    fig.legend(handles, labels, loc='outside upper center', ncols=len(handles),
               fontsize='small', frameon=False)

    _save(fig, export_dir, 'F2_context_correct.pdf', params)
    return fig


def plot_perseveration_and_slips(cache, params: AnalysisParams, export_dir: Path,
                                 width_scale: float = 1.0) -> plt.Figure:
    """F3 - perseveration errors and context slips per block, over training and summarised.

    Left: time courses for a readable subset (F3_CURVE_CONDITIONS).
    Right: every condition in F3_CONDITIONS, summarised as the mean of its whole curve.

    F3_CONDITIONS omits the fast-RNN controls and the Oracle; see the comment on it in the
    config for why, and state both in the caption rather than plotting them.

    width_scale widens the figure if the family outgrows the panel; the defaults are tuned for
    ~15 alpha_z values at paper size.
    """
    pw, ph = FigSize.wide
    rw = pw * 1.1                       # the summary panel carries ~16 x positions, so it needs
                                        # more width than the FigSize.small slot it used to get
    fig = plt.figure(figsize=((pw + rw) * width_scale, ph * 2),
                     dpi=params.dpi, layout='constrained')
    gs = gridspec.GridSpec(2, 2, width_ratios=[pw, rw], figure=fig)
    ax_p, ax_s   = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_ap, ax_as = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    curve_conds = params.curve_conditions or F3_CURVE_CONDITIONS
    summ_p, summ_s = [], []

    for cond in F3_CONDITIONS:
        runs = cache.get((cond, params.headline_noise), [])
        if not runs:
            continue
        per_seed = metrics_for_runs(runs, params)
        info = _info(cond)
        for ax, field, store in ((ax_p, 'persev', summ_p), (ax_s, 'slips', summ_s)):
            x, mean_c, sem_c = _curve(per_seed[field], params)
            if x is None:
                continue
            store.append((cond, *_whole_curve(mean_c, sem_c, params)))
            if cond not in curve_conds:
                continue
            ax.plot(x, mean_c, color=info.color, linewidth=params.linewidth,
                    alpha=params.alpha, label=info.label)
            ax.fill_between(x, mean_c - sem_c, mean_c + sem_c,
                            color=info.color, alpha=params.band_alpha, linewidth=0)

    x_label = f'Block group (x{params.aggregate_blocks})'
    ax_p.set_ylabel('Perseveration errors / block')
    ax_p.set_xlabel(x_label)
    ax_s.set_ylabel('Context slips / block')
    ax_s.set_xlabel(x_label)

    _dose_response_panel(ax_ap, summ_p, 'Perseveration errors', params)
    _dose_response_panel(ax_as, summ_s, 'Context slips', params)

    handles, labels = ax_p.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncols=min(len(handles), 4),
               fontsize='small', frameon=False, handlelength=1.2,
               columnspacing=1.0, handletextpad=0.4)

    _save(fig, export_dir, 'F3_perseveration_and_slips.pdf', params)
    return fig


def plot_slips_vs_noise(cache, params: AnalysisParams, export_dir: Path,
                        conditions: Sequence[str] | None = None) -> plt.Figure:
    """F4 — asymptotic slip rate against observation noise, with the ideal observer.

    The causal figure: if slips come from noise-driven context confusion, a model that holds a
    latent state should stay flat where one that does not should climb with noise.
    """
    conds = list(HEADLINE_CONDITIONS if conditions is None else conditions)
    pw, ph = FigSize.wide
    fig, (ax, ax_nc) = plt.subplots(
        2, 1, figsize=(pw, ph * 1.55), dpi=params.dpi, sharex=True,
        height_ratios=[2.2, 1], layout='constrained')

    series = [(c, _solo_info(c).label, _solo_info(c).color) for c in conds]
    series.append(('__ideal__', 'Ideal observer', '0.35'))

    for cond, label, color in series:
        xs, ys, es, ncs = [], [], [], []
        for noise in NOISE_LEVELS:
            # The observer reads the attack stream, which depends only on env_seed and
            # noise_std — identical across conditions — so any condition's runs will do.
            src = cond if cond != '__ideal__' else conds[0]
            runs = cache.get((src, noise), [])
            if not runs:
                continue
            per_seed = (ideal_observer_metrics(runs, params) if cond == '__ideal__'
                        else metrics_for_runs(runs, params))
            x, mean_c, sem_c = _curve(per_seed['slips'], params)
            xt, mean_t, sem_t = _curve(per_seed['ttc'], params)
            if x is None:
                continue
            m, e = _asymptote(mean_c, sem_c, params)
            xs.append(noise); ys.append(m); es.append(e)
            ncs.append(np.nan if xt is None else _asymptote(mean_t, sem_t, params)[0])
        if not xs:
            continue
        style = dict(linestyle='--') if cond == '__ideal__' else {}
        ax.errorbar(xs, ys, yerr=es, marker='o', markersize=3.5, capsize=2,
                    color=color, linewidth=params.linewidth, alpha=params.alpha,
                    label=label, **style)
        ax_nc.plot(xs, ncs, marker='o', markersize=3.0, color=color,
                   linewidth=params.linewidth, alpha=params.alpha, **style)

    # No memoryless reference here: it is an error *rate*, and converting it to slips-per-block
    # needs each model's own post-criterion window length, which makes it a different number per
    # curve rather than a shared reference. p_context_error_single_obs stays available for the
    # design table in the docs. The ideal observer is already in slips/block, so it transfers.
    ax.set_ylabel('Asymptotic\ncontext slips / block')
    ax.legend(frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.2,
              fontsize='small')
    # Read the slip rate against this. Perseveration and slips partition a block's errors at the
    # criterion, so as trials-to-criterion grows more of the block sits *before* it and errors
    # are booked as perseveration instead. That is why a model can show a falling slip rate
    # while getting monotonically worse — the perseveration panel of F3 is the monotone signal.
    ax_nc.set_ylabel('Trials to\ncriterion')
    ax_nc.set_xlabel('Observation noise std')
    ax_nc.set_ylim(0, None)

    _save(fig, export_dir, 'F4_slips_vs_noise.pdf', params)
    return fig


def plot_belief_dynamics(cache, params: AnalysisParams, export_dir: Path,
                         pos_max: int = 80) -> plt.Figure:
    """F5 - the belief through a block, on the 0 = current / 0.5 = hedge / 1 = previous scale.

    This single panel carries the whole within-block story: how far the belief starts from the
    new context (trial 0), how fast it converges, and where it ends up.

    It is also the panel that keeps the interpretation honest. Two earlier framings did not
    survive it:
      - "the RNN settles at a shifted decision boundary" - it does not settle anywhere biased;
        the curve decays monotonically and is simply still in transit when the block ends.
      - "the RNN carries a graded estimate while NG has discrete states" - the RNN's belief
        distribution looks smeared only because it is sampled mid-transit. Past trial ~60 the
        two distributions are nearly identical (mean 0.127 vs 0.119 at noise 0.30).
    What is left is a single mechanism: the RNN updates context ~10x more slowly, so it spends
    most of every block en route, which is when noise can knock it across the midpoint.
    """
    fig, ax = plt.subplots(figsize=FigSize.wide, dpi=params.dpi, layout='constrained')

    for cond in HEADLINE_CONDITIONS:
        runs = cache.get((cond, params.headline_noise), [])
        if not runs:
            continue
        info = _solo_info(cond)
        curves = []
        for logger, config in runs:
            tr = extract_belief_trials(logger, config, params)
            pos, bn = tr['pos_in_block'], tr['belief_norm']
            prof = np.full(pos_max, np.nan)
            for p in range(pos_max):
                m = (pos == p) & ~np.isnan(bn)
                if m.sum() >= 5:
                    prof[p] = np.nanmean(bn[m])
            curves.append(prof)
        if not curves:
            continue
        arr = np.stack(curves)
        m   = np.nanmean(arr, axis=0)
        e   = (np.nanstd(arr, axis=0, ddof=1) / np.sqrt(len(curves))
               if len(curves) > 1 else np.zeros_like(m))
        x = np.arange(pos_max)
        ax.plot(x, m, color=info.color, linewidth=params.linewidth,
                alpha=params.alpha, label=info.label)
        ax.fill_between(x, m - e, m + e, color=info.color, alpha=params.band_alpha, linewidth=0)

    ax.axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.35)
    ax.annotate('hedge', xy=(0.98, 0.5), xycoords=('axes fraction', 'data'),
                va='bottom', ha='right', color='0.35', fontsize='small')
    ax.set_xlabel('Trial within block')
    ax.set_ylabel('Belief\n(0 = current, 1 = previous)')
    ax.set_ylim(-0.03, 1.0)
    ax.legend(frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.25)

    _save(fig, export_dir, 'F5_belief_dynamics.pdf', params)
    return fig


def plot_diagnostics(cache, params: AnalysisParams, export_dir: Path) -> plt.Figure:
    """D - belief-vs-behaviour agreement: does the reported context match the predictions?

    The belief head and the xy predictions are two independent readouts of the same run. If a
    model reports one context while its actual predictions sit at another, the readout has come
    loose from the behaviour - which is itself a result, not a measurement problem.

    Readout magnitude is checked in summarize() rather than plotted: it is a validity guard
    (a belief vector collapsing to the origin would make its angle meaningless), and on a scale
    nobody reads as uncertainty. The 0/0.5/1 belief scale in F5 carries that information in a
    form that does.
    """
    fig, ax = plt.subplots(figsize=FigSize.wide, dpi=params.dpi, layout='constrained')

    for cond in HEADLINE_CONDITIONS:
        runs = cache.get((cond, params.headline_noise), [])
        if not runs:
            continue
        per_seed = metrics_for_runs(runs, params)
        info = _solo_info(cond)
        x, mean_c, sem_c = _curve(per_seed['agreement'], params)
        if x is None:
            continue
        ax.plot(x, mean_c, color=info.color, linewidth=params.linewidth,
                alpha=params.alpha, label=info.label)
        ax.fill_between(x, mean_c - sem_c, mean_c + sem_c,
                        color=info.color, alpha=params.band_alpha, linewidth=0)

    ax.set_ylabel('Belief-behaviour agreement')
    ax.set_xlabel(f'Block group (x{params.aggregate_blocks})')
    ax.legend(frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.25)

    _save(fig, export_dir, 'D_belief_behaviour_agreement.pdf', params)
    return fig


# ---------------------------------------------------------------------------
# Control: does the belief head change the primary task?
# ---------------------------------------------------------------------------

def compare_head_vs_no_head(params: AnalysisParams,
                            conditions: Sequence[str] | None = None,
                            noise_levels: Sequence[float] | None = None,
                            n_miniblocks: int = 6) -> None:
    """Print the xy adaptation curve with and without the context-belief output.

    The head adds a supervised gradient that shapes the hidden state toward encoding rotation.
    If that changed the primary task, the perseveration comparison would be about a different
    experiment in each arm. This scores both arms with the *existing* rotating-targets metric —
    analyze_block_switch_adaptation's normalized state error, 0=correct, 0.5=chance,
    1=predicting the other rotation's target — which reads only the xy dims and so is directly
    comparable across arms.

    Requires the sweep to have been run with PILOT_INCLUDE_NO_HEAD_CONTROL.
    """
    from rotating_targets_analysis import analyze_block_switch_adaptation

    conds  = list(HEADLINE_CONDITIONS if conditions is None else conditions)
    noises = list(NOISE_LEVELS if noise_levels is None else noise_levels)

    print('\n── Control: xy behaviour with vs without the belief head ──')
    print('   normalized state error per trial position after a switch '
          '(0=correct, 0.5=chance, 1=other rotation)')
    print(f"{'condition':<22}{'noise':>7}{'head':>7}" +
          ''.join(f'{f"t{p+1}":>8}' for p in range(5)))

    for cond in conds:
        for noise in noises:
            for encoding, tag in ((CONTEXT_OUTPUT_ENCODING, 'on'), (None, 'off')):
                runs = load_runs(cond, noise, params.n_seeds, encoding=encoding)
                if not runs:
                    continue
                curves = []
                for logger, config in runs:
                    res = analyze_block_switch_adaptation(
                        logger, config, n_miniblocks_to_track=n_miniblocks)
                    if res['n_switches'] > 0:
                        curves.append(np.nanmean(res['trial_norm_errors'], axis=0))
                if not curves:
                    continue
                m = np.nanmean(np.stack(curves), axis=0)
                print(f'{_info(cond).label:<22}{noise:>7}{tag:>7}' +
                      ''.join(f'{v:>8.3f}' for v in m[:5]))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(cache, params: AnalysisParams) -> None:
    """Printed table of the asymptotic metrics at the headline noise level."""
    print(f"\n── Asymptotic metrics @ noise_std={params.headline_noise} "
          f"(last {params.asymptotic_n_last_groups} block groups) ──")
    print(f"{'condition':<22}{'correct':>9}{'persev':>9}{'slips/blk':>11}"
          f"{'belief':>9}{'no crit':>9}{'TTC':>8}{'|belief|':>10}{'agree':>8}")
    flagged: List[str] = []
    rows = [(c, cache.get((c, params.headline_noise), [])) for c in CONDITION_INFO]
    rows.append(('Ideal observer', cache.get((HEADLINE_CONDITIONS[0], params.headline_noise), [])))
    for cond, runs in rows:
        if not runs:
            continue
        per_seed = (ideal_observer_metrics(runs, params) if cond == 'Ideal observer'
                    else metrics_for_runs(runs, params))
        vals = []
        for field in ('correct_tail', 'persev', 'slips', 'belief_norm', 'no_criterion', 'ttc',
                      'belief_mag', 'agreement'):
            x, mean_c, sem_c = _curve(per_seed[field], params)
            vals.append(np.nan if x is None else _asymptote(mean_c, sem_c, params)[0])
        label = _info(cond).label if cond in CONDITION_INFO else cond
        # A condition whose reported belief disagrees with its own attack predictions has not
        # solved the task, it has solved the readout — and its belief metrics look *good*
        # precisely because they have stopped describing the behaviour. Flag rather than let
        # that be read as performance. Measured at noise 0.20: the RNN at WU_lr 0.005/0.01
        # reports near-perfect context (slips 0.03-0.04, better than most NG) while its attack
        # prediction sits at 0.73-0.75 for the whole block and never adapts at all.
        flag = ' *' if (not np.isnan(vals[7]) and vals[7] < AGREEMENT_FLOOR) else ''
        if flag:
            flagged.append(label)
        print(f"{label:<22}{vals[0]:>9.3f}{vals[1]:>9.2f}{vals[2]:>11.2f}"
              f"{vals[3]:>11.3f}{vals[4]:>9.3f}{vals[5]:>8.1f}{vals[6]:>10.3f}"
              f"{vals[7]:>8.3f}{flag}")

    if flagged:
        print(f"\n  * belief-behaviour agreement < {AGREEMENT_FLOOR} for: {', '.join(flagged)}.")
        print( "    The belief head has partly decoupled from the attack prediction there, so the")
        print( "    belief columns overstate them. Check the xy behaviour directly")
        print( "    (analyze_block_switch_adaptation) before concluding anything from their slips.")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Check the belief readout against synthetic data, independent of any trained model.

    A "model" that emits the ground-truth context dims must decode to the true rotation exactly;
    one that emits the previous block's context must score as fully perseverative. This is the
    assumption the whole analysis rests on, so it is worth testing rather than assuming.
    """
    from configs import RotatingTargetsConfig
    from datasets import RotatingTargetsDataset

    cfg = RotatingTargetsConfig()
    cfg.train_rotations = list(TRAIN_ROTATIONS)
    cfg.enable_context_output(CONTEXT_OUTPUT_ENCODING)
    cfg.block_size  = 14 * cfg.n_colors * 2
    cfg.no_of_blocks = 6
    ds = RotatingTargetsDataset(cfg)
    obs = np.array(ds.data_sequence)
    ll  = np.array(ds.llcid_sequence)
    nc  = cfg.n_colors

    decoded = np.arctan2(obs[:, nc + 1], obs[:, nc])
    err = _angular_error_deg(decoded, ll)
    assert err.max() < 1e-9, f'context round-trip failed: max {err.max()} deg'
    print(f'  round-trip decode                 : max {err.max():.2e} deg   OK')

    assert np.allclose(np.linalg.norm(obs[:, nc:nc + 2], axis=1), cfg.target_radius)
    print('  |context| == target_radius        : OK')

    # A perfect belief scores 100% correct; the previous block's belief scores 0% and is entirely
    # perseverative (never reaches criterion, so slips are NaN by definition).
    rots_rad = np.deg2rad(np.asarray(cfg.train_rotations, dtype=float))
    true_slot = _nearest_slot(ll, rots_rad)
    assert (_nearest_slot(decoded, rots_rad) == true_slot).all()
    print('  perfect belief -> 100% correct     : OK')

    params = AnalysisParams(criterion_n=3, last_ts_in_a_block=15)
    block_of = np.concatenate([[0], np.cumsum(np.diff(ll) != 0)])
    stale = np.roll(ll, 1); stale[0] = ll[0]
    for bi in np.unique(block_of)[1:]:
        sel = block_of == bi
        stale[sel] = ll[np.flatnonzero(sel)[0] - 1]
    r = cfg.target_radius
    vec = lambda a: r * np.stack([np.cos(a), np.sin(a)], axis=1)
    prev = np.array([ll[np.flatnonzero(block_of == b)[0] - 1] if b > 0 else np.nan
                     for b in block_of])
    d_cur = np.linalg.norm(vec(stale) - vec(ll), axis=1)
    d_alt = np.linalg.norm(vec(stale) - vec(prev), axis=1)
    trials = dict(
        belief_rad=stale, belief_mag=np.ones_like(stale),
        belief_norm=np.where(np.isnan(prev), np.nan, d_cur / (d_cur + d_alt + 1e-12)),
        true_rad=ll, prev_rot=prev,
        belief_slot=_nearest_slot(stale, rots_rad), true_slot=true_slot,
        correct=_nearest_slot(stale, rots_rad) == true_slot,
        behav_rad=np.full(len(ll), np.nan), block_id=block_of,
        pos_in_block=np.concatenate([np.arange(int((block_of == b).sum()))
                                     for b in np.unique(block_of)]),
        t_index=np.arange(len(ll)),
    )
    m = block_criterion_metrics(trials, params)
    assert np.all(m['persev'] > 0) and np.all(np.isnan(m['slips'])), m
    print('  stale belief -> all perseveration  : OK')

    # A belief parked on the previous context must score exactly 1.0 on the 0/0.5/1 scale.
    bn = trials['belief_norm'][~np.isnan(trials['belief_norm'])]
    assert np.allclose(bn, 1.0), f'stale belief should be 1.0, got {bn.min()}..{bn.max()}'
    print('  stale belief -> belief_norm = 1.0  : OK')
    print('\nSELF-TEST PASSED')


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> dict:
    global runs_cache
    params     = AnalysisParams()
    export_dir = EXPORT_ROOT / 'figures'

    # Reload when new runs have landed or conditions have been added, not just when the cache is
    # absent — otherwise re-running in the same session keeps reporting the sweep as incomplete.
    if cache_is_stale(globals().get('runs_cache'), params):
        runs_cache = load_all_runs(params)

    if not runs_cache:
        print(f'No results under {EXPORT_ROOT}. Run rotation_slips_perseveration_sweep.py first.')
        return {}

    plot_belief_trajectory(runs_cache, params, export_dir)
    plot_context_correct(runs_cache, params, export_dir)
    plot_perseveration_and_slips(runs_cache, params, export_dir)
    plot_slips_vs_noise(runs_cache, params, export_dir)
    plot_belief_dynamics(runs_cache, params, export_dir)
    plot_diagnostics(runs_cache, params, export_dir)
    summarize(runs_cache, params)
    return runs_cache


if __name__ == '__main__':
    print('Self-test (synthetic, no trained model needed):')
    _self_test()
    runs_cache = main()
