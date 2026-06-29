"""
rotating_targets_analysis.py — Analysis functions for the rotating-targets task.

Implements analyses from the Yu et al. (2025) comparison plan:
  A1: Trial-level prediction error in the first mini-block after a block switch
  A2: Normalized state error (0=current rotation, 0.5=chance, 1=alternate rotation)
  A3: Per-mini-block learning curves after a block switch
  A4: Z-trajectory at block transitions (NeuraGEM only)
  A5: Novel rotation generalization — same metrics on Phase 3a (unseen angles)

All training-phase analyses (A1–A4) target the last 1/3 of Phase 2, where all
models have had full opportunity to learn.  A5 targets Phase 3a ('Inference only'),
which runs with test_rotations and keeps weights plastic when
config.test_no_of_steps_in_weight_space=1.
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import FigSize


# ── Geometry helpers ──────────────────────────────────────────────────────────

def get_target_positions(config, rotation_deg):
    """Return (n_colors, 2) target positions for the given rotation (degrees)."""
    nc = config.n_colors
    theta = np.radians(rotation_deg)
    angles_0 = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    base = config.target_radius * np.stack([np.cos(angles_0), np.sin(angles_0)], axis=1)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return (R @ base.T).T  # (nc, 2)


def _nearest_rotation_deg(rot_rad, rotation_set, tol=1e-5):
    """Return the closest degree in rotation_set for a given radian value, or None."""
    for deg in rotation_set:
        if abs(rot_rad - np.radians(deg)) < tol:
            return deg
    return None


# ── Logger extraction ─────────────────────────────────────────────────────────

def flatten_logger(logger, config):
    """
    Return flat numpy arrays: (ii, oi, ll, li).

    Each row corresponds to one logged timestep (stride=1, so no overlap).
    """
    ii = np.concatenate(logger.inputs, axis=0).reshape(-1, config.input_size)
    oi = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1, config.output_size)
    ll = np.concatenate(logger.context_ids, axis=0).reshape(-1)
    li = None
    if logger.latent_values:
        li = np.concatenate(logger.latent_values, axis=0).reshape(-1, sum(config.latent_dims))
    return ii, oi, ll, li


def get_phase2_last_third(logger):
    """
    Return (last_third_start, phase2_end) as flat logged-timestep indices.

    'last_third_start' is the index where the last 1/3 of Phase 2 begins.
    """
    phase2_start = logger.others['timestep_passive_learning_ended']
    phase2_end   = logger.others['timestep_learning_ended']
    last_third_len   = (phase2_end - phase2_start) // 3
    last_third_start = phase2_end - last_third_len
    return last_third_start, phase2_end


def get_block_switches(ll, start, end):
    """Return list of flat timestep indices where llcid changes, within [start, end)."""
    return [t for t in range(max(1, start), end) if ll[t] != ll[t - 1]]


def get_phase3a_range(logger):
    """
    Return (phase3a_start, phase3a_end) for the 'Inference only' phase.

    Returns (None, None) if Phase 3 was not run.
    """
    phase3a_start = None
    phase3a_end   = None
    for name, ts in logger.phases:
        if name == 'Inference only':
            phase3a_start = ts
        elif name == 'No inference nor learning' and phase3a_start is not None:
            phase3a_end = ts
            break
    if phase3a_start is not None and phase3a_end is None:
        # Phase 3b was not run (or data wasn't in phases); use end of log
        phase3a_end = len(logger.inputs)
    return phase3a_start, phase3a_end


# ── Core adaptation analysis ───────────────────────────────────────────────────

def _analyze_adaptation(ii, oi, ll, config, rotation_set,
                         t_start, t_end, n_miniblocks_to_track=6,
                         reference_rotation_deg=None):
    """
    Core block-switch adaptation analysis over an arbitrary time range and rotation set.

    Parameters
    ----------
    ii, oi, ll : flat numpy arrays from flatten_logger
    config     : RotatingTargetsConfig
    rotation_set : list of rotation angles (degrees) to use for target geometry
    t_start, t_end : time range (flat logged-timestep indices)
    n_miniblocks_to_track : mini-blocks after each switch to analyse
    reference_rotation_deg : if given, always use this angle as the "alternate"
        for normalized state error instead of picking the first non-current angle.
        Recommended for novel-rotation tests with many angles — pass the original
        (trained) reference rotation (e.g. 0°) so the metric is well-defined.
        When None, falls back to the first non-current angle (works for 2-rotation case).

    Returns a dict:
      trial_errors      (n_switches, n_colors)            Euclidean ‖pred−obs‖
      trial_norm_errors (n_switches, n_colors)            normalized state error
      miniblock_errors  (n_switches, n_miniblocks)        mean Euclidean per mini-block
      miniblock_norm    (n_switches, n_miniblocks)        mean normalized per mini-block
      n_switches        int
      switches          list of switch timestep indices
    """
    switches = get_block_switches(ll, t_start, t_end)
    nc = config.n_colors

    # Pre-compute targets for all rotation_set angles plus the reference (if any)
    target_degs = list(rotation_set)
    if reference_rotation_deg is not None and reference_rotation_deg not in target_degs:
        target_degs.append(reference_rotation_deg)
    all_targets = {deg: get_target_positions(config, deg) for deg in target_degs}

    n_sw = len(switches)
    trial_errors      = np.full((n_sw, nc), np.nan)
    trial_norm_errors = np.full((n_sw, nc), np.nan)
    miniblock_errors  = np.full((n_sw, n_miniblocks_to_track), np.nan)
    miniblock_norm    = np.full((n_sw, n_miniblocks_to_track), np.nan)

    for si, t_sw in enumerate(switches):
        curr_rot_rad = ll[t_sw]
        curr_rot_deg = _nearest_rotation_deg(curr_rot_rad, rotation_set)
        if curr_rot_deg is None:
            continue

        if reference_rotation_deg is not None:
            alt_rot_deg = reference_rotation_deg
        else:
            alt_degs = [d for d in rotation_set if d != curr_rot_deg]
            if not alt_degs:
                continue
            alt_rot_deg = alt_degs[0]

        curr_targets = all_targets[curr_rot_deg]
        alt_targets  = all_targets[alt_rot_deg]

        # Scan forward for cue frames in the new block
        scan_end = min(t_sw + (n_miniblocks_to_track + 1) * nc * 2, t_end, len(ll) - 2)
        cue_ts = [
            t for t in range(t_sw, scan_end)
            if ii[t, :nc].sum() > 0.5 and abs(ll[t] - curr_rot_rad) < 1e-5
        ]

        mb_err_lists  = [[] for _ in range(n_miniblocks_to_track)]
        mb_norm_lists = [[] for _ in range(n_miniblocks_to_track)]

        trial_pos = 0  # position within current mini-block (0 .. nc-1)
        mb_idx    = 0  # which mini-block since the switch

        for t in cue_ts:
            if mb_idx >= n_miniblocks_to_track:
                break

            color_idx = int(np.argmax(ii[t, :nc]))
            pred_xy   = oi[t + 1, -2:]
            obs_xy    = ii[t + 1, -2:]

            err      = np.linalg.norm(pred_xy - obs_xy)
            d_curr   = np.linalg.norm(pred_xy - curr_targets[color_idx])
            d_alt    = np.linalg.norm(pred_xy - alt_targets[color_idx])
            norm_err = d_curr / (d_curr + d_alt + 1e-9)

            if mb_idx == 0 and trial_pos < nc:
                trial_errors[si, trial_pos]      = err
                trial_norm_errors[si, trial_pos] = norm_err

            mb_err_lists[mb_idx].append(err)
            mb_norm_lists[mb_idx].append(norm_err)

            trial_pos += 1
            if trial_pos >= nc:
                trial_pos = 0
                mb_idx += 1

        for mb in range(n_miniblocks_to_track):
            if mb_err_lists[mb]:
                miniblock_errors[si, mb] = np.mean(mb_err_lists[mb])
                miniblock_norm[si, mb]   = np.mean(mb_norm_lists[mb])

    return dict(
        trial_errors=trial_errors,
        trial_norm_errors=trial_norm_errors,
        miniblock_errors=miniblock_errors,
        miniblock_norm=miniblock_norm,
        n_switches=n_sw,
        switches=switches,
    )


# ── A1 / A2 / A3: Phase-2 block-switch adaptation ─────────────────────────────

def analyze_block_switch_adaptation(logger, config, n_miniblocks_to_track=6):
    """
    A1/A2/A3: block-switch adaptation in the last 1/3 of Phase 2, using train_rotations.

    Trial position: ordinal rank (1..n_colors) of each color in the first mini-block
    after the switch. Trial 1 = first observation; trials 2-N test zero-shot transfer.
    """
    ii, oi, ll, _ = flatten_logger(logger, config)
    last_third_start, phase2_end = get_phase2_last_third(logger)
    return _analyze_adaptation(
        ii, oi, ll, config,
        rotation_set=config.train_rotations,
        t_start=last_third_start,
        t_end=phase2_end,
        n_miniblocks_to_track=n_miniblocks_to_track,
    )


# ── A5: Novel rotation generalization ─────────────────────────────────────────

def analyze_novel_rotation_test(logger, config, n_miniblocks_to_track=6):
    """
    A5: Block-switch adaptation in Phase 3a ('Inference only'), using test_rotations.

    Phase 3a runs immediately after Phase 2 with novel rotation angles.
    When config.test_no_of_steps_in_weight_space=1, weights remain plastic here.

    Normalized state error uses config.train_rotations[0] (typically 0°) as the fixed
    reference "alternate", matching the paper's Session 3 convention.

    Returns the same dict as analyze_block_switch_adaptation, or None if Phase 3
    was not run or test_rotations is empty.
    """
    phase3a_start, phase3a_end = get_phase3a_range(logger)
    if phase3a_start is None:
        return None

    test_rots = config.test_rotations if config.test_rotations else config.train_rotations
    if not test_rots:
        return None

    ii, oi, ll, _ = flatten_logger(logger, config)
    reference_deg = config.train_rotations[0] if config.train_rotations else None
    return _analyze_adaptation(
        ii, oi, ll, config,
        rotation_set=test_rots,
        t_start=phase3a_start,
        t_end=phase3a_end,
        n_miniblocks_to_track=n_miniblocks_to_track,
        reference_rotation_deg=reference_deg,
    )


# ── A4: Z-trajectory ──────────────────────────────────────────────────────────

def analyze_z_trajectory(logger, config, window=80):
    """
    Project Z onto the axis connecting the two rotation centroids and return
    the displacement fraction (0 = old centroid, 1 = new centroid) for each
    timestep after a block switch in the last 1/3 of Phase 2.

    Centroids are computed as the mean Z in the first 2/3 of Phase 2.

    Returns (trajectories, switches):
      trajectories : (n_switches, window) float array, NaN where out of range
      switches     : list of switch timestep indices
    """
    ii, oi, ll, li = flatten_logger(logger, config)
    if li is None:
        return None, []

    phase2_start     = logger.others['timestep_passive_learning_ended']
    last_third_start, phase2_end = get_phase2_last_third(logger)
    switches = get_block_switches(ll, last_third_start, phase2_end)

    # Centroids from the first 2/3 of Phase 2 (stable representations)
    centroids = {}
    for deg in config.train_rotations:
        rot_rad = np.radians(deg)
        mask    = np.abs(ll[phase2_start:last_third_start] - rot_rad) < 1e-5
        z_vals  = li[phase2_start:last_third_start][mask]
        if len(z_vals) > 0:
            centroids[deg] = z_vals.mean(axis=0)

    if len(centroids) < 2:
        return None, switches

    n_sw         = len(switches)
    trajectories = np.full((n_sw, window), np.nan)

    for si, t_sw in enumerate(switches):
        prev_rot_rad = ll[t_sw - 1]
        curr_rot_rad = ll[t_sw]
        prev_deg = _nearest_rotation_deg(prev_rot_rad, config.train_rotations)
        curr_deg = _nearest_rotation_deg(curr_rot_rad, config.train_rotations)
        if prev_deg not in centroids or curr_deg not in centroids:
            continue

        z_old = centroids[prev_deg]
        z_new = centroids[curr_deg]
        axis  = z_new - z_old
        total_dist = np.linalg.norm(axis)
        if total_dist < 1e-8:
            continue

        t_end = min(t_sw + window, phase2_end, len(ll))
        for i, t in enumerate(range(t_sw, t_end)):
            trajectories[si, i] = np.dot(li[t] - z_old, axis) / (total_dist ** 2)

    return trajectories, switches


# ── A6: Rotation sweep (per-angle error across all test angles) ────────────────

def analyze_rotation_sweep(logger, config, block_range=None):
    """
    For each angle in Phase 3a, compute mean Euclidean error on trials 2-5
    (0-indexed positions 1-4) of the first mini-block after that angle appears.

    Error metric: ‖pred_xy − analytical_target_mean(θ, color)‖ using
    get_target_positions(config, θ) — NOT the noisy sampled observation on that trial.
    This measures how well the model's prediction aligns with the true task structure,
    independent of observation noise.

    Parameters
    ----------
    block_range : (start_idx, end_idx) into the ordered list of state-block switches
                  in Phase 3a (0-indexed). None = use all blocks.
                  Use this to compare early Phase-3a performance (still learning,
                  e.g. block_range=(0, 15)) versus late (stabilised, e.g. (35, 50)).

    Returns
    -------
    dict mapping rotation_deg (int) -> mean error (float) or np.nan if that angle
    never appeared in the selected block range.  Keyed over config.test_rotations.
    """
    phase3a_start, phase3a_end = get_phase3a_range(logger)
    if phase3a_start is None:
        return {deg: np.nan for deg in (config.test_rotations or config.train_rotations)}

    ii, oi, ll, _ = flatten_logger(logger, config)
    nc        = config.n_colors
    test_rots = config.test_rotations if config.test_rotations else config.train_rotations
    all_targets = {deg: get_target_positions(config, deg) for deg in test_rots}

    switches = get_block_switches(ll, phase3a_start, phase3a_end)
    switch_list = []
    for t_sw in switches:
        deg = _nearest_rotation_deg(ll[t_sw], test_rots)
        if deg is not None:
            switch_list.append((t_sw, deg))

    if block_range is not None:
        bstart, bend = block_range
        switch_list = switch_list[bstart:bend]

    angle_errors = {deg: [] for deg in test_rots}

    for t_sw, curr_rot_deg in switch_list:
        curr_rot_rad = np.radians(curr_rot_deg)
        curr_targets = all_targets[curr_rot_deg]

        # Collect cue frames in the first mini-block (one full permutation = nc colors)
        scan_end = min(t_sw + 2 * nc * 2, phase3a_end, len(ll) - 2)
        cue_ts = [
            t for t in range(t_sw, scan_end)
            if ii[t, :nc].sum() > 0.5 and abs(ll[t] - curr_rot_rad) < 1e-5
        ]
        first_mb = cue_ts[:nc]  # first mini-block only

        for trial_pos, t in enumerate(first_mb):
            if trial_pos == 0:
                continue  # trial 1 = identification; can't be "correct"
            color_idx = int(np.argmax(ii[t, :nc]))
            pred_xy   = oi[t + 1, -2:]
            # Error vs analytical ground truth mean — not the noisy sampled observation
            target_xy = curr_targets[color_idx]
            angle_errors[curr_rot_deg].append(np.linalg.norm(pred_xy - target_xy))

    return {
        deg: float(np.mean(errs)) if errs else np.nan
        for deg, errs in angle_errors.items()
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _mean_sem(arr):
    """Return (mean, SEM) over axis=0, ignoring NaNs."""
    n = np.sum(~np.isnan(arr), axis=0).clip(1)
    m = np.nanmean(arr, axis=0)
    s = np.nanstd(arr, axis=0) / np.sqrt(n)
    return m, s


_DEFAULT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']


def _make_palette(labels):
    return dict(zip(labels, _DEFAULT_COLORS))


def plot_adaptation_comparison(results, config, title=None, include_title_in_panel=False):
    """
    P1–P4 side by side for all model conditions (Phase 2, trained rotations).

    Parameters
    ----------
    results : dict mapping label -> output of analyze_block_switch_adaptation()
              plus optional 'z_trajectory' key.
    config  : RotatingTargetsConfig
    """
    nc      = config.n_colors
    labels  = list(results.keys())
    palette = _make_palette(labels)

    fig, axes = plt.subplots(1, 4, figsize=FigSize.row(4, FigSize.small))
    x_trial = np.arange(1, nc + 1)

    ax = axes[0]
    for label, res in results.items():
        m, s = _mean_sem(res['trial_errors'])
        c = palette.get(label, 'grey')
        ax.plot(x_trial, m, '-o', ms=4, label=label, color=c)
        ax.fill_between(x_trial, m - s, m + s, alpha=0.18, color=c)
    ax.set_xlabel('Trial in mini-block')
    ax.set_ylabel('‖pred − obs‖')
    if include_title_in_panel:
        ax.set_title('P1: Trial-level error\n(Phase 2, trained rotations)', fontsize=8)
    ax.set_xticks(x_trial)
    ax.legend( framealpha=0.7)

    ax = axes[1]
    for label, res in results.items():
        m, s = _mean_sem(res['trial_norm_errors'])
        c = palette.get(label, 'grey')
        ax.plot(x_trial, m, '-o', ms=4, label=label, color=c)
        ax.fill_between(x_trial, m - s, m + s, alpha=0.18, color=c)
    ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4, label='Chance')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Trial in mini-block')
    ax.set_ylabel('Normalized state error')
    if include_title_in_panel:
        ax.set_title('P2: Normalized error\n(0=correct, 0.5=chance, 1=wrong)', fontsize=8)
    ax.set_xticks(x_trial)

    ax = axes[2]
    n_mb = list(results.values())[0]['miniblock_errors'].shape[1]
    x_mb = np.arange(1, n_mb + 1)
    for label, res in results.items():
        m, s = _mean_sem(res['miniblock_errors'])
        c = palette.get(label, 'grey')
        ax.plot(x_mb, m, '-o', ms=4, label=label, color=c)
        ax.fill_between(x_mb, m - s, m + s, alpha=0.18, color=c)
    ax.set_xlabel('Mini-block # after switch')
    ax.set_ylabel('Mean ‖pred − obs‖')
    if include_title_in_panel:
        ax.set_title('P3: Mini-block learning curve', fontsize=8)
    ax.set_xticks(x_mb)

    ax = axes[3]
    for label, res in results.items():
        traj = res.get('z_trajectory')
        if traj is not None:
            m, s = _mean_sem(traj)
            x = np.arange(len(m))
            c = palette.get(label, 'grey')
            ax.plot(x, m, lw=1.2, label=label, color=c)
            ax.fill_between(x, m - s, m + s, alpha=0.18, color=c)
    ax.axhline(0, color='grey', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(1, color='grey', lw=0.5, ls=':', alpha=0.5)
    ax.set_xlabel('Timesteps after block switch')
    ax.set_ylabel('Z displacement fraction')
    if include_title_in_panel:
        ax.set_title('P4: Z-trajectory\n(0=old context, 1=new context)', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=9, y=1.02)
    plt.tight_layout()
    return fig


def plot_miniblock_norm_comparison(results, config, title=None):
    """P3b variant: normalized state error per mini-block."""
    n_mb = list(results.values())[0]['miniblock_norm'].shape[1]
    x    = np.arange(1, n_mb + 1)
    palette = _make_palette(list(results.keys()))

    fig, ax = plt.subplots(figsize=FigSize.wide)
    for label, res in results.items():
        m, s = _mean_sem(res['miniblock_norm'])
        c = palette.get(label, 'grey')
        ax.plot(x, m, '-o', ms=4, label=label, color=c)
        ax.fill_between(x, m - s, m + s, alpha=0.18, color=c)
    ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4, label='Chance')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Mini-block # after switch')
    ax.set_ylabel('Normalized state error')
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_novel_rotation_comparison(novel_results, config, title=None):
    """
    P5/P6: Adaptation to novel rotations (Phase 3a).

    novel_results : dict mapping label -> analyze_novel_rotation_test() output
    """
    nc      = config.n_colors
    labels  = list(novel_results.keys())
    palette = _make_palette(labels)

    fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3, FigSize.wide))
    x_trial = np.arange(1, nc + 1)

    ax = axes[0]
    for label, res in novel_results.items():
        if res is None:
            continue
        m, s = _mean_sem(res['trial_norm_errors'])
        c = palette.get(label, 'grey')
        ax.plot(x_trial, m, '-o', ms=4, label=label, color=c)
        ax.fill_between(x_trial, m - s, m + s, alpha=0.18, color=c)
    ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4, label='Chance')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Trial in mini-block')
    ax.set_ylabel('Normalized state error')
    ax.set_title('Novel rotation — trial-level\n(0=correct, 0.5=chance, 1=wrong)', fontsize=8)
    ax.set_xticks(x_trial)
    ax.legend()

    ax = axes[1]
    for label, res in novel_results.items():
        if res is None:
            continue
        m, s = _mean_sem(res['miniblock_norm'])
        n_mb = len(m)
        x_mb = np.arange(1, n_mb + 1)
        c = palette.get(label, 'grey')
        ax.plot(x_mb, m, '-o', ms=4, label=label, color=c)
        ax.fill_between(x_mb, m - s, m + s, alpha=0.18, color=c)
    ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Mini-block # after switch')
    ax.set_ylabel('Normalized state error')
    ax.set_title('Novel rotation — mini-block curve', fontsize=8)

    ax = axes[2]
    for label, res in novel_results.items():
        if res is None:
            continue
        m, s = _mean_sem(res['miniblock_errors'])
        n_mb = len(m)
        x_mb = np.arange(1, n_mb + 1)
        c = palette.get(label, 'grey')
        ax.plot(x_mb, m, '-o', ms=4, label=label, color=c)
        ax.fill_between(x_mb, m - s, m + s, alpha=0.18, color=c)
    ax.set_xlabel('Mini-block # after switch')
    ax.set_ylabel('Mean ‖pred − obs‖')
    ax.set_title('Novel rotation — Euclidean error', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=9, y=1.02)
    plt.tight_layout()
    return fig


# ── Z-space visualization (NeuraGEM) ──────────────────────────────────────────

def _softmax2(z):
    """Softmax along last axis for a 2-column array."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def plot_z_by_rotation(logger, config, time_window=None):
    """
    Visualize what Z values NeuraGEM uses for each rotation context.

    Three panels:
      A — 2D scatter of raw (pre-softmax) Z[0] vs Z[1], colored by rotation.
          Stars mark per-rotation centroids.  With softmax activation the
          effective signal is Z[0]-Z[1], so points cluster along a diagonal.
      B — Effective 1D context signal softmax(Z)[0] over time around
          the Phase 2 → Phase 3a boundary.  Background shaded by rotation.
      C — Context tuning curve: mean softmax(Z)[0] ± std for each rotation
          angle ordered by angle, annotated as training vs novel.

    Parameters
    ----------
    logger      : Logger from a NeuraGEM run (Phase 3a must be present for novel)
    config      : RotatingTargetsConfig
    time_window : (t_start, t_end) for Panel B.  Defaults to last 3 state-blocks
                  of Phase 2 + all of Phase 3a.
    """
    ii, oi, ll, li = flatten_logger(logger, config)
    if li is None:
        raise ValueError('No latent values logged.')

    phase2_start = logger.others['timestep_passive_learning_ended']
    phase2_end   = logger.others['timestep_learning_ended']
    p3a_start, p3a_end = get_phase3a_range(logger)

    train_rots = list(config.train_rotations)
    test_rots  = list(config.test_rotations) if config.test_rotations else []
    all_rots   = train_rots + [r for r in test_rots if r not in train_rots]

    # Circular HSV colormap: 0° and 360° share the same hue
    angle_cmap = plt.get_cmap('hsv')

    def _rot_color(deg):
        return angle_cmap(deg / 360.0)

    train_set  = set(train_rots)
    all_rots_s = sorted(all_rots)

    # Use second half of Phase 2 as "settled" training representation
    phase2_mid = phase2_start + (phase2_end - phase2_start) // 2

    fig, axes = plt.subplots(1, 2, figsize=FigSize.row(2, FigSize.wide))

    # ── Panel A: context signal over time ──────────────────────────────────
    ax = axes[0]

    if time_window is not None:
        t0, t1 = time_window
    else:
        t0 = max(phase2_start, phase2_end - 3 * config.block_size)
        t1 = p3a_end if p3a_end is not None else phase2_end

    t_idx   = np.arange(t0, t1)
    z_win   = li[t0:t1]
    l_win   = ll[t0:t1]
    context = _softmax2(z_win)[:, 0]

    # Background shaded by rotation angle — no per-angle legend entries
    for deg in all_rots_s:
        rot_rad = np.radians(deg)
        mask = np.abs(l_win - rot_rad) < 1e-5
        ax.fill_between(t_idx, 0, 1, where=mask,
                        alpha=0.3, color=_rot_color(deg))

    ax.plot(t_idx, context, lw=0.8, color='k', alpha=0.8, label='softmax(Z)[0]')

    if p3a_start is not None and t0 <= p3a_start <= t1:
        ax.axvline(p3a_start, color='k', lw=1.5, ls='--', label='↳ novel rotations')

    for t_ in range(t0 + 1, t1):
        if ll[t_] != ll[t_ - 1]:
            ax.axvline(t_, color='grey', lw=0.5, alpha=0.35)

    # Colorbar replaces the per-angle legend
    sm = plt.cm.ScalarMappable(cmap=angle_cmap, norm=plt.Normalize(0, 360))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Rotation (°)', fraction=0.04, pad=0.02,
                 ticks=[0, 90, 180, 270, 360])

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('softmax(Z)[0]')
    ax.legend(loc='upper right', fontsize=6)


    # ── Panel B: tuning curve ─────────────────────────────────────────────
    ax = axes[1]
    angles   = all_rots_s
    rot_mean, rot_std = [], []

    for deg in angles:
        rot_rad = np.radians(deg)
        if deg in train_set:
            z_src = li[phase2_mid:phase2_end]
            l_src = ll[phase2_mid:phase2_end]
        else:
            if p3a_start is None:
                rot_mean.append(np.nan); rot_std.append(np.nan)
                continue
            skip  = p3a_start + config.block_size
            z_src = li[skip:p3a_end]
            l_src = ll[skip:p3a_end]

        mask = np.abs(l_src - rot_rad) < 1e-5
        z    = z_src[mask]
        if len(z) == 0:
            rot_mean.append(np.nan); rot_std.append(np.nan)
            continue
        sig = _softmax2(z)[:, 0]
        rot_mean.append(sig.mean())
        rot_std.append(sig.std())

    rot_mean = np.array(rot_mean)
    rot_std  = np.array(rot_std)

    # Track whether we've already added the generic "Trained" / "Novel" legend entry
    _seen = {'train': False, 'novel': False}
    for i, deg in enumerate(angles):
        if np.isnan(rot_mean[i]):
            continue
        is_train = deg in train_set
        key      = 'train' if is_train else 'novel'
        leg      = ('Trained' if not _seen['train'] else '_nolegend_') if is_train \
                   else ('Novel'   if not _seen['novel'] else '_nolegend_')
        _seen[key] = True
        ax.errorbar(deg, rot_mean[i], yerr=rot_std[i],
                    fmt='o' if is_train else '^',
                    color=_rot_color(deg),
                    ms=9 if is_train else 5,
                    capsize=3, elinewidth=1.0,
                    markeredgecolor='k', markeredgewidth=0.8 if is_train else 0.3,
                    label=leg)

    ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4, label='Chance (0.5)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-15, max(angles) + 15)
    ax.set_xticks([a for a in angles if a % 90 == 0])
    ax.set_xlabel('Rotation angle (°)')
    ax.set_ylabel('softmax(Z)[0]  ± std')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_rotation_angle_sweep(sweep_results, config, block_range_label=None):
    """
    P9: Mean Euclidean error (trials 2-5, first mini-block) vs rotation angle.

    Error is vs the analytical target mean (get_target_positions), not a sampled
    observation — see analyze_rotation_sweep for details.

    sweep_results : dict mapping model label -> output of analyze_rotation_sweep()
    block_range_label : optional string appended to y-label to identify which
        block range was used (e.g. 'early blocks 0-15' or 'late blocks 35-50').
    """
    labels  = list(sweep_results.keys())
    palette = _make_palette(labels)

    train_rots = set(config.train_rotations)

    fig, ax = plt.subplots(figsize=FigSize.wide)

    for label, sweep in sweep_results.items():
        angles = sorted(sweep.keys())
        errors = [sweep[a] for a in angles]
        c = palette.get(label, 'grey')
        ax.plot(angles, errors, '-o', ms=3, label=label, color=c)

    # Mark trained rotations with vertical dashed lines
    for tr in sorted(train_rots):
        ax.axvline(tr, color='grey', lw=0.8, ls='--', alpha=0.5)

    ax.set_xlabel('Rotation angle (°)')
    ylabel = '‖pred − target mean‖ (trials 2–5)'
    if block_range_label:
        ylabel += f'\n{block_range_label}'
    ax.set_ylabel(ylabel)
    ax.set_xticks([a for a in sorted(set(sweep_results[labels[0]].keys()))
                   if a % 45 == 0])
    ax.legend()
    plt.tight_layout()
    return fig


# ── Z color-modulation analysis ───────────────────────────────────────────────

def analyze_z_color_modulation(logger, config, phase='phase2'):
    """
    Quantify whether Z encodes cue color identity in addition to (or instead of)
    the ground-truth rotation context.

    Alignment
    ---------
    `li[t]` is Z *after* the LU step that processed the batch containing timestep t.
    With `pass_previous_latent=True` and seq_len=2 (one cue frame + one outcome frame
    per batch), the LU gradient at the cue position comes from ∂(attack-xy loss)/∂Z,
    meaning the logged Z at a cue frame is specifically informed by predicting *that*
    color's attack position.  We therefore sample Z at cue frames only.

    Two R² measures per Z dimension
    ---------------------------------
    rotation_r2      : OLS R² from Z ~ rotation one-hot  (global rotation encoding)
    within_color_r2  : OLS R² from rotation-residual Z ~ color one-hot
                       (color variance *within* each rotation block)

    A rotation-only encoder yields within_color_r2 ≈ 0.
    A color-leaking encoder yields within_color_r2 > 0.

    Parameters
    ----------
    phase : 'phase2' | 'phase3a' | 'all'
        Time range to analyse.  'phase2' uses the last 1/3 of the blocked phase.

    Returns
    -------
    results : dict
        rotation_r2      (Z_dim,) — rotation variance explained
        within_color_r2  (Z_dim,) — within-rotation color variance explained
        z_by_color_rot   (n_rots, n_colors, Z_dim) — mean Z per (rotation, color)
        unique_rots      list of rotation angles used
        n_cue_samples    int
    fig : matplotlib Figure
        Left panels: mean Z per color, one line per rotation (up to 4 Z dims shown).
        Right panel: bar chart comparing rotation_r2 vs within_color_r2 per Z dim.
    """
    ii, oi, ll, li = flatten_logger(logger, config)
    if li is None:
        raise ValueError('No latent values logged.')

    nc    = config.n_colors
    Z_dim = sum(config.latent_dims)

    if phase == 'phase2':
        t_start, t_end = get_phase2_last_third(logger)
        rotation_set   = config.train_rotations
    elif phase == 'phase3a':
        t_start, t_end = get_phase3a_range(logger)
        if t_start is None:
            raise ValueError('Phase 3a not found in logger.')
        rotation_set   = config.test_rotations or config.train_rotations
    else:  # 'all'
        t_start      = 0
        t_end        = len(ii)
        rotation_set = sorted(set(
            list(config.train_rotations) + list(config.test_rotations or [])
        ))

    # ── Collect (color_idx, Z, rotation_deg) at every cue timestep ───────────
    colors, z_at_cue, rot_degs = [], [], []
    for t in range(int(t_start), min(int(t_end), len(ii) - 1)):
        if ii[t, :nc].sum() < 0.5:
            continue
        rot_deg = _nearest_rotation_deg(ll[t], rotation_set)
        if rot_deg is None:
            continue
        colors.append(int(np.argmax(ii[t, :nc])))
        z_at_cue.append(li[t])
        rot_degs.append(rot_deg)

    if len(colors) == 0:
        raise ValueError('No cue timesteps found in the selected phase/rotation set.')

    colors   = np.array(colors)
    z_at_cue = np.array(z_at_cue)   # (N, Z_dim)
    rot_degs = np.array(rot_degs)

    unique_rots = sorted(set(rot_degs.tolist()))
    n_rots      = len(unique_rots)
    rot_idx     = np.array([unique_rots.index(r) for r in rot_degs])

    # ── OLS helper ────────────────────────────────────────────────────────────
    def _r2_ols(X, y):
        beta   = np.linalg.lstsq(X, y, rcond=None)[0]
        ss_res = np.sum((y - X @ beta) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(np.clip(1.0 - ss_res / (ss_tot + 1e-12), 0.0, 1.0))

    rot_onehot   = np.eye(n_rots)[rot_idx]
    color_onehot = np.eye(nc)[colors]

    rotation_r2 = np.array([_r2_ols(rot_onehot, z_at_cue[:, d]) for d in range(Z_dim)])

    # Pooled partial color R²: color variance after removing rotation mean
    within_color_r2 = np.zeros(Z_dim)
    for d in range(Z_dim):
        y        = z_at_cue[:, d]
        rot_beta = np.linalg.lstsq(rot_onehot, y, rcond=None)[0]
        y_resid  = y - rot_onehot @ rot_beta
        within_color_r2[d] = _r2_ols(color_onehot, y_resid)

    # Per-rotation color R²: computed independently within each rotation context.
    # This catches rotation-specific color modulation that the pooled metric misses
    # (e.g. Z dims that fan out by color only under one rotation).
    per_rot_color_r2 = np.zeros((n_rots, Z_dim))
    for ri in range(n_rots):
        mask  = rot_idx == ri
        z_sub = z_at_cue[mask]
        c_sub = color_onehot[mask]
        if mask.sum() >= nc:
            for d in range(Z_dim):
                per_rot_color_r2[ri, d] = _r2_ols(c_sub, z_sub[:, d])

    # ── Mean Z per (rotation, color) ─────────────────────────────────────────
    z_by_color_rot = np.full((n_rots, nc, Z_dim), np.nan)
    for ri, rot in enumerate(unique_rots):
        for ci in range(nc):
            mask = (rot_degs == rot) & (colors == ci)
            if mask.sum() > 0:
                z_by_color_rot[ri, ci] = z_at_cue[mask].mean(axis=0)

    # ── Figure ────────────────────────────────────────────────────────────────
    n_z_panels = min(Z_dim, 4)   # cap to avoid absurdly wide figures
    n_cols     = n_z_panels + 1
    fig, axes  = plt.subplots(1, n_cols, figsize=FigSize.row(n_cols, FigSize.small))
    axes       = list(np.atleast_1d(axes))

    rot_cmap = plt.get_cmap('tab10', max(n_rots, 2))

    for d in range(n_z_panels):
        ax = axes[d]
        for ri, rot in enumerate(unique_rots):
            means = z_by_color_rot[ri, :, d]
            ax.plot(np.arange(nc), means, '-o', ms=4, lw=1.2,
                    label=f'{rot:.0f}°  col R²={per_rot_color_r2[ri, d]:.2f}',
                    color=rot_cmap(ri))
        ax.set_xlabel('Color index')
        ax.set_ylabel(f'Mean Z[{d}]')
        ax.set_xticks(np.arange(nc))
        ax.set_title(f'Z[{d}]  rot R²={rotation_r2[d]:.2f}', fontsize=7)
        ax.legend(fontsize=5)

    # Bar chart: rotation R² + per-rotation color R² side by side
    ax  = axes[-1]
    x   = np.arange(Z_dim)
    n_bars = 1 + n_rots          # rotation + one bar per rotation for color
    w   = 0.8 / n_bars
    ax.bar(x - (n_bars / 2 - 0.5) * w, rotation_r2, width=w,
           label='Rotation R²', color='tab:blue', alpha=0.85)
    for ri, rot in enumerate(unique_rots):
        offset = (-n_bars / 2 + 1 + ri) * w
        ax.bar(x + offset, per_rot_color_r2[ri], width=w,
               label=f'Color R² @{rot:.0f}°', color=rot_cmap(ri), alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Z[{d}]' for d in range(Z_dim)], rotation=45, fontsize=6)
    ax.set_ylabel('Variance explained (R²)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5)
    ax.set_title('Z: rotation vs color (per rotation)', fontsize=7)

    plt.tight_layout()

    return dict(
        rotation_r2=rotation_r2,
        within_color_r2=within_color_r2,
        per_rot_color_r2=per_rot_color_r2,
        z_by_color_rot=z_by_color_rot,
        unique_rots=unique_rots,
        n_cue_samples=len(colors),
    ), fig
