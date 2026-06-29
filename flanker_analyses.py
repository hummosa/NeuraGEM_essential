"""
flanker_analyses.py — Reusable trial extraction, selection, and plotting utilities
for the flanker task stages. Import in run_flanker.py or analysis notebooks.
"""

import itertools
import numpy as np


# ── Trial extraction ──────────────────────────────────────────────────────────

def extract_trials(logger, config, rt_threshold=0.5):
    """
    Unpack a logger into per-trial aligned arrays.

    Parameters
    ----------
    logger : Logger object from train_model
    config : stage config (used for arrows_duration, input_size, output_size,
             response_start_timestep)
    rt_threshold : |output| threshold for declaring a decision

    Returns
    -------
    dict with keys:
        correct       (n_trials, ad)   — 1.0 if correct at each timestep
        output_traj   (n_trials, ad)   — raw model output (last dim) per timestep
        z_traj        (n_trials, ad, Z_dim) — Z values per timestep
        trial_type    (n_trials,)      — hlcids value per trial (float)
        context_id    (n_trials,)      — context_ids value per trial (float)
        true_dir      (n_trials,)      — true direction ±1.0
        is_correct    (n_trials,)      — bool, correct at final timestep
        response_side (n_trials,)      — sign of output at final timestep (+1/-1/0)
        rt            (n_trials,)      — first ts where |output|>threshold; ad if never
        rt_threshold  float
        ad            int
        n_trials      int
    """
    ad  = config.arrows_duration
    ii  = np.concatenate(logger.inputs,            axis=0).reshape(-1, config.input_size)
    oi  = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1, config.output_size)
    tt  = np.concatenate(logger.hlcids,            axis=0).reshape(-1)
    cid = np.concatenate(logger.context_ids,       axis=0).reshape(-1)

    z_raw  = np.concatenate(logger.latent_values, axis=0)
    z_flat = z_raw.reshape(-1, z_raw.shape[-1])

    n_ts     = (len(ii) // ad) * ad
    n_trials = n_ts // ad

    true_dir    = ii[:n_ts, -1].reshape(n_trials, ad)
    output_traj = oi[:n_ts, -1].reshape(n_trials, ad)
    correct     = (true_dir * output_traj > 0).astype(float)
    z_traj      = z_flat[:n_ts].reshape(n_trials, ad, -1)

    decided = np.abs(output_traj) > rt_threshold
    rt = np.where(decided.any(axis=1),
                  decided.argmax(axis=1).astype(float),
                  float(ad))

    # Sign-normalize output by the model's own final decision (sign of output at t=-1).
    # Positive = accumulating toward the eventual decision; negative = against it.
    # Cancellation-free: left and right decisions are aligned on the same axis.
    final_decision_sign = np.sign(output_traj[:, -1:])   # (n_trials, 1)
    signed_output = output_traj * final_decision_sign     # (n_trials, ad)

    return dict(
        correct       = correct,
        output_traj   = output_traj,
        signed_output = signed_output,
        z_traj        = z_traj,
        trial_type    = tt[:n_ts].reshape(n_trials, ad)[:, 0],
        context_id    = cid[:n_ts].reshape(n_trials, ad)[:, 0],
        true_dir      = true_dir[:, 0],
        is_correct    = correct[:, -1].astype(bool),
        response_side = np.sign(output_traj[:, -1]),
        rt            = rt,
        rt_threshold  = rt_threshold,
        ad            = ad,
        n_trials      = n_trials,
    )


# ── Trial selection ───────────────────────────────────────────────────────────

def select_trials(trials, trial_type=None, is_correct=None, response_side=None):
    """
    Return a boolean mask (n_trials,) for the requested trial subset.

    trial_type   : int, float, or list — match against trials['trial_type'] (hlcids)
    is_correct   : True = correct only, False = errors only
    response_side: +1.0 rightward, -1.0 leftward
    """
    mask = np.ones(trials['n_trials'], dtype=bool)
    if trial_type is not None:
        mask &= np.isin(trials['trial_type'], np.atleast_1d(trial_type))
    if is_correct is not None:
        mask &= (trials['is_correct'] == is_correct)
    if response_side is not None:
        mask &= (trials['response_side'] == response_side)
    return mask


# ── History grouping ──────────────────────────────────────────────────────────

def build_history_groups(trials, n_back=2, current_mask=None, congruent_types=(0, 2)):
    """
    Group trials by the congruency of the n_back preceding trials.

    For each trial passing `current_mask`, the history key is a tuple of booleans
    (True = congruent) ordered oldest-to-most-recent: (t-n_back, ..., t-1).

    Parameters
    ----------
    congruent_types : trial_type values treated as congruent.
        Default (0, 2) = near-cong and far-cong in Stage 3/4.
        For Stage 1/2 where trial_type IS the congruency flag, use (1,).

    Returns
    -------
    dict: history_key (tuple of bool, length n_back) → boolean mask (n_trials,)
    """
    is_cong = np.isin(trials['trial_type'], congruent_types)
    n       = trials['n_trials']
    if current_mask is None:
        current_mask = np.ones(n, dtype=bool)

    groups = {k: [] for k in itertools.product([True, False], repeat=n_back)}
    for t in range(n_back, n):
        if not current_mask[t]:
            continue
        key = tuple(bool(is_cong[t - n_back + i]) for i in range(n_back))
        groups[key].append(t)

    bool_groups = {}
    for k, idx_list in groups.items():
        m = np.zeros(n, dtype=bool)
        if idx_list:
            m[np.array(idx_list)] = True
        bool_groups[k] = m
    return bool_groups


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _style_trial_ax(ax, ad, response_start):
    ax.axvspan(-0.5, response_start - 0.5, alpha=0.08, color='k')
    ax.set_xticks(range(ad))
    ax.set_xlabel('Timestep within trial')
    ax.legend(fontsize=5)


def plot_accuracy_by_timestep(ax, trials, specs, config, linestyles=None):
    """
    Plot mean accuracy per timestep for multiple trial groups.

    specs : list of (mask, label, color) — mask is boolean (n_trials,)
    """
    correct = trials['correct']
    ad      = trials['ad']
    if linestyles is None:
        linestyles = ['-'] * len(specs)
    for (mask, label, color), ls in zip(specs, linestyles):
        if mask.any():
            ax.plot(correct[mask].mean(axis=0), color=color, linestyle=ls,
                    label=f'{label} (n={mask.sum()})')
    ax.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
    _style_trial_ax(ax, ad, config.response_start_timestep)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.3, 1.05)


def plot_rt_distribution(ax, trials, specs, config, fit_gaussian=True, linestyles=None,
                         undecided='extra_bin'):
    """
    Plot RT as an empirical probability mass function (markers connected by lines)
    plus an optional Gaussian fit (dashed overlay).

    PMF sums to 1.0; undecided trials are always included.

    Parameters
    ----------
    undecided : 'extra_bin'  — add a t=ad bin to the right for undecided trials (default)
                'last_bin'   — stack undecided trials onto the last existing bin (t=ad-1)

    specs : list of (mask, label, color) — mask is boolean (n_trials,)
    linestyles : applied to the empirical PMF line; Gaussian always plotted dashed.
    """
    rt           = trials['rt']
    ad           = trials['ad']
    rt_threshold = trials['rt_threshold']
    if linestyles is None:
        linestyles = ['-'] * len(specs)

    use_extra = (undecided == 'extra_bin')
    x_bins = list(range(ad + 1)) if use_extra else list(range(ad))

    for (mask, label, color), ls in zip(specs, linestyles):
        if not mask.any():
            continue
        n_total   = int(mask.sum())
        rt_m      = rt[mask]
        decided   = rt_m < ad
        n_decided = int(decided.sum())

        if use_extra:
            # decided bins 0..ad-1, undecided bin = ad
            pmf = np.array([(rt_m == t).sum() / n_total for t in range(ad + 1)])
        else:
            # decided bins 0..ad-2, undecided stacked into bin ad-1
            pmf = np.array([(rt_m == t).sum() / n_total for t in range(ad - 1)]
                           + [(rt_m >= ad - 1).sum() / n_total])

        ax.plot(x_bins, pmf, marker='o', markersize=4, color=color,
                linestyle=ls, linewidth=0.8, label=f'{label} (n={n_total})', zorder=3)

        if fit_gaussian and n_decided >= 3:
            try:
                from scipy.stats import norm
                p_decided = n_decided / n_total
                mu, sigma = norm.fit(rt_m[decided])
                x_fine = np.linspace(-0.5, ad - 0.5, 300)
                y_fine = norm.pdf(x_fine, mu, sigma) * p_decided
                ax.plot(x_fine, y_fine, color=color, linestyle='--',
                        linewidth=1.2, alpha=0.7, zorder=2)
            except Exception:
                pass

    # x-axis: decided bins always labelled by timestep; extra bin labelled 'und.'
    if use_extra:
        tick_labels = [str(t) for t in range(ad)] + ['und.']
        ax.set_xticks(x_bins)
        ax.set_xticklabels(tick_labels, fontsize=6)
        ax.set_xlabel('Timestep within trial')
        ax.axvspan(-0.5, config.response_start_timestep - 0.5, alpha=0.08, color='k')
        ax.legend(fontsize=5)
    else:
        _style_trial_ax(ax, ad, config.response_start_timestep)
    ax.set_ylabel(f'P(RT = t)  [threshold = {rt_threshold}]')
    ax.set_ylim(bottom=0)


def plot_z_by_timestep(ax, trials, specs, z_dim, config, linestyles=None):
    """
    Plot mean Z value for dimension `z_dim` across timesteps within a trial.

    specs : list of (mask, label, color) — mask is boolean (n_trials,)
    """
    z_traj = trials['z_traj']  # (n_trials, ad, Z_dim)
    ad     = trials['ad']
    if linestyles is None:
        linestyles = ['-'] * len(specs)
    for (mask, label, color), ls in zip(specs, linestyles):
        if mask.any():
            ax.plot(z_traj[mask, :, z_dim].mean(axis=0), color=color,
                    linestyle=ls, label=label)
    _style_trial_ax(ax, ad, config.response_start_timestep)
    ax.set_ylabel(f'Z dim {z_dim}')


def plot_scalar_bars(ax, trials, specs, measure, group_spacing=None):
    """
    Bar chart with mean ± SEM for a scalar-per-trial measure.

    Parameters
    ----------
    specs   : list of (mask, label, color) — mask is boolean (n_trials,)
    measure : 'accuracy'  — correctness at the RT-crossing timestep
                            (last timestep used for undecided trials)
              'rt'        — RT in timesteps (arrows_duration for undecided)
              int         — Z dimension index; value at RT-crossing timestep
    group_spacing : list of int, optional
                    Extra gap (in bar-widths) inserted before bar at each listed index.
                    E.g. [2, 4] adds a gap before the 3rd and 5th bars.

    Returns
    -------
    x_positions : array — x positions used, for further annotation.
    """
    ad = trials['ad']
    n  = trials['n_trials']
    rt_ts = np.minimum(trials['rt'].astype(int), ad - 1)   # clamp undecided to last ts

    if measure == 'accuracy':
        values_all = trials['correct'][np.arange(n), rt_ts]
        ylabel = 'Accuracy at decision'
        ylim_bottom = 0.5 #* np.mean(values_all)
    elif measure == 'rt':
        values_all = trials['rt']
        ylabel = 'RT (timesteps)'
        ylim_bottom = 0.5 * np.mean(values_all)
    elif isinstance(measure, int):
        # Z dimension value at RT-crossing timestep
        values_all = trials['z_traj'][np.arange(n), rt_ts, measure]
        ylabel = f'Z dim {measure} at decision'
        ylim_bottom = 0.5 * np.mean(values_all)
    elif isinstance(measure, tuple) and len(measure) == 2 and measure[0] == 'z_end':
        # Z at the final timestep — same as z_start since Z is constant within a trial
        values_all = trials['z_traj'][:, -1, measure[1]]
        ylabel = f'Z dim {measure[1]} (end of trial)'
        ylim_bottom = 0.5 * np.mean(values_all)
    elif isinstance(measure, tuple) and len(measure) == 2 and measure[0] == 'z_start':
        # Z at the first timestep of the trial — the incoming Z carried from the previous trial's LU update,
        # before this trial's own prediction error has influenced Z.
        values_all = trials['z_traj'][:, 0, measure[1]]
        ylabel = f'Z dim {measure[1]} (start of trial)'
        ylim_bottom = 0.5 * np.mean(values_all)
    else:
        raise ValueError(f'Unknown measure: {measure!r}')

    means, sems, labels, colors, valid_idx = [], [], [], [], []
    for i, (mask, label, color) in enumerate(specs):
        if not mask.any():
            continue
        v = values_all[mask]
        means.append(v.mean())
        sems.append(v.std(ddof=1) / np.sqrt(len(v)))
        labels.append(f'{label}\n(n={int(mask.sum())})')
        colors.append(color)
        valid_idx.append(i)

    # Build x positions with optional extra gaps between groups
    gaps = set(group_spacing or [])
    x, pos = [], 0.0
    for k, orig_i in enumerate(valid_idx):
        if k > 0 and orig_i in gaps:
            pos += 0.7   # extra gap
        x.append(pos)
        pos += 1.0
    x = np.array(x)

    ax.bar(x, means, color=colors, alpha=0.75, width=0.6, zorder=2)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='k',
                capsize=3, linewidth=1, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.grid(axis='y', linewidth=0.4, alpha=0.4, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return x


# ── Config / model helpers ────────────────────────────────────────────────────

def sync_gating(stage_config, from_config):
    """
    Copy runtime gating flags from from_config to stage_config.
    Replaces the repeated 4-line block in run_flanker.py.
    """
    for attr in ('pre_gating', 'post_gating', 'use_add_gating', 'use_mul_gating'):
        setattr(stage_config, attr, getattr(from_config, attr))


def mirror_to_model(model, stage_config, attrs=('Z_lr', 'no_of_steps_in_latent_space')):
    """
    Mirror stage_config attributes onto model.config so LU reads the new values.
    model.config is a direct reference (not a copy), so this patches the right object.
    """
    for attr in attrs:
        if hasattr(stage_config, attr):
            setattr(model.config, attr, getattr(stage_config, attr))
