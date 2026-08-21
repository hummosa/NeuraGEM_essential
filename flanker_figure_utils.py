"""
flanker_figure_utils.py — shared panel primitives and data loading for flanker figures.

Every flanker figure is built from the same three ingredients, so they live here rather
than in any one figure script:

    loading    collect_effects / collect_sessions / pretrain_curves — variant-aware
    per-seed   session_curves — the curves that get averaged across seeds
    panels     bars_with_seeds, band, dots_with_ci, series

Figure scripts import from here:

    flanker_sweep_figures.py   the original six group figures
    flanker_model_figures.py   the model card and the manipulation series

Conventions
───────────
Every panel shows the mean across seeds with SEM and one dot per seed, because the seed
is the unit of analysis. Sizes always come from `plot_style.FigSize` — never a literal
`figsize` — so the paper/dev scale switch keeps working.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import matplotlib.pyplot as plt

import plot_style
plot_style.set_plot_style()
from plot_style import FigSize

from flanker_analyses import extract_trials, lagged_factors
from flanker_metrics import session_effects
from flanker_sweep import (SWEEP_RUNS, describe_runs, load_condition,
                           load_pretrain_curve, pretrain_tag, result_path, use_run)
from flanker_sweep_config import RT_THRESHOLD, SEEDS


# ── Palette ───────────────────────────────────────────────────────────────────

COL = dict(cong='#4393c3', incong='#d6604d',
           near_cong='#4393c3', far_cong='#a8c8e8',
           near_incong='#8b0000', far_incong='#f4a582',
           correct='#4393c3', error='#8b0000', neutral='#777777',
           pass_='#2166ac', fail='#b2182b', null='#999999')

CELLS = [('near_cong', 'near\ncong'), ('far_cong', 'far\ncong'),
         ('near_incong', 'near\nincong'), ('far_incong', 'far\nincong')]


# ── Loading ───────────────────────────────────────────────────────────────────

#: Backwards-compatible alias — the implementation now lives in flanker_sweep, next to
#: RUN_NAME itself, so there is one place that knows which runs exist.
sweep_run = use_run


def out_dir_for(p_congruent, variant='baseline'):
    """Directory that holds one condition's results — where its figures belong too."""
    return os.path.dirname(result_path(0, p_congruent, variant))


def sweep_root():
    """Parent of every condition folder — where cross-condition figures belong."""
    return os.path.dirname(os.path.dirname(result_path(0, 0.5)))


def collect_sessions(p_congruent, variant='baseline', rt_threshold=RT_THRESHOLD):
    """Load one condition; return (per-seed effects, per-seed curves)."""
    effects, curves = [], []
    for res in load_condition(p_congruent, variant=variant):
        trials = extract_trials(res['train_logger'], res['config'], rt_threshold=rt_threshold)
        effects.append(session_effects(trials))
        curves.append(session_curves(trials))
    return effects, curves


def collect_effects(p_congruent, variant='baseline', rt_threshold=RT_THRESHOLD):
    """Per-seed effects only — skips the curve building when a figure has no time axis."""
    out = []
    for res in load_condition(p_congruent, variant=variant):
        trials = extract_trials(res['train_logger'], res['config'], rt_threshold=rt_threshold)
        out.append(session_effects(trials))
    return out


def pretrain_curves(variant='baseline', n_bins=25):
    """Stage-1 learning curves for every seed of the model set `variant` needs."""
    tag = pretrain_tag(variant)
    per_seed = {'all': [], 'cong': [], 'incong': []}
    x = None
    for seed in range(SEEDS):
        rec = load_pretrain_curve(seed, tag)
        if rec is None:
            continue
        correct   = rec['correct'].astype(float)
        congruent = rec['congruent'] == 1.0
        x, curves = _binned(correct, {'all': np.ones(len(correct), bool),
                                      'cong': congruent, 'incong': ~congruent}, n_bins)
        for key, curve in curves.items():
            per_seed[key].append(curve)
    return x, {k: np.array(v) for k, v in per_seed.items() if v}


def _binned(values, masks, n_bins):
    """Mean of `values` in `n_bins` equal slices of trial position, per mask."""
    n     = len(values)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    x     = 0.5 * (edges[:-1] + edges[1:])
    out   = {}
    for key, m in masks.items():
        curve = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = np.zeros(n, dtype=bool)
            sel[a:b] = True
            sel &= m
            curve.append(values[sel].mean() if sel.any() else np.nan)
        out[key] = np.array(curve)
    return x, out


def _stack(effects, key):
    """Per-seed values of one effect, as an array."""
    return np.array([e[key] for e in effects], dtype=float)


def _stack_curve(curves, group, key):
    return np.array([c[group][key] for c in curves], dtype=float)


# ── Per-session curves ────────────────────────────────────────────────────────

def session_curves(trials, n_bins=20, rt_bin_width=0.25):
    """Per-session curves that later get averaged across seeds."""
    f = lagged_factors(trials, n_back=2)
    cong, near = f['cong'] == 1, f['near'] == 1
    acc  = trials['correct_at_decision'].astype(float)
    rt   = trials['rt_interp']
    ad   = trials['ad']
    corr = trials['correct_at_decision']

    masks = {'near_cong': near & cong,    'far_cong': ~near & cong,
             'near_incong': near & ~cong, 'far_incong': ~near & ~cong}

    out = {'acc_by_ts': {}, 'accum': {}, 'rt_density': {}, 'learning': {}}

    for key, m in masks.items():
        out['acc_by_ts'][key] = trials['correct'][m].mean(axis=0)

    for key, m in [('correct_cong', corr & cong), ('error_cong', ~corr & cong),
                   ('correct_incong', corr & ~cong), ('error_incong', ~corr & ~cong)]:
        out['accum'][key] = (trials['signed_output'][m].mean(axis=0)
                             if m.any() else np.full(ad, np.nan))

    # RTs now run past the trial: trials that never crossed are extrapolated and capped,
    # so the density axis has to reach rt_cap rather than stopping at the last timestep.
    hi       = float(trials.get('rt_cap', ad))
    rt_bins  = np.arange(0, hi + rt_bin_width, rt_bin_width)
    out['rt_bins'] = 0.5 * (rt_bins[:-1] + rt_bins[1:])
    density_masks = dict(masks)
    density_masks.update({'cong': cong, 'incong': ~cong, 'correct': corr, 'error': ~corr})
    for key, m in masks.items():                       # cell x outcome
        density_masks[f'{key}_corr'] = m & corr
        density_masks[f'{key}_err']  = m & ~corr
    for key, m in density_masks.items():
        vals = rt[m]
        vals = vals[~np.isnan(vals)]
        dens, _ = np.histogram(vals, bins=rt_bins, density=True)
        out['rt_density'][key] = dens if len(vals) else np.full(len(rt_bins) - 1, np.nan)

    x, learning = _binned(acc, {'all': np.ones_like(cong), 'cong': cong, 'incong': ~cong}, n_bins)
    out['learning_x'] = x
    out['learning']   = learning
    return out


# ── Panels ────────────────────────────────────────────────────────────────────

def bars_with_seeds(ax, groups, ylabel, baseline=None, connect=False, rotation=0,
                    title=None):
    """
    Bar chart of across-seed means with SEM, overlaid with one dot per seed.

    groups  : list of (values_per_seed, label, color)
    connect : join the same seed across bars — use for within-subject contrasts
    """
    x      = np.arange(len(groups))
    means  = [np.nanmean(v) for v, _, _ in groups]
    sems   = [np.nanstd(v, ddof=1) / np.sqrt(max(np.sum(~np.isnan(v)), 1)) for v, _, _ in groups]
    colors = [c for _, _, c in groups]

    ax.bar(x, means, color=colors, alpha=0.65, width=0.62, zorder=2)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='k', capsize=3, linewidth=1, zorder=4)

    n_seeds = max(len(v) for v, _, _ in groups)
    if connect and len(groups) > 1:
        per_seed = np.full((n_seeds, len(groups)), np.nan)
        for j, (v, _, _) in enumerate(groups):
            per_seed[:len(v), j] = v
        for row in per_seed:
            ax.plot(x, row, color='k', alpha=0.13, linewidth=0.5, zorder=3)

    rng = np.random.default_rng(0)
    for j, (v, _, _) in enumerate(groups):
        jitter = (rng.random(len(v)) - 0.5) * 0.22 if not connect else np.zeros(len(v))
        ax.scatter(np.full(len(v), x[j]) + jitter, v, s=5, color='k',
                   alpha=0.45, zorder=5, linewidths=0)

    if baseline is not None:
        ax.axhline(baseline, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in groups],
                       rotation=rotation, ha='center' if rotation == 0 else 'right')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _label_width(label):
    """Rough width of a tick label, in inches, at the current tick font size."""
    try:
        pt = float(plt.rcParams['xtick.labelsize'])
    except (TypeError, ValueError):
        pt = 6.0
    longest = max(len(line) for line in str(label).split('\n'))
    return 0.62 * pt * longest / 72 + 0.06        # mean advance of sans-serif + a gap


def bar_panel_width(groups, margin=0.55, min_slot=0.28):
    """Paper-ready width for one bar panel: enough for the tick labels it has to carry."""
    slot = max(min_slot, max(_label_width(lbl) for _, lbl, _ in groups))
    return margin + slot * len(groups)


def bar_row(panels, height=2.0):
    """
    A row of bar panels, each as wide as its own labels need — draws them and returns
    (fig, axes).

    panels : list of (groups, kwargs), where `groups` and `kwargs` are what
             `bars_with_seeds` takes.

    A row of equal-width panels is the wrong shape for bar charts: it leaves a two-bar
    panel half empty while a four-bar panel crushes its tick labels together, and the
    figure ends up far wider than the ink in it. Widths come from the labels instead, via
    `FigSize.custom`, so the dev/paper switch keeps working.
    """
    widths = [bar_panel_width(groups) for groups, _ in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=FigSize.custom(sum(widths), height),
                             gridspec_kw={'width_ratios': widths})
    axes = np.atleast_1d(axes)
    for ax, (groups, kw) in zip(axes, panels):
        bars_with_seeds(ax, groups, **kw)
    return fig, axes


def band(ax, x, arr, label, color, linestyle='-'):
    """Mean ± SEM band across seeds. arr is (n_seeds, n_points)."""
    arr = np.asarray(arr, dtype=float)
    n   = np.sum(~np.isnan(arr), axis=0)
    mu  = np.nanmean(arr, axis=0)
    se  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    ax.plot(x, mu, color=color, linestyle=linestyle, linewidth=1.0, label=label)
    ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.18, linewidth=0)


def series(ax, x, per_level, label, color, marker='o', seed_lines=False):
    """
    Mean ± SEM across seeds at each level of a manipulation.

    per_level : (n_levels, n_seeds) — the same seeds at every level, so the thin
                per-seed lines are the within-subject contrast the statistics test.
    """
    arr = np.asarray(per_level, dtype=float)
    mu  = np.nanmean(arr, axis=1)
    se  = np.nanstd(arr, axis=1, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=1))
    if seed_lines:
        for s in range(arr.shape[1]):
            ax.plot(x, arr[:, s], color=color, alpha=0.15, linewidth=0.4, zorder=1)
    ax.errorbar(x, mu, yerr=se, marker=marker, markersize=3, color=color,
                capsize=2, linewidth=1.0, label=label, zorder=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def dots_with_ci(ax, y, values, color, label=None, marker='o', filled=True):
    """One row of a scorecard: mean with a 95% CI across seeds, at height y."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan
    mean = v.mean()
    sem  = v.std(ddof=1) / np.sqrt(len(v))
    ax.errorbar(mean, y, xerr=1.96 * sem, color=color, capsize=2, linewidth=0.9,
                marker=marker, markersize=3.5, label=label,
                markerfacecolor=color if filled else 'none', zorder=3)
    return mean


def compact_legend(ax, **kw):
    """
    In-axes legend that fits a paper-sized panel.

    No frame and tight spacing, so the legend costs a corner of the axes rather than
    pushing the data around. Font size is left to rcParams — never pass `fontsize=`.
    """
    kw.setdefault('frameon', False)
    kw.setdefault('handlelength', 1.0)
    kw.setdefault('handletextpad', 0.4)
    kw.setdefault('borderpad', 0.2)
    kw.setdefault('labelspacing', 0.25)
    return ax.legend(**kw)


def share_ylim(*axes, hide_inner=True):
    """
    Put a group of panels on one y scale, so their heights can be read against each other.

    Applied after plotting rather than through `sharey=` at subplot creation, because only
    some panels in a row measure the same thing — sharing the whole row would squash the
    rest. `hide_inner` drops the repeated y-label and tick labels from every panel but the
    first, which is only right when the panels are genuinely side by side.
    """
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for i, ax in enumerate(axes):
        ax.set_ylim(lo, hi)
        if hide_inner and i:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)


def _interactive_kernel():
    """True inside a Jupyter / VS Code interactive kernel, where figures can be shown."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    return ip is not None and hasattr(ip, 'kernel')


def save(fig, path, note=None):
    """
    Save at paper size and report where it went (and how big it actually is).

    In a Jupyter / VS Code interactive window the figure is also displayed before it is
    closed, so running a figure script there shows the panels as well as writing the PDF.
    """
    fig.savefig(path, bbox_inches='tight')
    w, h = fig.get_size_inches()
    print(f'Exported: {path}  [{w:.1f}x{h:.1f} in]' + (f'  — {note}' if note else ''))
    if _interactive_kernel():
        from IPython.display import display
        display(fig)
    plt.close(fig)
    return path
