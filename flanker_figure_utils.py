"""
flanker_figure_utils.py — shared panel primitives and data loading for flanker figures.

Every flanker figure is built from the same three ingredients, so they live here rather
than in any one figure script:

    loading    collect_effects / collect_sessions / pretrain_curves — variant-aware
    per-seed   session_curves — the curves that get averaged across seeds
    panels     bars_with_seeds, band, dots_with_ci, series
    specs      spec_* — panel definitions shared by the group and single-session scripts

Figure scripts import from here:

    flanker_sweep_figures.py   the group figures, in the order the story is told
    run_flanker.py             the single-session workbench, for the panels they share

One panel, two callers
──────────────────────
The two scripts differ in one thing only: what a replicate is. At group level it is a
seed, in the workbench it is the single session. `_as_replicates` absorbs that — it takes
either a list of per-seed effect dicts or one `session_effects` dict — and the `spec_*`
builders on top of it return panel definitions both scripts hand to `bar_row` / `bar_grid`.
So a change to a shared panel lands in both figures at once, which is the point.

The cost, stated once: with a single replicate `bars_with_seeds` draws one dot and no
whisker, so a workbench panel built this way has no error bar where the mask-based
`flanker_analyses.plot_scalar_bars` would show a trial-level SEM. That is already true of
`plot_circularity`'s bar panels in run_flanker's Result 4c. Use `plot_scalar_bars` when
the trial-level spread is the point; use these when the group figure and the workbench
figure must not drift apart.

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
from plot_style import (FLANKER_CELLS, FLANKER_COLORS, FigSize, flanker_color,
                        outcome_style)

from flanker_analyses import extract_trials, lagged_factors
from flanker_metrics import session_effects
from flanker_sweep import (SWEEP_RUNS, describe_runs, load_condition,
                           load_pretrain_curve, pretrain_tag, result_path, use_run)
from flanker_sweep_config import RT_THRESHOLD, SEEDS, VARIANTS


# ── Palette ───────────────────────────────────────────────────────────────────
#
# One scheme for every flanker figure, defined in plot_style: hue = congruency,
# shade = distance, fill = outcome. Re-exported under the short names the figure
# scripts already use.

COL   = FLANKER_COLORS
CELLS = FLANKER_CELLS


# ── Loading ───────────────────────────────────────────────────────────────────

#: Backwards-compatible alias — the implementation now lives in flanker_sweep, next to
#: RUN_NAME itself, so there is one place that knows which runs exist.
sweep_run = use_run


def out_dir_for(variant):
    """Directory that holds one variant's results — where its figures belong too."""
    return os.path.dirname(result_path(0, variant))


def sweep_root():
    """Parent of every variant folder — where cross-variant figures belong."""
    return os.path.dirname(os.path.dirname(result_path(0, next(iter(VARIANTS)))))


def collect_sessions(variant, rt_threshold=RT_THRESHOLD):
    """Load one variant; return (per-seed effects, per-seed curves)."""
    effects, curves = [], []
    for res in load_condition(variant):
        trials = extract_trials(res['train_logger'], res['config'], rt_threshold=rt_threshold)
        effects.append(session_effects(trials))
        curves.append(session_curves(trials))
    return effects, curves


def collect_effects(variant, rt_threshold=RT_THRESHOLD):
    """Per-seed effects only — skips the curve building when a figure has no time axis."""
    out = []
    for res in load_condition(variant):
        trials = extract_trials(res['train_logger'], res['config'], rt_threshold=rt_threshold)
        out.append(session_effects(trials))
    return out


def pretrain_curves(variant, n_bins=25):
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


def _as_replicates(effects, key):
    """
    Per-replicate values of one effect, whichever caller is asking.

    `effects` is either a list of per-seed dicts (the group scripts, one replicate per
    seed) or a single `flanker_metrics.session_effects` dict (the workbench, one
    replicate). Everything downstream — means, SEMs, the seed dots — then works unchanged.
    """
    if isinstance(effects, dict):
        return np.array([effects[key]], dtype=float)
    return _stack(effects, key)


def _has(effects, key):
    """Whether an effect key exists at all — `z_grad` is optional, so its keys may not be."""
    probe = effects if isinstance(effects, dict) else (effects[0] if effects else {})
    return key in probe


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

    # RT as a PMF over the integer crossing timestep, with the trials that never crossed
    # as a final `und.` category rather than folded into the last real bin — the same
    # convention `flanker_analyses.plot_rt` uses for a single session. With four eligible
    # timesteps this is the resolution the data actually has, and a non-response is a
    # different outcome from a slow response rather than a large value of one.
    rt_int  = np.asarray(trials['rt'], dtype=float)
    decided = np.asarray(trials['decided'], dtype=bool)
    out['rt_x'] = np.arange(ad + 1, dtype=float)          # last slot is 'und.'
    out['rt_pmf'] = {}

    # The interpolated density is kept alongside for the sub-timestep view; it covers the
    # decided trials only, scaled so its area is the decided proportion.
    hi       = float(ad)
    rt_bins  = np.arange(0, hi + rt_bin_width, rt_bin_width)
    out['rt_bins'] = 0.5 * (rt_bins[:-1] + rt_bins[1:])
    density_masks = dict(masks)
    density_masks.update({'cong': cong, 'incong': ~cong, 'correct': corr, 'error': ~corr})
    for key, m in masks.items():                       # cell x outcome
        density_masks[f'{key}_corr'] = m & corr
        density_masks[f'{key}_err']  = m & ~corr
    for key, m in density_masks.items():
        n_total = int(m.sum())
        if not n_total:
            out['rt_pmf'][key] = np.full(ad + 1, np.nan)
            out['rt_density'][key] = np.full(len(rt_bins) - 1, np.nan)
            continue
        p_und = float((~decided[m]).mean())
        out['rt_pmf'][key] = np.array([(rt_int[m] == t).sum() / n_total for t in range(ad)]
                                      + [p_und])
        vals = rt[m & decided]
        vals = vals[~np.isnan(vals)]
        dens, _ = np.histogram(vals, bins=rt_bins, density=True)
        out['rt_density'][key] = (dens * (1.0 - p_und) if len(vals)
                                  else np.full(len(rt_bins) - 1, np.nan))

    x, learning = _binned(acc, {'all': np.ones_like(cong), 'cong': cong, 'incong': ~cong}, n_bins)
    out['learning_x'] = x
    out['learning']   = learning
    return out


# ── Panels ────────────────────────────────────────────────────────────────────

def bars_with_seeds(ax, groups, ylabel, baseline=None, connect=False, rotation=0,
                    title=None, hollow=None, ylim=None):
    """
    Bar chart of across-seed means with SEM, overlaid with one dot per seed.

    groups  : list of (values_per_seed, label, color)
    connect : join the same seed across bars — use for within-subject contrasts
    hollow  : optional list of bools, one per group. True draws the bar as an outline
              instead of a filled block — the house convention for an *error* cell
              (`plot_style.outcome_style`), so outcome never has to spend a hue.
    ylim    : optional (lo, hi); either may be None to leave that side automatic. Bars
              are drawn from zero, so an accuracy panel whose cells all sit between 0.65
              and 0.98 spends two thirds of its height on empty space and the contrast
              the panel exists to show reads as flat. Clipping the axis is the fix, and
              `baseline` then marks where chance is so the truncation stays honest.
    """
    x      = np.arange(len(groups))
    means  = [np.nanmean(v) for v, _, _ in groups]
    # ddof=1 is NaN for a single replicate (one session rather than a sweep), so fall
    # back to no whisker instead of a NaN that matplotlib silently drops.
    sems   = [(np.nanstd(v, ddof=1) / np.sqrt(max(np.sum(~np.isnan(v)), 1))
               if np.sum(~np.isnan(v)) > 1 else 0.0) for v, _, _ in groups]
    colors = [c for _, _, c in groups]

    hollow = [False] * len(groups) if hollow is None else list(hollow)
    for xi, mean, color, is_hollow in zip(x, means, colors, hollow):
        ax.bar(xi, mean, width=0.62, zorder=2,
               **outcome_style(not is_hollow, kind='bar', color=color))
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
    # Last, so it wins over the autoscaling that the bars and seed dots just triggered.
    if ylim is not None:
        ax.set_ylim(*ylim)


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


def bar_grid(rows, height=2.0):
    """
    A grid of bar panels — `bar_row` when one row is not enough. Returns (fig, axes).

    rows : list of rows, each a list of (groups, kwargs) exactly as `bar_row` takes.

    Column widths come from the widest tick label in that column, for the same reason
    `bar_row` sizes by label: a grid of equal columns leaves the two-bar panels half empty
    and crushes the six-bar ones. Short rows leave their trailing axes hidden rather than
    blank-framed.
    """
    ncol   = max(len(row) for row in rows)
    widths = [max(bar_panel_width(groups)
                  for row in rows for j, (groups, _) in enumerate(row) if j == i)
              for i in range(ncol)]
    fig, axes = plt.subplots(len(rows), ncol, squeeze=False,
                             figsize=FigSize.custom(sum(widths), height * len(rows)),
                             gridspec_kw={'width_ratios': widths})
    for row, ax_row in zip(rows, axes):
        for ax, (groups, kw) in zip(ax_row, row):
            bars_with_seeds(ax, groups, **kw)
        for ax in ax_row[len(row):]:
            ax.set_visible(False)
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


def landing_marks(ev):
    """
    Where the trial *after* each kind of event actually starts, as `exchange_panel` marks.

    `focus_in` is the state a trial inherited, so lag +1 is what the next trial started
    from — the error's own update is already in it. Reading those two x positions off the
    exchange curve is what turns the control gap into a behavioural price.
    """
    next_lag = list(ev['lags']).index(1)
    return [(np.nanmean(ev[key], axis=0)[next_lag], colour, label)
            for key, colour, label in (('focus_err', COL['error'], 'after error'),
                                       ('focus_corr', COL['cong'], 'after correct'))]


def exchange_panel(ax, curves_x, curves_y, ylabel, marks=(), n_grid=8, title=None):
    """
    What a given inherited control state buys — the exchange rate, averaged over sessions.

    `curves_x` / `curves_y` are one array per replicate: the focus bin centres and the
    measure in each bin, as `flanker_metrics.event_locked` returns them (`curve_x` with
    `curve_y` for accuracy or `curve_rt` for RT). The bin edges are session quantiles, so
    no two sessions share them — each curve is resampled onto a common 0–1 parameterisation
    before averaging, and the mean x is plotted against the mean y. The band is SEM across
    replicates and is dropped when there is only one, which is the single-session case
    rather than an error.

    marks : (x, colour, label) triples for the landing points worth naming — the states
            the trial after an error and after a correct trial actually inherited, so the
            gap can be read off the curve as a cost rather than left as a number.

    Both curve arguments are sequences *over replicates*, never a single curve: a caller
    with one session has to wrap it, `[ev['curve_x']]`, the way run_flanker.py's Result 4c
    already does for `curve_x` and `curve_y`. It does not wrap `curve_rt`, which rides
    along unused there, so an RT panel added to that call site needs the same wrapping —
    passing the bare array iterates its scalars and fails in `len(cy)`.
    """
    grid = np.linspace(0, 1, n_grid)
    xs = np.array([np.interp(grid, np.linspace(0, 1, len(cx)), cx) for cx in curves_x])
    ys = np.array([np.interp(grid, np.linspace(0, 1, len(cy)), cy) for cy in curves_y])
    mu_x, mu_y = xs.mean(axis=0), ys.mean(axis=0)
    ax.plot(mu_x, mu_y, color=COL['neutral'], linewidth=1.2, marker='o', markersize=2.5)
    if len(ys) > 1:
        se_y = ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])
        ax.fill_between(mu_x, mu_y - se_y, mu_y + se_y, color=COL['neutral'],
                        alpha=0.18, linewidth=0)
    for x, colour, label in marks:
        ax.axvline(x, color=colour, linewidth=0.9, linestyle='--')
        # A one-off annotation deliberately below the global tick size, so two labels fit
        # inside a paper-width panel without pushing the curve around.
        ax.text(x, ax.get_ylim()[1], ' ' + label, rotation=90, fontsize=4.5,
                color=colour, va='top')
    ax.set_xlabel('Control state inherited')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def plot_circularity(axes, ev, replicate='seeds'):
    """
    Why an error does not improve the next trial, in five panels.

    `ev` holds event-locked traces stacked over replicates — sessions at group level,
    single events within one session. Each trace is (n_replicates, n_lags); the scalar
    keys (`start_gap`, `upd_err`, `upd_corr`, `frac_err_noisy`, `frac_err_clean`,
    `dfocus_err_noisy`, `dfocus_err_clean`) are each (n_replicates,).

    The argument: the control deficit *precedes* the error (panel 1) — a trial that goes
    wrong already inherited a lower focus, so conditioning on the outcome and then reading
    the state is circular. The average correction that follows an error is real and larger
    than a correct trial's, but it is an average over two very different events: the centre
    slot's own evidence misled the model (noise-driven — attending it less genuinely lowered
    that trial's error) or clean centre evidence that the flankers simply outweighed
    (flanker-driven — attending it more is the fix). Panels 2 and 3 show the split
    (`error_diagnosis_effects`/`event_locked`'s median split on `centre_evidence`) rather
    than the pooled correction, since the two kinds teach Z in opposite directions.
    """
    lags = ev['lags']
    n_rep = ev['focus_err'].shape[0]

    def _trace(ax, arr, colour, label):
        mu = np.nanmean(arr, axis=0)
        se = (np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])) if n_rep > 1 else None
        ax.errorbar(lags, mu, yerr=se, marker='o', markersize=3.5, capsize=2,
                    linewidth=1.2, color=colour, label=label)
        return mu

    # 1. The state around the event — the whole argument in one panel.
    _trace(axes[0], ev['focus_corr'], COL['cong'], 'after a correct trial')
    mu_e = _trace(axes[0], ev['focus_err'], COL['error'], 'after an ERROR')
    axes[0].axvline(0, color='k', linewidth=0.7, alpha=0.35)
    axes[0].set_xticks(lags)
    axes[0].set_xlabel('Trial, relative to the event')
    axes[0].set_ylabel('Control state inherited\n(Z focus on target)')
    axes[0].set_title('1. The deficit comes first')
    # Headroom below the traces for the annotation, and the key inside the panel — above
    # it would sit on the title.
    lo0, hi0 = axes[0].get_ylim()
    axes[0].set_ylim(lo0 - 0.32 * (hi0 - lo0), hi0)
    compact_legend(axes[0], loc='upper left', framealpha=0.85)
    axes[0].annotate('already low before\nthe error happens', xy=(-1.0, mu_e[1]),
                     xytext=(0.04, 0.06), textcoords='axes fraction', fontsize=4.5,
                     color=COL['error'],
                     arrowprops=dict(arrowstyle='->', color=COL['error'], lw=0.6))

    # 2. What kind of error. A median split of incongruent errors on centre_evidence:
    #    noise-driven (the centre slot's own samples misled the model) vs flanker-driven
    #    (centre evidence was fine, the flankers just won). Baseline 0.5 is the split point
    #    by construction over *all* incongruent trials — a departure from it says errors are
    #    not evenly drawn from both.
    bars_with_seeds(axes[1],
                    [(ev['frac_err_noisy'], 'noise-driven\n(centre bad)', COL['error']),
                     (ev['frac_err_clean'], 'flanker-driven\n(centre ok)', COL['far_incong'])],
                    'Fraction of incongruent errors', baseline=0.5)
    axes[1].set_title('2. What kind of error')

    # 3. What each kind teaches Z. This is why panel 1's average "error update" is
    #    misleading: a flanker-driven error is exactly what an error monitor would want to
    #    correct, and does; a noise-driven error tells the model, correctly for that trial,
    #    to trust the centre less — the opposite of what the next trial needs.
    bars_with_seeds(axes[2],
                    [(ev['dfocus_err_noisy'], 'noise-driven\nerror', COL['error']),
                     (ev['dfocus_err_clean'], 'flanker-driven\nerror', COL['far_incong']),
                     (ev['upd_corr'],         'correct trial\n(avg.)', COL['cong'])],
                    'Δ Z focus (this trial\'s update)', baseline=0.0)
    axes[2].set_title('3. What each teaches Z')

    # 4. What the residual costs behaviourally — this is PIA, drawn rather than tabulated.
    keep = [i for i, l in enumerate(lags) if l != 0]      # the event trial itself is not PIA
    _trace_x = lags[keep]
    for arr, colour, label in ((ev['acc_corr'], COL['cong'], 'after a correct trial'),
                               (ev['acc_err'], COL['error'], 'after an ERROR')):
        mu = np.nanmean(arr[:, keep], axis=0)
        se = (np.nanstd(arr[:, keep], axis=0, ddof=1) / np.sqrt(arr.shape[0])) if n_rep > 1 else None
        axes[3].errorbar(_trace_x, mu, yerr=se, marker='o', markersize=3.5, capsize=2,
                         linewidth=1.2, color=colour, label=label)
    axes[3].axvspan(-0.4, 0.4, color='k', alpha=0.07)
    axes[3].set_xticks(lags)
    axes[3].set_xlabel('Trial, relative to the event')
    axes[3].set_ylabel('Accuracy')
    axes[3].set_title('4. So the next trial is worse (PIA)')

    # 5. The exchange rate between control and accuracy, with both landing points marked.
    #    Drawn by `exchange_panel`, which flanker_sweep_figures.fig_z_update calls again
    #    for the RT version of the same curve.
    exchange_panel(axes[4], ev['curve_x'], ev['curve_y'],
                   'Accuracy, incongruent trials',
                   marks=landing_marks(ev), title='5. What the gap costs')
    for ax in (axes[0], axes[3], axes[4]):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return axes


# ── Shared panel specs ────────────────────────────────────────────────────────
#
# Each returns a list of (groups, kwargs) that `bar_row` or `bar_grid` draws. Both the
# group figures and run_flanker.py build from these, so the two never drift.

def spec_post_conflict(effects):
    """
    Post-incongruent slowing and accuracy — the conflict twin of the post-error panels.

    Trial A is post-correct throughout (`flanker_metrics.post_conflict_effects` enforces
    it), so this is conflict adaptation rather than post-error adaptation; without that
    restriction the two are the same measure, because incongruent trials error more.

    Trial B is split by congruency for the reason the post-error figure splits it: a
    target-focused state helps incongruent B and hurts congruent B. The third bar in the
    RT and accuracy panels is the lag-2 cell contrast (II->I against CC->I) — same unit,
    so it sits with the lag-1 measure it should be read against rather than in a panel of
    its own. Panel 2 is the decided-only RT companion; a large gap between it and panel 1
    means the contrast is carrying non-responses rather than speed.
    """
    def g(key, label, color):
        return (_as_replicates(effects, key), label, color)

    return [
        ([g('pcs_BI', 'B incong', COL['incong']),
          g('pcs_BC', 'B cong', COL['cong']),
          g('pcs_II_vs_CC', 'II→I\nvs CC→I', COL['neutral'])],
         dict(ylabel='Post-incongruent slowing (RT)', baseline=0.0, title='PCS')),
        ([g('pcs_BI_decided', 'B incong', COL['incong']),
          g('pcs_BC_decided', 'B cong', COL['cong'])],
         dict(ylabel='Post-incongruent slowing (RT)', baseline=0.0,
              title='PCS, decided only')),
        ([g('pca_BI', 'B incong', COL['incong']),
          g('pca_BC', 'B cong', COL['cong']),
          g('pca_II_vs_CC', 'II→I\nvs CC→I', COL['neutral'])],
         dict(ylabel='Post-incongruent accuracy change', baseline=0.0, title='PCA')),
        ([g('focus_in_diff_conflict_BI', 'B incong', COL['incong']),
          g('focus_in_diff_conflict_BC', 'B cong', COL['cong'])],
         dict(ylabel='Δ inherited Z focus', baseline=0.0, title='the state behind it')),
    ]


def spec_rt_by_outcome(effects, decided=True):
    """
    RT for correct against error responses, in each congruency x distance cell.

    Human flanker errors are fast: on an incongruent trial the flankers reach threshold
    before the target does, so an error beats a correct response. Positive `fasterr` is
    that signature.

    `decided=True` reads the trials that actually crossed threshold inside the window.
    That is the default and the version SIGNATURES scores, because `rt_interp` parks a
    non-response at the trial end and errors fail to decide far more often than correct
    responses — so the pooled contrast reports censoring as much as speed. Pass
    `decided=False` for the uncensored version and read the two together.

    Three panels: the RT levels for correct trials, the same for errors, and the contrast.
    The two level panels are drawn on one shared y scale by the caller (they are the same
    quantity), which the contrast panel is not part of.
    """
    suffix = '_decided' if decided else ''
    note   = ' (decided)' if decided else ''

    def g(key, label, color):
        return (_as_replicates(effects, key), label, color)

    return [
        ([g(f'rt_{k}_corr{suffix}', lbl, COL[k]) for k, lbl in CELLS],
         dict(ylabel=f'RT (timesteps){note}', title='correct')),
        ([g(f'rt_{k}_err{suffix}', lbl, COL[k]) for k, lbl in CELLS],
         dict(ylabel=f'RT (timesteps){note}', title='errors',
              hollow=[True] * len(CELLS))),
        ([g(f'fasterr_{k}{suffix}', lbl, COL[k]) for k, lbl in CELLS],
         dict(ylabel='RT correct − RT error', baseline=0.0,
              title='+ = errors are faster')),
    ]


#: Slot groups as they are labelled on a figure, per grouping. 'geometry' is the fixed
#: slot layout; 'role' is what the slots held on that trial, which swaps with distance.
SLOT_GROUPINGS = {
    'geometry': [('centre', 'centre'), ('near', 'near\npair'), ('far', 'far\npair')],
    'role':     [('flank', 'flanker\nslots'), ('empty', 'empty\nslots')],
}

#: Congruency order inside a slot-group panel: the baseline condition first. Congruent
#: trials have nothing to be misled by, so their update is what the incongruent one is
#: read against.
CONGRUENCY_ORDER = ('cong', 'incong')

#: Shade per slot group, on the existing three-step ramp within each congruency hue —
#: pooled/mid for the centre slot, the dark near shade and the light far shade for the
#: pairs. The role groups reuse the same two shades, which is honest: 'flanker' IS the
#: near pair on a near trial and the far pair on a far one.
SLOT_SHADE = {'centre': '', 'near': 'near_', 'far': 'far_',
              'flank': 'near_', 'empty': 'far_'}


def spec_z_slot_update(effects, grouping='geometry', measure='dz'):
    """
    What a trial's update did to each slot, correct against error — two panels.

    `flanker_metrics.z_slot_effects` computes both groupings and the docstring there says
    why both are needed: a near display leaves slots 0 and 4 empty and a far display
    leaves 1 and 3, so the fixed geometry mixes "distractor" with "nothing there".

    `measure='dz'` is the change in the softmaxed gate and therefore sums to ~0 across the
    five slots — a rise at centre is necessarily a fall elsewhere, which is what the zero
    line is for. `measure='zgrad'` is the aggregated dL/dZ that drove the update, under no
    such constraint, and exists only when the run logged gradients.

    The two panels are deliberately not on a shared y scale, for the reason
    `fig_z_update` gives: an error's update is several times a correct trial's, so sharing
    flattens the correct panel onto its baseline and hides what it is there to show.
    """
    groups = SLOT_GROUPINGS[grouping]
    ylabel = ('Δ Z per slot (softmax gate)' if measure == 'dz'
              else 'dL/dZ per slot (raw gradient)')

    def bars(outcome):
        return [(_as_replicates(effects, f'{measure}_{g}_{cn}_{outcome}'),
                 f'{lbl}\n{"inc" if cn == "incong" else "con"}',
                 COL[f'{SLOT_SHADE[g]}{cn}'])
                for g, lbl in groups for cn in CONGRUENCY_ORDER]

    return [(bars('corr'), dict(ylabel=ylabel, baseline=0.0,
                                title=f'correct trials — {grouping}')),
            (bars('err'), dict(ylabel=ylabel, baseline=0.0,
                               title=f'errors — {grouping}',
                               hollow=[True] * (2 * len(groups))))]


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
