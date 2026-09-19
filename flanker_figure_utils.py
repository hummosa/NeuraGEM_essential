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
    flanker_run_one_network.py             the single-session workbench, for the panels they share

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
from plot_style import (FLANKER_CELLS, FLANKER_CELLS_BY_DISTANCE, FLANKER_COLORS,
                        FigSize, flanker_color, outcome_style)

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
#: Cell order matching the human paper's figure — distance outer, congruency inner.
CELLS_BY_DISTANCE = FLANKER_CELLS_BY_DISTANCE


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


def _flag(effects, key):
    """A regime flag across replicates — True only when every replicate agrees.

    The flags (`dgain_varies`) are stored as floats because the per-seed effects dict is
    flat and numeric. Unanimity is the right rule: a mixed sweep would mean seeds were run
    under different latent activations, which is a configuration error, not a figure to
    draw half of.
    """
    return bool(np.all(_as_replicates(effects, key) > 0.5))


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

    out = {'acc_by_ts': {}, 'accum': {}, 'rt_density': {}, 'learning': {},
           'target_delay': trials.get('target_delay', 0)}

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

#: Extra x gap, in bar widths, inserted at each `group_spacing` index.
_BAR_GAP = 0.75


def _bar_positions(n, group_spacing=None):
    """Bar x positions, with an extra gap before each index in `group_spacing`."""
    gaps, x, pos = set(group_spacing or []), [], 0.0
    for i in range(n):
        if i and i in gaps:
            pos += _BAR_GAP
        x.append(pos)
        pos += 1.0
    return np.array(x)


def bars_with_seeds(ax, groups, ylabel, baseline=None, connect=False, rotation=0,
                    title=None, hollow=None, ylim=None, group_spacing=None,
                    super_labels=None):
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
    group_spacing : optional list of indices; an extra gap is inserted before each, the
              same idiom `flanker_analyses.plot_scalar_bars` uses. Use it to block a
              panel into two comparable halves rather than splitting it into two axes —
              bars separated by a panel boundary cannot be read against each other by
              height, which is the whole job of a level panel.
    super_labels : optional list of (label, first_index, last_index). Draws a rule under
              the tick labels spanning those bars with `label` beneath it, so a panel can
              carry two levels of grouping on one axis — the per-bar condition on the
              tick labels and the block on the rule. Follows the human paper's figure.
    """
    x      = _bar_positions(len(groups), group_spacing)
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

    # The second level of grouping, drawn below the tick labels. x is in data coordinates
    # so the rule tracks the bars, y is a fixed offset in POINTS below the axis — not an
    # axes fraction, which would have to guess how tall the tick labels are and lands the
    # rule through them the moment the panel height or the font size changes.
    if super_labels:
        from matplotlib.transforms import offset_copy
        fs      = float(plt.rcParams['xtick.labelsize'])
        n_lines = max(len(str(lbl).split('\n')) for _, lbl, _ in groups)
        drop    = float(ax.xaxis.get_tick_padding()) + 1.35 * fs * n_lines + 2.0
        tr      = ax.get_xaxis_transform()
        rule_tr = offset_copy(tr, fig=ax.figure, y=-drop, units='points')
        text_tr = offset_copy(tr, fig=ax.figure, y=-(drop + 2.0), units='points')
        for lbl, i0, i1 in super_labels:
            ax.plot([x[i0] - 0.45, x[i1] + 0.45], [0, 0], transform=rule_tr,
                    color='k', linewidth=0.8, clip_on=False, zorder=6)
            ax.text(0.5 * (x[i0] + x[i1]), 0, lbl, transform=text_tr,
                    ha='center', va='top', clip_on=False)

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


def bar_panel_width(groups, margin=0.55, min_slot=0.28, group_spacing=None,
                    super_labels=None):
    """Paper-ready width for one bar panel: enough for the labels it has to carry.

    Tick labels set the slot width, but a `super_labels` rule spans a block of bars and can
    be wider than the block, so it has to widen the slot too — a two-bar block under
    "after an incongruent error" is the case that overflows. Dividing the label by its
    block length is what makes this a no-op for a panel whose super label already fits
    (`spec_rt_by_outcome`'s "Correct" over four bars), so no existing figure changes size.
    """
    slot = max(min_slot, max(_label_width(lbl) for _, lbl, _ in groups))
    for lbl, i0, i1 in (super_labels or []):
        slot = max(slot, _label_width(lbl) / (i1 - i0 + 1))
    n_gaps = len(set(group_spacing or []) - {0})
    return margin + slot * (len(groups) + _BAR_GAP * n_gaps)


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
    widths = [bar_panel_width(groups, group_spacing=kw.get('group_spacing'),
                              super_labels=kw.get('super_labels'))
              for groups, kw in panels]
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
    widths = [max(bar_panel_width(groups, group_spacing=kw.get('group_spacing'),
                                  super_labels=kw.get('super_labels'))
                  for row in rows for j, (groups, kw) in enumerate(row) if j == i)
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
    with one session has to wrap it, `[ev['curve_x']]`, the way flanker_run_one_network.py's Result 4c
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
    Why an error does not improve the next trial, in five panels — or six.

    Pass a sixth axis to get the RT companion to panel 5: the same exchange curve read in
    RT instead of accuracy. It is opt-in so that `fig_circularity` (group_6) keeps its
    five-panel layout, while a caller that wants the pair in one figure can ask for it.
    A six-axis caller must also wrap `curve_rt` over replicates, exactly as it wraps
    `curve_x` and `curve_y` — see `exchange_panel`.

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
    #    Drawn by `exchange_panel`, which flanker_sweep_figures.fig_control_axes uses for
    #    the same curve along each of the two control axes.
    marks = landing_marks(ev)
    exchange_panel(axes[4], ev['curve_x'], ev['curve_y'],
                   'Accuracy, incongruent trials',
                   marks=marks, title='5. What the gap costs')
    styled = [axes[0], axes[3], axes[4]]

    # 6. The same exchange rate in RT. A control account says the gap has to be paid for in
    #    speed as well as in accuracy, so the two panels are only informative together: a
    #    model that charges for control in one currency and not the other is not
    #    reproducing the human trade-off. Drawn only when a sixth axis is supplied, and
    #    only when the caller wrapped `curve_rt` over replicates the way exchange_panel
    #    requires — group_13's first row draws this same pair across seeds.
    if len(axes) > 5 and 'curve_rt' in ev:
        exchange_panel(axes[5], ev['curve_x'], ev['curve_rt'],
                       'RT (timesteps), incongruent trials',
                       marks=marks, title='6. And what it costs in RT')
        styled.append(axes[5])

    for ax in styled:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return axes


def control_plane(ax, per_session, measure='acc', n_bins=7, cmap=None, label=None):
    """
    The (focus, gain) plane of the inherited gate, with behaviour as the landscape.

    `per_session` is a list of `flanker_metrics.control_axes` dicts — one per replicate.
    Trials are pooled across replicates to bin the plane, because per-session quantile
    grids are not comparable cell for cell; the landing points and the arrow are averaged
    across replicates instead, so the summary that carries the argument stays across-seed.

    The arrow runs from the state a trial after a CORRECT trial inherited to the state a
    trial after an ERROR inherited. Reading the arrow against the landscape is the whole
    point of the panel: its two components are priced differently, and on the accuracy
    map they push in opposite directions while on the RT map they add. A one-dimensional
    exchange curve cannot show that, which is how the post-error contrasts came to look
    self-contradictory.

    Bins are quantile edges on the pooled data, so cells hold comparable numbers of trials
    rather than comparable areas; cells with too few trials are left blank.
    """
    ok = [s for s in per_session if s.get('ok')]
    if not ok:
        ax.set_visible(False)
        return None
    fo = np.concatenate([s['focus'] for s in ok])
    ga = np.concatenate([s['gain'] for s in ok])
    zz = np.concatenate([s[measure] for s in ok])

    fe = np.nanquantile(fo, np.linspace(0.02, 0.98, n_bins + 1))
    ge = np.nanquantile(ga, np.linspace(0.02, 0.98, n_bins + 1))
    grid = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            sel = ((fo >= fe[i]) & (fo < fe[i + 1])
                   & (ga >= ge[j]) & (ga < ge[j + 1]))
            if sel.sum() >= 25:
                grid[j, i] = np.nanmean(zz[sel])
    mesh = ax.pcolormesh(fe, ge, grid, cmap=cmap or ('viridis' if measure == 'acc' else 'magma_r'),
                         shading='flat', rasterized=True)
    cb = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=4.5)
    cb.set_label(label or ('Accuracy' if measure == 'acc' else 'RT (timesteps)'), fontsize=5)

    fc = float(np.nanmean([s['focus_corr'] for s in ok]))
    gc = float(np.nanmean([s['gain_corr'] for s in ok]))
    fer = float(np.nanmean([s['focus_err'] for s in ok]))
    ger = float(np.nanmean([s['gain_err'] for s in ok]))
    # White halo under the arrow and markers: the landscape runs to near-black at the slow
    # corner of the RT map, where a plain black arrow disappears.
    ax.annotate('', xy=(fer, ger), xytext=(fc, gc),
                arrowprops=dict(arrowstyle='-|>', linewidth=2.6, color='w',
                                shrinkA=0, shrinkB=0, alpha=0.9))
    ax.annotate('', xy=(fer, ger), xytext=(fc, gc),
                arrowprops=dict(arrowstyle='-|>', linewidth=1.1, color='k',
                                shrinkA=0, shrinkB=0))
    for x, y, colour, lbl in ((fc, gc, COL['cong'], 'after correct'),
                              (fer, ger, COL['incong'], 'after error')):
        ax.plot([x], [y], 'o', ms=4.2, color=colour, zorder=5, label=lbl,
                markeredgecolor='w', markeredgewidth=0.7)

    # What the displacement actually costs, so the arrow ties back to the scorecard row it
    # explains rather than being read as a direction only.
    delta = float(np.nanmean([s[f'{measure}_err'] - s[f'{measure}_corr'] for s in ok]))
    name = 'PIA' if measure == 'acc' else 'PES'
    ax.text(0.03, 0.03, f'{name} = {delta:+.3f}', transform=ax.transAxes, fontsize=5,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='none', alpha=0.75))

    ax.set_xlabel('focus_in  (centre − flankers)')
    ax.set_ylabel('gain  (mean gate weight)')
    compact_legend(ax, loc='upper left')
    return mesh


# ── Shared panel specs ────────────────────────────────────────────────────────
#
# Each returns a list of (groups, kwargs) that `bar_row` or `bar_grid` draws. Both the
# group figures and flanker_run_one_network.py build from these, so the two never drift.

def spec_post_error(effects):
    """
    Post-error effects, with trial A's congruency as a visible factor.

    Every bar is post-error MINUS post-correct within one (A, B) cell, so a positive PES
    bar is slower after an error and a positive PIA bar is more accurate after one. The
    ylabels say the subtraction rather than naming the acronym, because "PIA" alone does
    not tell a reader which way is which.

    Two factors, two visual channels: trial B's congruency is the hue (the house code —
    blue congruent, red incongruent) and trial A's congruency is the rule beneath the tick
    labels. A has no hue of its own to spend: hue is congruency, shade is distance and fill
    is outcome already, so `super_labels` is where a third grouping goes.

    Why both factors are crossed rather than pooled is in
    `flanker_metrics.post_error_effects`: pooling B averages an effect against its own
    opposite, and pooling A makes the post-correct baseline a mixture. The A-incongruent
    column is what the scorecard scores as `pes_BI` / `pia_BI` / `peri`.
    """
    def g(key, label, color):
        return (_as_replicates(effects, key), label, color)

    def ab(stem):
        return [g(f'{stem}_AI_BI', 'B incong', COL['incong']),
                g(f'{stem}_AI_BC', 'B cong',   COL['cong']),
                g(f'{stem}_AC_BI', 'B incong', COL['incong']),
                g(f'{stem}_AC_BC', 'B cong',   COL['cong'])]

    blocks = dict(group_spacing=[2],
                  super_labels=[('after an\nincongruent error', 0, 1),
                                ('after a\ncongruent error',    2, 3)])
    return [
        (ab('pes'), dict(ylabel='RT: after error − after correct', baseline=0.0,
                         title='PES  (+ = slower after an error)', **blocks)),
        (ab('pia'), dict(ylabel='Accuracy: after error − after correct', baseline=0.0,
                         title='PIA  (+ = more accurate after an error)', **blocks)),
        ([g('peri_AI', 'A incong', COL['incong']),
          g('peri_AC', 'A cong',   COL['cong'])],
         dict(ylabel='Congruency effect on RT:\nafter correct − after error', baseline=0.0,
              title='PERI  (+ = interference drops)')),
        (ab('focus_in_diff'),
         dict(ylabel='Inherited Z focus:\nafter error − after correct', baseline=0.0,
              title='the state behind it', **blocks)),
    ]


#: The two axes of the control update, and what each one is blind to. Mirrors the
#: focus/gain split `control_axes` applies to the inherited state — see
#: `flanker_metrics.control_effects`.
CONTROL_UPDATE = {
    'dfocus': ('Δ focus this trial\n(centre − flankers)', 'where the update points the gate'),
    'dgain':  ('Δ gain this trial\n(mean over all slots)', 'how hard the update gates'),
}


def spec_control_update(effects, measure='dfocus'):
    """
    What one trial teaches the gate, by condition cell and outcome.

    Returns ONE panel, so callers compose: `spec_control_update(e, 'dfocus') +
    spec_control_update(e, 'dgain')` is the two-panel figure, and a caller in the softmax
    regime simply omits the gain half (there `delta_z` sums to ~0 across slots, so Δ gain
    is identically zero — `dgain_varies` reports it).

    Correct and error share one axis here, deliberately. They used to be two panels on
    separate scales, on the argument that an error's update is several times a correct
    trial's and would flatten it; measured, the ratio is ~3-4x, so the correct bars land at
    roughly a quarter height — readable, and now comparable by height with the errors,
    which two panels could never be. Errors are hollow, the house convention for an error
    cell, so outcome costs no hue.

    Grouped in `CELLS` order (congruency first), because the contrast this panel exists for
    is congruent-vs-incongruent, not the bar-for-bar read against the human figure that
    `CELLS_BY_DISTANCE` serves.
    """
    ylabel, title = CONTROL_UPDATE[measure]
    n = len(CELLS)
    bars = ([(_as_replicates(effects, f'{measure}_{k}_corr'), lbl, COL[k])
             for k, lbl in CELLS]
            + [(_as_replicates(effects, f'{measure}_{k}_err'), lbl, COL[k])
               for k, lbl in CELLS])
    return [(bars, dict(ylabel=ylabel, title=title, baseline=0.0,
                        hollow=[False] * n + [True] * n,
                        group_spacing=[n],
                        super_labels=[('Correct', 0, n - 1), ('Error', n, 2 * n - 1)]))]


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

    **Laid out to be read bar-for-bar against the human figure.** Two things follow from
    that and neither is the house default elsewhere. The cells run in
    `CELLS_BY_DISTANCE` order — distance outer, congruency inner — rather than the
    congruency-first `CELLS` order the rest of the panels use, because that is the order
    the published figure plots. And correct and error sit in one axes separated by a gap
    rather than in two panels: bars split across a panel boundary cannot be compared by
    height, which is the only thing a level panel is for. Outcome is named on the rule
    beneath the tick labels, and still rides on fill as well (hollow = error, the house
    convention) so the two halves stay separable if the figure is cropped.

    Two panels: the eight RT levels, and the contrast that SIGNATURES scores. They are
    different quantities, so they do not share a y scale.
    """
    suffix = '_decided' if decided else ''
    note   = ' (decided)' if decided else ''
    cells  = CELLS_BY_DISTANCE

    def g(key, label, color):
        return (_as_replicates(effects, key), label, color)

    levels = ([g(f'rt_{k}_corr{suffix}', lbl, COL[k]) for k, lbl in cells]
              + [g(f'rt_{k}_err{suffix}', lbl, COL[k]) for k, lbl in cells])
    n = len(cells)

    return [
        (levels,
         dict(ylabel=f'RT (timesteps){note}',
              hollow=[False] * n + [True] * n,
              group_spacing=[n],
              super_labels=[('Correct', 0, n - 1), ('Error', n, 2 * n - 1)])),
        ([g(f'fasterr_{k}{suffix}', lbl, COL[k]) for k, lbl in cells],
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

    The two panels are deliberately not on a shared y scale: an error's update is several
    times a correct trial's, so sharing flattens the correct panel onto its baseline and
    hides what it is there to show. (`spec_control_update`, which does share an axis across
    outcome, is measuring a coarser quantity — one number per trial, not five — where the
    3-4x ratio still leaves the correct bars readable.)
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
