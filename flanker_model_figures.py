"""
flanker_model_figures.py — the behavioural snapshot of one model configuration.

Two suites:

  **Model card** (one configuration, e.g. spatial_steep at p_congruent=0.5)
      F1  congruency basics        accuracy and RT, congruent vs incongruent
      F2  distance                 the four cells, and near − far within each congruency
      F4  RT by outcome            RT split by correct vs error, with the decided fractions
      F5  adaptation               post-error, sequential congruency, lag contrasts
      F6  scorecard                every human signature on one axis, model vs baseline

  **Manipulation series** (how each knob moves behaviour)
      F7  proportion congruent     two models across p(congruent)
      F8  spatial gradient         four training gradients
      F9  Z decay                  the usable range, 1x-5x

  Within-trial dynamics (F3) come from flanker_sweep_figures.fig_within_trial, which the
  card calls directly rather than duplicating.

Run:
    python flanker_model_figures.py                                  # spatial_steep, p=0.5
    python flanker_model_figures.py --variant baseline --p 0.5
    python flanker_model_figures.py --card-only / --series-only

Panel primitives and loading live in flanker_figure_utils.py; the measures themselves in
flanker_metrics.py. Each seed is a subject: every bar is a mean across seeds with SEM and
one dot per seed.
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt

from plot_style import FigSize

from flanker_figure_utils import (CELLS, COL, band, bars_with_seeds, collect_effects,
                                  collect_sessions, dots_with_ci, out_dir_for, save,
                                  series, sweep_root, _stack, _stack_curve)
from flanker_metrics import SIGNATURES
from flanker_sweep_config import P_CONGRUENT_LEVELS
from flanker_sweep_figures import fig_within_trial

DEFAULT_VARIANT = 'spatial_steep'
DEFAULT_P       = 0.5

#: Training-gradient series: (variant, x = p_corr[1] − p_corr[2] gap, label)
SPATIAL_SERIES = [('spatial_flat', 0.00, 'flat'), ('baseline', 0.10, 'default'),
                  ('spatial_steep', 0.23, 'steep'), ('spatial_steeper', 0.34, 'steeper')]

#: Z_decay series, truncated at 5x. Beyond that the model is degenerate rather than
#: merely less controlled — near-incongruent accuracy falls below chance — so those
#: points would set the axis for a regime we never use.
DECAY_SERIES = [('baseline', 1, '1x'), ('decay2x', 2, '2x'), ('decay5x', 5, '5x')]


def _stamp(fig, text, variant, p, n):
    fig.suptitle(f'{text} — {variant}, p(congruent)={p}, {n} seeds', fontsize=7)


def _cell_bars(effects, prefix, suffix=''):
    """The four condition cells as bar groups, in the standard order and colours."""
    return [(_stack(effects, f'{prefix}_{k}{suffix}'), lbl, COL[k]) for k, lbl in CELLS]


def _mean_curve(curves, group, keys):
    """Average several per-seed curves — used to collapse near/far into one congruency."""
    return np.nanmean([_stack_curve(curves, group, k) for k in keys], axis=0)


# ── F1: congruency basics ─────────────────────────────────────────────────────

def fig_congruency_basics(effects, curves, out_dir, p, variant):
    """The core flanker effect: accuracy, RT, when it appears, and the RT distributions."""
    fig, axes = plt.subplots(1, 4, figsize=FigSize.row(4, panel=FigSize.large))

    bars_with_seeds(axes[0],
                    [(_stack(effects, 'acc_cong'), 'congruent', COL['cong']),
                     (_stack(effects, 'acc_incong'), 'incongruent', COL['incong'])],
                    'Accuracy', baseline=0.5, connect=True, title='Accuracy')
    bars_with_seeds(axes[1],
                    [(_stack(effects, 'rt_cong'), 'congruent', COL['cong']),
                     (_stack(effects, 'rt_incong'), 'incongruent', COL['incong'])],
                    'RT (timesteps)', connect=True, title='Reaction time')

    ts = np.arange(len(curves[0]['acc_by_ts']['near_cong']))
    for cn, label, color in [('cong', 'congruent', COL['cong']),
                             ('incong', 'incongruent', COL['incong'])]:
        band(axes[2], ts, _mean_curve(curves, 'acc_by_ts', [f'near_{cn}', f'far_{cn}']),
             label, color)
    axes[2].axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    axes[2].set_xticks(ts)
    axes[2].set_xlabel('Timestep within trial')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('When the cost appears', fontsize=6)
    axes[2].legend(fontsize=5)

    rt_x = curves[0]['rt_bins']
    for key, label, color in [('cong', 'congruent', COL['cong']),
                              ('incong', 'incongruent', COL['incong'])]:
        band(axes[3], rt_x, _stack_curve(curves, 'rt_density', key), label, color)
    axes[3].set_xlabel('RT (timesteps)')
    axes[3].set_ylabel('Density')
    axes[3].set_title('RT distribution', fontsize=6)
    axes[3].legend(fontsize=5)

    _stamp(fig, 'Congruency: the core effect', variant, p, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/card_1_congruency.pdf')


# ── F2: distance ──────────────────────────────────────────────────────────────

def fig_distance(effects, out_dir, p, variant):
    """Flanker distance: the four cells and the two directional contrasts."""
    fig, axes = plt.subplots(1, 4, figsize=FigSize.row(4, panel=FigSize.large))

    bars_with_seeds(axes[0], _cell_bars(effects, 'acc'), 'Accuracy', baseline=0.5,
                    title='Accuracy by cell')
    bars_with_seeds(axes[1], _cell_bars(effects, 'rt'), 'RT (timesteps)',
                    title='RT by cell')
    bars_with_seeds(axes[2],
                    [(_stack(effects, 'dist_effect_acc_cong'),   'congruent\n(expect +)',   COL['cong']),
                     (_stack(effects, 'dist_effect_acc_incong'), 'incongruent\n(expect −)', COL['incong'])],
                    'Near − far (accuracy)', baseline=0.0, title='Distance effect: accuracy')
    bars_with_seeds(axes[3],
                    [(_stack(effects, 'dist_effect_rt_cong'),   'congruent\n(expect −)',   COL['cong']),
                     (_stack(effects, 'dist_effect_rt_incong'), 'incongruent\n(expect +)', COL['incong'])],
                    'Near − far (RT)', baseline=0.0, title='Distance effect: RT')

    _stamp(fig, 'Flanker distance', variant, p, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/card_2_distance.pdf')


# ── F4: RT split by outcome ───────────────────────────────────────────────────

def fig_rt_by_outcome(effects, curves, out_dir, p, variant):
    """
    RT in each cell for correct and error responses separately.

    Flanker-driven errors and target-driven correct responses are different processes,
    and distance can move them in opposite directions. Panel 5 is the guard rail: RTs of
    trials that never crossed threshold are extrapolated, so a cell whose decided
    fraction is low is reporting failure-to-decide as much as speed.
    """
    fig, axes = plt.subplots(2, 3, figsize=FigSize.grid(2, 3, panel=FigSize.large))
    ax = axes.ravel()

    bars_with_seeds(ax[0], _cell_bars(effects, 'rt', '_corr'), 'RT (timesteps)',
                    title='RT — correct responses')
    bars_with_seeds(ax[1], _cell_bars(effects, 'rt', '_err'), 'RT (timesteps)',
                    title='RT — errors')

    bars_with_seeds(ax[2],
                    [(_stack(effects, 'dist_effect_rt_cong_corr'),   'cong\ncorrect',   COL['near_cong']),
                     (_stack(effects, 'dist_effect_rt_cong_err'),    'cong\nerror',     COL['far_cong']),
                     (_stack(effects, 'dist_effect_rt_incong_corr'), 'incong\ncorrect', COL['near_incong']),
                     (_stack(effects, 'dist_effect_rt_incong_err'),  'incong\nerror',   COL['far_incong'])],
                    'Near − far (RT)', baseline=0.0, title='Distance × outcome')

    # Distance leaves the RT distribution almost unchanged, so near/far are collapsed
    # here and the panel shows the split that does separate: congruency x outcome.
    rt_x = curves[0]['rt_bins']
    for cn, on, label, color, ls in [('cong', 'corr', 'congruent, correct', COL['cong'], '-'),
                                     ('cong', 'err', 'congruent, error', COL['cong'], '--'),
                                     ('incong', 'corr', 'incongruent, correct', COL['incong'], '-'),
                                     ('incong', 'err', 'incongruent, error', COL['incong'], '--')]:
        keys = [f'near_{cn}_{on}', f'far_{cn}_{on}']
        band(ax[3], rt_x, _mean_curve(curves, 'rt_density', keys), label, color, linestyle=ls)
    ax[3].set_xlabel('RT (timesteps)')
    ax[3].set_ylabel('Density')
    ax[3].set_title('RT distribution by outcome', fontsize=6)
    ax[3].legend(fontsize=5)

    # Decided fraction: bars for errors, open markers for the matching correct cells.
    bars_with_seeds(ax[4], _cell_bars(effects, 'dec', '_err'),
                    'Crossed threshold in window', baseline=1.0,
                    title='Decided fraction (bars: errors)')
    for j, (key, _) in enumerate(CELLS):
        vals = _stack(effects, f'dec_{key}_corr')
        ax[4].scatter([j], [np.nanmean(vals)], s=14, facecolors='none', edgecolors='k',
                      linewidths=0.7, zorder=6)
    ax[4].set_ylim(0, 1.05)

    bars_with_seeds(ax[5], _cell_bars(effects, 'fasterr'), 'RT correct − RT error',
                    baseline=0.0, title='Positive = errors are faster')

    _stamp(fig, 'RT by outcome and distance', variant, p, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/card_4_rt_by_outcome.pdf')


# ── F5: adaptation ────────────────────────────────────────────────────────────

def fig_adaptation(effects, out_dir, p, variant):
    """Trial-to-trial adaptation: post-error, sequential congruency, and which lag."""
    fig, axes = plt.subplots(1, 4, figsize=FigSize.row(4, panel=FigSize.large))

    bars_with_seeds(axes[0],
                    [(_stack(effects, 'pes_BI'), 'slowing\nB incong', COL['incong']),
                     (_stack(effects, 'pes_BC'), 'slowing\nB cong',   COL['cong']),
                     (_stack(effects, 'peri'),   'PERI',              COL['neutral'])],
                    'Post-error effect (RT)', baseline=0.0, title='Post-error slowing / PERI')
    bars_with_seeds(axes[1],
                    [(_stack(effects, 'pia_BI'), 'B incong', COL['incong']),
                     (_stack(effects, 'pia_BC'), 'B cong',   COL['cong'])],
                    'Post-error accuracy change', baseline=0.0, title='Post-error accuracy')
    bars_with_seeds(axes[2],
                    [(_stack(effects, 'sce_acc_switch'), 'acc\nswitch', COL['cong']),
                     (_stack(effects, 'sce_acc_repeat'), 'acc\nrepeat', COL['incong']),
                     (_stack(effects, 'sce_rt_switch'),  'RT\nswitch',  COL['far_cong']),
                     (_stack(effects, 'sce_rt_repeat'),  'RT\nrepeat',  COL['far_incong'])],
                    'Sequential congruency effect', baseline=0.0,
                    title='Gratton vs. repetition priming')
    bars_with_seeds(axes[3],
                    [(_stack(effects, 'lag1_contrast_acc'), 'lag 1', COL['cong']),
                     (_stack(effects, 'lag2_contrast_acc'), 'lag 2', COL['incong'])],
                    'History contrast (accuracy)', baseline=0.0, connect=True,
                    title='Which lag carries it')

    _stamp(fig, 'Trial-to-trial adaptation', variant, p, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/card_5_adaptation.pdf')


# ── F6: benchmark scorecard ───────────────────────────────────────────────────

def _effect_size(values, sign):
    """Cohen's d across seeds, flipped so positive always means human-consistent."""
    v = np.asarray(values, dtype=float) * sign
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.array([])
    return v / v.std(ddof=1)


def fig_scorecard(effects, baseline_effects, out_dir, p, variant, extra_rows=()):
    """
    Every human signature on one axis: does this model reproduce it?

    Each effect is divided by its across-seed SD, so accuracy, RT and latent measures
    become comparable, and multiplied by the sign a human dataset shows — positive is
    always "matches humans". Filled markers are the model, open markers the shallow-gradient
    baseline, so the panel doubles as "what did the training gradient buy?". The count on
    the right is seeds with the predicted sign, which is the honest summary when a group
    mean rests on two outliers.
    """
    rows = list(SIGNATURES) + list(extra_rows)
    fig, ax = plt.subplots(figsize=FigSize.custom(3.4, 0.22 * len(rows) + 0.7))

    for i, (key, label, sign, _source) in enumerate(rows):
        y = len(rows) - i
        if baseline_effects and key in baseline_effects[0]:
            dots_with_ci(ax, y + 0.18, _effect_size(_stack(baseline_effects, key), sign),
                         COL['neutral'], filled=False)
        vals = _stack(effects, key) * sign
        vals = vals[~np.isnan(vals)]
        d    = _effect_size(_stack(effects, key), sign)
        mean, sem = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        ok   = int((vals > 0).sum())
        if mean - 1.96 * sem > 0:
            color = COL['pass_']
        elif mean + 1.96 * sem < 0:
            color = COL['fail']
        else:
            color = COL['null']
        dots_with_ci(ax, y - 0.18, d, color)
        ax.text(1.02, y, f'{ok}/{len(vals)}', transform=ax.get_yaxis_transform(),
                fontsize=5, va='center', color=color)

    ax.axvline(0, color='k', linewidth=0.7, alpha=0.6)
    ax.set_yticks([len(rows) - i for i in range(len(rows))])
    ax.set_yticklabels([lbl for _, lbl, _, _ in rows], fontsize=5)
    ax.set_ylim(0.3, len(rows) + 0.9)
    ax.set_xlabel("Effect size across seeds (signed so + = matches humans)")
    ax.set_title(f'{variant} (filled) vs baseline (open), p(congruent)={p}', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return save(fig, f'{out_dir}/card_6_scorecard.pdf')


# ── F7-F9: manipulation series ────────────────────────────────────────────────

def _series_arrays(conditions, keys):
    """Load a list of (variant, p) conditions; return {key: (n_levels, n_seeds)}."""
    per_level = [collect_effects(p, v) for v, p in conditions]
    return {k: np.array([_stack(e, k) for e in per_level]) for k in keys}


def fig_proportion_series(out_dir, variant, reference='baseline'):
    """Congruency and distance effects across p(congruent), for two model sets."""
    keys = ['cong_effect_acc', 'cong_effect_rt', 'dist_effect_acc_incong']
    data = {name: _series_arrays([(name, p) for p in P_CONGRUENT_LEVELS], keys)
            for name in (reference, variant)}

    fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3, panel=FigSize.large))
    labels = ['Congruency effect (accuracy)', 'Congruency effect (RT)',
              'Near − far (acc, incongruent)']
    for ax, key, ylabel in zip(axes, keys, labels):
        series(ax, P_CONGRUENT_LEVELS, data[reference][key], reference, COL['neutral'])
        series(ax, P_CONGRUENT_LEVELS, data[variant][key], variant, COL['incong'],
               seed_lines=True)
        ax.axhline(0, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
        ax.set_xlabel('P(congruent)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(P_CONGRUENT_LEVELS)
    axes[0].legend(fontsize=5)

    fig.suptitle('List-wide proportion congruent — thin lines are subjects', fontsize=7)
    fig.tight_layout()
    return save(fig, f'{out_dir}/series_proportion_congruent.pdf')


def fig_spatial_series(out_dir, p=DEFAULT_P):
    """How the distance effect, the congruency effect and attention scale with training."""
    gaps = [g for _, g, _ in SPATIAL_SERIES]
    keys = ['dist_effect_acc_cong', 'dist_effect_acc_incong', 'cong_effect_acc',
            'acc_overall', 'acc_near_incong', 'att_centre', 'att_near', 'att_far']
    data = _series_arrays([(v, p) for v, _, _ in SPATIAL_SERIES], keys)

    fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3, panel=FigSize.large))

    series(axes[0], gaps, data['dist_effect_acc_cong'], 'congruent', COL['cong'])
    series(axes[0], gaps, data['dist_effect_acc_incong'], 'incongruent', COL['incong'],
           seed_lines=True)
    axes[0].axhline(0, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    axes[0].set_ylabel('Near − far (accuracy)')
    axes[0].set_title('Distance effect scales with training', fontsize=6)
    axes[0].legend(fontsize=5)

    series(axes[1], gaps, data['cong_effect_acc'], 'congruency effect', COL['incong'])
    series(axes[1], gaps, data['acc_overall'], 'overall accuracy', COL['neutral'])
    series(axes[1], gaps, data['acc_near_incong'], 'near-incongruent acc', COL['near_incong'])
    axes[1].axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    axes[1].set_ylabel('Accuracy / effect size')
    axes[1].set_title('The manipulation is not selective', fontsize=6)
    # Headroom above the accuracy line, so the legend does not sit on the data.
    axes[1].set_ylim(top=1.0)
    axes[1].legend(fontsize=5, loc='upper right', framealpha=0.85)

    for key, label, color in [('att_centre', 'centre', COL['neutral']),
                              ('att_near', 'near', COL['near_incong']),
                              ('att_far', 'far', COL['far_incong'])]:
        series(axes[2], gaps, data[key], label, color)
    axes[2].axhline(0.2, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    axes[2].set_ylabel('Attention weight (softmax)')
    axes[2].set_title('Attention barely reallocates', fontsize=6)
    axes[2].legend(fontsize=5)

    for ax in axes:
        ax.set_xlabel('Training gradient, p_corr[1] − p_corr[2]')
        ax.set_xticks(gaps)
    fig.suptitle(f'Spatial gradient during training — p(congruent)={p}', fontsize=7)
    fig.tight_layout()
    return save(fig, f'{out_dir}/series_spatial_gradient.pdf')


def fig_decay_series(out_dir, p=DEFAULT_P, matched_level=0.8):
    """
    Z_decay across its usable range, against the matched-focus reference.

    Decay pulls the latent toward zero, and zero is a uniform softmax — it sets *where*
    the control state settles. The dashed line is p(congruent)=0.8, which reaches the same
    focus by a different route; where the two disagree, control magnitude is not what
    matters.
    """
    xs   = [x for _, x, _ in DECAY_SERIES]
    keys = ['focus_all', 'cong_effect_acc', 'dist_effect_acc_incong']
    data = _series_arrays([(v, p) for v, _, _ in DECAY_SERIES], keys)
    ref  = collect_effects(matched_level, 'baseline')

    fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3, panel=FigSize.large))
    labels = ['Z focus (control state)', 'Congruency effect (accuracy)',
              'Near − far (acc, incongruent)']
    for ax, key, ylabel in zip(axes, keys, labels):
        series(ax, xs, data[key], 'Z_decay series', COL['incong'], seed_lines=True)
        if ref:
            ax.axhline(np.nanmean(_stack(ref, key)), color=COL['cong'], linewidth=0.8,
                       linestyle='--', label=f'p(cong)={matched_level}')
        ax.set_xlabel('Z_decay (x baseline)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, _, lbl in DECAY_SERIES])
    axes[2].axhline(0, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    axes[0].legend(fontsize=5)

    fig.suptitle(f'Z_decay — usable range only, p(congruent)={p}', fontsize=7)
    fig.tight_layout()
    return save(fig, f'{out_dir}/series_z_decay.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def _with_pc_modulation(effects, variant, low=min(P_CONGRUENT_LEVELS), high=max(P_CONGRUENT_LEVELS)):
    """
    Add the list-wide proportion-congruent modulation as a per-seed effect.

    The same pretrained model is tested at every level, so this is a within-subject
    contrast: the congruency effect at p=high minus the same seed's effect at p=low.
    Returns the extra scorecard row, or None when the levels are not on disk.
    """
    lo, hi = collect_effects(low, variant), collect_effects(high, variant)
    if len(lo) != len(effects) or len(hi) != len(effects):
        return None
    diff = _stack(hi, 'cong_effect_acc') - _stack(lo, 'cong_effect_acc')
    for e, d in zip(effects, diff):
        e['pc_modulation'] = float(d)
    return ('pc_modulation', f'Proportion congruent ({high}−{low})', +1, 'Logan & Zbrodoff 1979')


def build_card(variant=DEFAULT_VARIANT, p=DEFAULT_P, reference='baseline'):
    out_dir = out_dir_for(p, variant)
    print(f'Model card: variant={variant}, p_congruent={p}')
    effects, curves = collect_sessions(p, variant)
    if not effects:
        print('No results found — run flanker_sweep.py first.')
        return
    baseline_effects = collect_effects(p, reference) if reference != variant else []

    fig_congruency_basics(effects, curves, out_dir, p, variant)
    fig_distance(effects, out_dir, p, variant)
    fig_within_trial(curves, out_dir, p, variant)
    fig_rt_by_outcome(effects, curves, out_dir, p, variant)
    fig_adaptation(effects, out_dir, p, variant)

    extra = _with_pc_modulation(effects, variant)
    if extra and baseline_effects:
        _with_pc_modulation(baseline_effects, reference)
    fig_scorecard(effects, baseline_effects, out_dir, p, variant,
                  extra_rows=[extra] if extra else ())


def build_series(variant=DEFAULT_VARIANT, p=DEFAULT_P):
    out_dir = sweep_root()
    print('Manipulation series')
    fig_proportion_series(out_dir, variant)
    fig_spatial_series(out_dir, p)
    fig_decay_series(out_dir, p)


def main(variant=DEFAULT_VARIANT, p=DEFAULT_P, card=True, do_series=True):
    if card:
        build_card(variant, p)
    if do_series:
        build_series(variant, p)


def args_from_argv():
    """`--variant <name> --p <level> [--card-only|--series-only]`, ignoring kernel args."""
    variant, p, card, do_series, argv = DEFAULT_VARIANT, DEFAULT_P, True, True, sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == '--variant' and i + 1 < len(argv):
            variant = argv[i + 1]
        elif arg == '--p' and i + 1 < len(argv):
            p = float(argv[i + 1])
        elif arg == '--card-only':
            do_series = False
        elif arg == '--series-only':
            card = False
    return variant, p, card, do_series


if __name__ == '__main__':
    main(*args_from_argv())
