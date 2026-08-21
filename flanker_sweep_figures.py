"""
flanker_sweep_figures.py — the standing group figures for one condition.

Every panel shows the mean across seeds with SEM error bars, plus one dot per seed so
the spread is visible rather than hidden inside an error bar. Where a comparison is
within-subject, thin grey lines connect the same seed across conditions — that is the
contrast the statistics actually test.

Panel primitives and loading live in flanker_figure_utils.py; the model-card and
manipulation-series figures live in flanker_model_figures.py.

Run:
    python flanker_sweep_figures.py                          # baseline, p_congruent=0.5
    python flanker_sweep_figures.py 0.2                      # a different proportion
    python flanker_sweep_figures.py --p 0.5 --variant spatial_steep

Requires flanker_sweep.py to have been run first.
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt

from plot_style import FigSize

from flanker_figure_utils import (CELLS, COL, band, bar_row, collect_sessions,
                                  compact_legend, out_dir_for, pretrain_curves, save,
                                  series, share_ylim, sweep_root, _stack, _stack_curve)
from flanker_sweep_config import P_CONGRUENT_LEVELS
from flanker_figure_utils import bar_row
DEFAULT_P = 0.5
DEFAULT_VARIANT = 'baseline'

#: Sweep run to read and write. None follows flanker_sweep_config.RUN_NAME;
#: see flanker_sweep.SWEEP_RUNS for what each run contains.
RUN = None


def _stamp(fig, text, variant, p, n):
    label = '' if variant == 'baseline' else f'{variant}, '
    fig.suptitle(f'{text} — {label}p(congruent)={p}, {n} seeds')


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_fingerprint(effects, out_dir, p, variant=DEFAULT_VARIANT):
    """Accuracy and RT in the four cells, plus the distance effects decomposed."""
    fig, axes = bar_row([
        ([(_stack(effects, f'acc_{k}'), lbl, COL[k]) for k, lbl in CELLS],
         dict(ylabel='Accuracy', baseline=0.5)),
        ([(_stack(effects, f'rt_{k}'), lbl, COL[k]) for k, lbl in CELLS],
         dict(ylabel='RT (timesteps)')),
        ([(_stack(effects, 'cong_effect_acc_near'), 'near', COL['near_incong']),
          (_stack(effects, 'cong_effect_acc_far'),  'far',  COL['far_incong'])],
         dict(ylabel='Congruency effect (accuracy)', baseline=0.0, connect=True)),
        ([(_stack(effects, 'dist_effect_acc_cong'),   'congruent\n(expect +)',   COL['cong']),
          (_stack(effects, 'dist_effect_acc_incong'), 'incongruent\n(expect −)', COL['incong'])],
         dict(ylabel='Near − far (accuracy)', baseline=0.0)),
    ])

    _stamp(fig, 'Behavioural fingerprint', variant, p, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_fingerprint.pdf')


def fig_learning_and_rt(curves, out_dir, p, variant=DEFAULT_VARIANT):
    """Learning over training, accuracy over the frozen session, and RT distributions."""
    fig, axes = plt.subplots(1, 4, figsize=FigSize.row(4, panel=FigSize.wide))

    # Short keys here: the legend has to fit in the thin empty strip along the bottom.
    acc_keys = [('cong', 'cong', COL['cong']), ('incong', 'incong', COL['incong']),
                ('all', 'all', 'k')]

    x_pre, pre = pretrain_curves(variant)
    if x_pre is not None:
        for key, label, color in acc_keys:
            if key in pre:
                band(axes[0], x_pre, pre[key], label, color)
        axes[0].axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
        axes[0].set_xlabel('Training trial')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Stage 1: plastic')
        # One flat row along the bottom strip, the only part of this panel with no data.
        compact_legend(axes[0], loc='lower center', ncol=3, columnspacing=0.6)

    x_test = curves[0]['learning_x']
    for key, label, color in acc_keys:
        band(axes[1], x_test, _stack_curve(curves, 'learning', key), label, color)
    axes[1].axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Test trial')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Stage 2: frozen')
    if x_pre is None:                        # otherwise stage 1 already carries the key
        compact_legend(axes[1], loc='lower right')

    rt_x = curves[0]['rt_bins']
    for ax, keys in [(axes[2], [('cong', 'congruent', COL['cong']),
                                ('incong', 'incongruent', COL['incong'])]),
                     (axes[3], [('correct', 'correct', COL['correct']),
                                ('error', 'error', COL['error'])])]:
        for key, label, color in keys:
            band(ax, rt_x, _stack_curve(curves, 'rt_density', key), label, color)
        ax.set_xlabel('RT (timesteps)')
        ax.set_ylabel('Density')
        compact_legend(ax, loc='upper center')

    # Both accuracy panels on one scale (stage 1 vs stage 2 is the comparison), and both
    # RT densities on another, so the two halves each read as a single axis.
    if x_pre is not None:
        share_ylim(axes[0], axes[1])
    share_ylim(axes[2], axes[3])

    _stamp(fig, 'Learning and reaction times', variant, p, len(curves))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_learning_and_rt.pdf')


def fig_within_trial(curves, out_dir, p, variant=DEFAULT_VARIANT):
    """Accuracy and evidence accumulation across timesteps within a trial."""
    fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3, panel=FigSize.wide))
    ad = len(curves[0]['acc_by_ts']['near_cong'])
    ts = np.arange(ad)

    styles = {'near_cong': '-', 'far_cong': '--', 'near_incong': '-', 'far_incong': '--'}
    for key, lbl in CELLS:
        band(axes[0], ts, _stack_curve(curves, 'acc_by_ts', key),
             lbl.replace('\n', '-'), COL[key], linestyle=styles[key])
    axes[0].axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    axes[0].set_xticks(ts)
    axes[0].set_xlabel('Timestep within trial')
    axes[0].set_ylabel('Accuracy')
    compact_legend(axes[0], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2,
                   columnspacing=0.9)

    for key, lbl, color, ls in [('correct_cong', 'correct, cong', COL['cong'], '-'),
                                ('error_cong', 'error, cong', COL['cong'], '--'),
                                ('correct_incong', 'correct, incong', COL['incong'], '-'),
                                ('error_incong', 'error, incong', COL['incong'], '--')]:
        band(axes[1], ts, _stack_curve(curves, 'accum', key), lbl, color, linestyle=ls)
    axes[1].axhline(0, color='k', linewidth=0.7, alpha=0.3)
    axes[1].set_xticks(ts)
    axes[1].set_xlabel('Timestep within trial')
    axes[1].set_ylabel('Output toward final choice\n(sign-normalised)')
    # Four curves leave no free corner here, so the key sits above the panel instead.
    compact_legend(axes[1], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2,
                   columnspacing=0.9)

    diff_near = (_stack_curve(curves, 'acc_by_ts', 'near_cong')
                 - _stack_curve(curves, 'acc_by_ts', 'near_incong'))
    diff_far  = (_stack_curve(curves, 'acc_by_ts', 'far_cong')
                 - _stack_curve(curves, 'acc_by_ts', 'far_incong'))
    band(axes[2], ts, diff_near, 'near', COL['near_incong'])
    band(axes[2], ts, diff_far,  'far',  COL['far_incong'])
    axes[2].axhline(0, color='k', linewidth=0.7, alpha=0.3)
    axes[2].set_xticks(ts)
    axes[2].set_xlabel('Timestep within trial')
    axes[2].set_ylabel('Congruency effect (accuracy)')
    compact_legend(axes[2], loc='lower right')

    _stamp(fig, 'Within-trial dynamics', variant, p, len(curves))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_within_trial.pdf')


def fig_history(effects, out_dir, p, variant=DEFAULT_VARIANT):
    """Sequential congruency: all four history cells, and the repetition control."""
    shades = ['#c6dbef', '#6baed6', '#3182bd', '#08519c']
    hist   = ['CC', 'CI', 'IC', 'II']

    fig, axes = bar_row([
        ([(_stack(effects, f'acc_{h}_to_I'), h, c) for h, c in zip(hist, shades)],
         dict(ylabel='Accuracy', connect=True, title='→ incongruent')),
        ([(_stack(effects, f'rt_{h}_to_I'), h, c) for h, c in zip(hist, shades)],
         dict(ylabel='RT (timesteps)', connect=True, title='→ incongruent')),
        ([(_stack(effects, f'rt_{h}_to_C'), h, c) for h, c in zip(hist, shades)],
         dict(ylabel='RT (timesteps)', connect=True, title='→ congruent')),
        ([(_stack(effects, 'sce_acc_switch'), 'acc\nswitch', COL['cong']),
          (_stack(effects, 'sce_acc_repeat'), 'acc\nrepeat', COL['incong']),
          (_stack(effects, 'sce_rt_switch'),  'RT\nswitch',  COL['far_cong']),
          (_stack(effects, 'sce_rt_repeat'),  'RT\nrepeat',  COL['far_incong'])],
         dict(ylabel='Sequential congruency effect', baseline=0.0)),
    ])

    # The two RT panels share a scale: incongruent trials are the slower ones, and that
    # gap is the point — it disappears if each panel picks its own limits.
    share_ylim(axes[1], axes[2])

    _stamp(fig, 'Trial history (t−2,t−1; post-correct only)', variant, p, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_history.pdf')


def fig_post_error(effects, out_dir, p, variant=DEFAULT_VARIANT):
    """Post-error adaptation and what drives the control update."""
    fig, axes = bar_row([
        ([(_stack(effects, 'pes_BI'), 'B incong', COL['incong']),
          (_stack(effects, 'pes_BC'), 'B cong',   COL['cong'])],
         dict(ylabel='Post-error slowing (RT)', baseline=0.0)),
        ([(_stack(effects, 'pia_BI'), 'B incong', COL['incong']),
          (_stack(effects, 'pia_BC'), 'B cong',   COL['cong'])],
         dict(ylabel='Post-error accuracy change', baseline=0.0)),
        ([(_stack(effects, 'focus_in_after_near_err'), 'after\nnear err', COL['near_incong']),
          (_stack(effects, 'focus_in_after_far_err'),  'after\nfar err',  COL['far_incong'])],
         dict(ylabel='Inherited Z focus', connect=True)),
        ([(_stack(effects, 'dfocus_near_err'),  'near\nerr',  COL['near_incong']),
          (_stack(effects, 'dfocus_far_err'),   'far\nerr',   COL['far_incong']),
          (_stack(effects, 'dfocus_near_corr'), 'near\ncorr', COL['near_cong']),
          (_stack(effects, 'dfocus_far_corr'),  'far\ncorr',  COL['far_cong'])],
         dict(ylabel='Δ Z focus (update)', baseline=0.0)),
    ])

    _stamp(fig, 'Post-error effects (incongruent trial A)', variant, p, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_post_error.pdf')


def fig_proportion_congruent(out_dir, variant=DEFAULT_VARIANT):
    """Congruency and distance effects as a function of how common congruent trials are."""
    levels, acc, rt, dist = [], [], [], []
    for p in P_CONGRUENT_LEVELS:
        effects, _ = collect_sessions(p, variant)
        if not effects:
            continue
        levels.append(p)
        acc.append(_stack(effects, 'cong_effect_acc'))
        rt.append(_stack(effects, 'cong_effect_rt'))
        dist.append(_stack(effects, 'dist_effect_acc_incong'))
    if not levels:
        return None

    fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3, panel=FigSize.wide))
    for ax, arr, ylabel, base in [(axes[0], np.array(acc), 'Congruency effect (accuracy)', None),
                                  (axes[1], np.array(rt),  'Congruency effect (RT)', None),
                                  (axes[2], np.array(dist), 'Near − far (acc, incongruent)', 0.0)]:
        series(ax, levels, arr, None, COL['incong'], seed_lines=True)
        if base is not None:
            ax.axhline(base, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
        ax.set_xlabel('P(congruent)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(levels)

    label = '' if variant == 'baseline' else f'{variant}, '
    fig.suptitle(f'List-wide proportion congruent — {label}lines are subjects')
    fig.tight_layout()
    name = 'group_proportion_congruent' + ('' if variant == 'baseline' else f'__{variant}')
    return save(fig, f'{out_dir}/{name}.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(p=DEFAULT_P, variant=DEFAULT_VARIANT, run=None):
    from flanker_sweep import use_run

    # Everything below resolves paths through RUN_NAME, so one context manager
    # steers both what is read and where the figures are written.
    with use_run(run or RUN or None) as active:
        print(f'sweep run: {active}')
        out_dir = out_dir_for(p, variant)
        print(f'Building group figures for p_congruent={p}, variant={variant}')

        effects, curves = collect_sessions(p, variant)
        if not effects:
            print('No results found — run flanker_sweep.py first.')
            return

        fig_fingerprint(effects, out_dir, p, variant)
        fig_learning_and_rt(curves, out_dir, p, variant)
        fig_within_trial(curves, out_dir, p, variant)
        fig_history(effects, out_dir, p, variant)
        fig_post_error(effects, out_dir, p, variant)
        fig_proportion_congruent(sweep_root(), variant)


def args_from_argv(default_p=DEFAULT_P, default_variant=DEFAULT_VARIANT):
    """
    (p_congruent, variant) from the command line.

    Accepts `--p 0.5 --variant spatial_steep` or a bare numeric argument. Inside a
    Jupyter / VS Code interactive window sys.argv belongs to the kernel rather than to
    this script (e.g. '--f=...kernel-abc.json'), so anything unrecognised is ignored.
    """
    p, variant, run, argv = default_p, default_variant, None, sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == '--p' and i + 1 < len(argv):
            p = float(argv[i + 1])
        elif arg == '--variant' and i + 1 < len(argv):
            variant = argv[i + 1]
        elif arg == '--run' and i + 1 < len(argv):
            run = argv[i + 1]
        elif not arg.startswith('-'):
            try:
                p = float(arg)
            except ValueError:
                continue
    return p, variant, run


if __name__ == '__main__':
    main(*args_from_argv())
