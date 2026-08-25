"""
flanker_sweep_figures.py — the group figures, in the order the story is told.

One session is one synthetic subject, so every panel is a mean across seeds with SEM and
one dot per seed; within-subject contrasts join the same seed with a thin grey line.
`run_flanker.py` is the single-session workbench for looking at mechanism; this file is
the evidence.

The story, and the figure that carries each step:

    1  fingerprint    the model shows the flanker effect, and it is larger for near flankers
    2  within_trial   the cost appears inside the trial, as flankers pull the decision away
    3  rt             RT and the trials that never decide, which are a condition effect too
    4  history        conflict adaptation: the four history cells, into I and into C
    5  post_error     the failure — errors do not recruit control
    6  circularity    the control deficit precedes the error, so post-error state is circular
    7  scorecard      every human signature on one axis, matched or not
    8  noise_series   and less stimulus noise makes errors informative again

The noise_series panels are per-seed effects from flanker_metrics.session_effects, stacked
across noise levels and re-plotted with x = arrow_noise_std:
    pes_BI, pia_BI    post_error_effects — RT / accuracy on incongruent trial B, after an
                      error vs. after a correct trial A (A itself restricted to incongruent).
    peri              post_error_effects — the congruency effect on RT after a correct A
                      minus after an error A; positive means an error shrinks interference.
    dfocus_err_noisy  error_diagnosis_effects — mean delta_focus on incongruent error trials
                      whose centre_evidence falls below the incongruent-trial median (a
                      "noise-driven" error, as opposed to a "flanker-driven" one above it).
    frac_err_noisy    error_diagnosis_effects — the share of incongruent errors that fall in
                      that noisy half, by the same median split.
    dist_effect_acc_* behaviour_effects — near-minus-far accuracy, plotted separately for
                      congruent and incongruent trials (the flanker distance effect).
    acc_<cell>        behaviour_effects — the four cell accuracies. Two panels plot these as
                      levels rather than contrasts, the congruent pair and the incongruent
                      pair, so a change in the near-minus-far contrast can be attributed to
                      one cell or the other.

Run:
    python flanker_sweep_figures.py                       # every variant in the sweep
    python flanker_sweep_figures.py --variant noise10
    python flanker_sweep_figures.py --run sweep_noise --variant noise04

Panel primitives and loading live in flanker_figure_utils.py; the per-seed measures in
flanker_metrics.py.
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt

from plot_style import FigSize

from flanker_figure_utils import (CELLS, COL, band, bar_row, bars_with_seeds,
                                  collect_effects, collect_sessions, compact_legend,
                                  dots_with_ci, out_dir_for, plot_circularity, save,
                                  series, share_ylim, sweep_root, _stack, _stack_curve)
from flanker_metrics import SIGNATURES
from flanker_sweep_config import NOISE_LADDER, VARIANTS

#: Sweep run to read and write. None follows flanker_sweep_config.RUN_NAME;
#: see flanker_sweep.SWEEP_RUNS for what each run contains.
RUN = None

DEFAULT_VARIANT = 'noise07' # noise10


def _stamp(fig, text, variant, n):
    fig.suptitle(f'{text} — {variant}, {n} seeds')


# ── 1. The flanker fingerprint ────────────────────────────────────────────────

def fig_fingerprint(effects, out_dir, variant):
    """Accuracy and RT in the four cells, and the distance effect within each congruency."""
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
    _stamp(fig, 'Behavioural fingerprint', variant, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_1_fingerprint.pdf')


# ── 2. Within-trial dynamics ──────────────────────────────────────────────────

def fig_within_trial(curves, out_dir, variant):
    """Where the congruency cost comes from: the decision being pulled away and back."""
    fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3, panel=FigSize.wide))
    ad = len(curves[0]['acc_by_ts']['near_cong'])
    ts = np.arange(ad)

    styles = {'near_cong': '-', 'far_cong': '--', 'near_incong': '-', 'far_incong': '--'}
    for key, lbl in CELLS:
        band(axes[0], ts, _stack_curve(curves, 'acc_by_ts', key),
             lbl.replace('\n', '-'), COL[key], linestyle=styles[key])
    axes[0].axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    axes[0].set_ylabel('P(target)')
    compact_legend(axes[0], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2,
                   columnspacing=0.9)

    for key, lbl, color, ls in [('correct_cong', 'correct, cong', COL['cong'], '-'),
                                ('error_cong', 'error, cong', COL['cong'], '--'),
                                ('correct_incong', 'correct, incong', COL['incong'], '-'),
                                ('error_incong', 'error, incong', COL['incong'], '--')]:
        band(axes[1], ts, _stack_curve(curves, 'accum', key), lbl, color, linestyle=ls)
    axes[1].axhline(0, color='k', linewidth=0.7, alpha=0.3)
    axes[1].set_ylabel('Output toward final choice\n(sign-normalised)')
    compact_legend(axes[1], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2,
                   columnspacing=0.9)

    diff_near = (_stack_curve(curves, 'acc_by_ts', 'near_cong')
                 - _stack_curve(curves, 'acc_by_ts', 'near_incong'))
    diff_far  = (_stack_curve(curves, 'acc_by_ts', 'far_cong')
                 - _stack_curve(curves, 'acc_by_ts', 'far_incong'))
    band(axes[2], ts, diff_near, 'near', COL['near_incong'])
    band(axes[2], ts, diff_far,  'far',  COL['far_incong'])
    axes[2].axhline(0, color='k', linewidth=0.7, alpha=0.3)
    axes[2].set_ylabel('Congruency effect on P(target)')
    compact_legend(axes[2], loc='lower right')

    for ax in axes:
        ax.set_xticks(ts)
        ax.set_xlabel('Timestep within trial')

    _stamp(fig, 'Within-trial dynamics', variant, len(curves))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_2_within_trial.pdf')


# ── 3. RT, and the trials that never decide ───────────────────────────────────

def fig_rt(curves, effects, out_dir, variant, interpolate=False):
    """
    RT as a PMF over timesteps, with the trials that never decided as their own `und.`
    category — the same convention as `flanker_analyses.plot_rt` for a single session.

    With four eligible response timesteps this is the resolution the data has; a
    non-response is a different outcome from a slow response, not a large value of one.
    `interpolate=True` swaps the first two panels for the sub-timestep density over decided
    trials, scaled so its area is the decided proportion, with the same `und.` marker.

    The third panel is the undecided rate per cell, which is a condition effect in its own
    right: incongruent trials fail to decide about three times as often as congruent ones.
    """
    fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3, panel=FigSize.wide))
    ad = len(curves[0]['rt_x']) - 1
    x  = curves[0]['rt_bins'] if interpolate else curves[0]['rt_x']
    group = 'rt_density' if interpolate else 'rt_pmf'

    for ax, keys, title in [(axes[0], [('cong', 'congruent', COL['cong']),
                                       ('incong', 'incongruent', COL['incong'])],
                             'by congruency'),
                            (axes[1], [('correct', 'correct', COL['correct']),
                                       ('error', 'error', COL['error'])], 'by outcome')]:
        for key, label, color in keys:
            band(ax, x, _stack_curve(curves, group, key), label, color)
        if interpolate:
            ax.set_xlabel('RT (timesteps)')
            ax.set_ylabel('Density')
        else:
            ax.set_xticks(list(range(ad + 1)))
            ax.set_xticklabels([str(t) for t in range(ad)] + ['und.'], fontsize=6)
            ax.set_xlabel('Timestep within trial')
            ax.set_ylabel('P(RT = t)')
        ax.set_title(title)
        compact_legend(ax, loc='upper center')
    share_ylim(axes[0], axes[1])

    bars_with_seeds(axes[2], [(1.0 - _stack(effects, f'dec_{k}'), lbl, COL[k])
                              for k, lbl in CELLS],
                    'Undecided fraction', title='never crossed threshold')

    _stamp(fig, 'Reaction time and non-responses', variant, len(curves))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_3_rt.pdf')


# ── 4. Conflict adaptation ────────────────────────────────────────────────────

def fig_history(effects, out_dir, variant):
    """
    The four history cells, into incongruent and into congruent.

    Labels read (t−2, t−1): CI means congruent then incongruent. Restricted to trials whose
    two predecessors were both correct, so this is conflict adaptation rather than
    post-error adaptation. The two targets are shown together because control that helps an
    incongruent trial should cost a congruent one — an effect that appears in only one of
    them is not a control adjustment.
    """
    shades = ['#c6dbef', '#6baed6', '#3182bd', '#08519c']
    hist   = ['CC', 'IC', 'CI', 'II']

    fig, axes = bar_row([
        ([(_stack(effects, f'acc_{h}_to_I'), h, c) for h, c in zip(hist, shades)],
         dict(ylabel='Accuracy', connect=True, title='→ incongruent')),
        ([(_stack(effects, f'acc_{h}_to_C'), h, c) for h, c in zip(hist, shades)],
         dict(ylabel='Accuracy', connect=True, title='→ congruent')),
        ([(_stack(effects, f'rt_{h}_to_I'), h, c) for h, c in zip(hist, shades)],
         dict(ylabel='RT (timesteps)', connect=True, title='→ incongruent')),
        ([(_stack(effects, f'rt_{h}_to_C'), h, c) for h, c in zip(hist, shades)],
         dict(ylabel='RT (timesteps)', connect=True, title='→ congruent')),
        ([(_stack(effects, 'sce_acc_switch'), 'acc\nswitch', COL['cong']),
          (_stack(effects, 'sce_acc_repeat'), 'acc\nrepeat', COL['incong']),
          (_stack(effects, 'sce_rt_switch'),  'RT\nswitch',  COL['far_cong']),
          (_stack(effects, 'sce_rt_repeat'),  'RT\nrepeat',  COL['far_incong'])],
         dict(ylabel='Sequential congruency effect', baseline=0.0,
              title='vs. repetition priming')),
    ])
    share_ylim(axes[0], axes[1])      # accuracy: the → I / → C gap is the point
    share_ylim(axes[2], axes[3])      # and again for RT

    _stamp(fig, 'Conflict adaptation — history (t−2, t−1), post-correct', variant, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_4_history.pdf')


# ── 5. The post-error failure ─────────────────────────────────────────────────

def fig_post_error(effects, out_dir, variant):
    """Post-error adaptation, and the latent update that explains why it is missing."""
    fig, axes = bar_row([
        ([(_stack(effects, 'pes_BI'), 'B incong', COL['incong']),
          (_stack(effects, 'pes_BC'), 'B cong',   COL['cong'])],
         dict(ylabel='Post-error slowing (RT)', baseline=0.0, title='PES')),
        ([(_stack(effects, 'pia_BI'), 'B incong', COL['incong']),
          (_stack(effects, 'pia_BC'), 'B cong',   COL['cong'])],
         dict(ylabel='Post-error accuracy change', baseline=0.0, title='PIA')),
        ([(_stack(effects, 'peri'), 'PERI', COL['neutral'])],
         dict(ylabel='Interference drop after an error', baseline=0.0, title='PERI')),
        ([(_stack(effects, 'dfocus_err_noisy'),  'error\ncentre bad',   COL['error']),
          (_stack(effects, 'dfocus_err_clean'),  'error\ncentre ok',    COL['far_incong']),
          (_stack(effects, 'dfocus_corr_noisy'), 'correct\ncentre bad', COL['far_cong']),
          (_stack(effects, 'dfocus_corr_clean'), 'correct\ncentre ok',  COL['cong'])],
         dict(ylabel='Δ Z focus (this trial\'s update)', baseline=0.0, rotation=30,
              title='what the trial teaches Z')),
    ])
    _stamp(fig, 'Post-error effects (incongruent trial A)', variant, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_5_post_error.pdf')


# ── 6. Why the error does not help the next trial ─────────────────────────────

def fig_circularity(trials_list, out_dir, variant):
    """
    The control deficit precedes the error, so reading the state after an error is circular.

    Sessions are the replicate here: `event_locked` returns per-event traces, which are
    averaged within a session before stacking, so the error bars are across seeds like
    every other group panel.
    """
    from flanker_metrics import event_locked

    per_session = [event_locked(tr) for tr in trials_list]
    ev = {'lags': per_session[0]['lags']}
    for key in ('focus_err', 'focus_corr', 'acc_err', 'acc_corr'):
        ev[key] = np.array([np.nanmean(s[key], axis=0) for s in per_session])
    for key in ('start_gap', 'upd_err', 'upd_corr',
                'frac_err_noisy', 'frac_err_clean',
                'dfocus_err_noisy', 'dfocus_err_clean'):
        ev[key] = np.array([s[key] for s in per_session])
    for key in ('curve_x', 'curve_y'):
        ev[key] = [s[key] for s in per_session]

    fig, axes = plt.subplots(1, 5, figsize=FigSize.row(5, panel=FigSize.large))
    plot_circularity(axes, ev)
    _stamp(fig, 'Post-error control — the deficit precedes the error', variant,
           len(trials_list))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_6_circularity.pdf')


# ── 7. The scorecard ──────────────────────────────────────────────────────────

def _effect_size(values, sign):
    """Cohen's d across seeds, flipped so positive always means human-consistent."""
    v = np.asarray(values, dtype=float) * sign
    v = v[~np.isnan(v)]
    return v / v.std(ddof=1) if len(v) > 1 else np.array([])


def fig_scorecard(effects, out_dir, variant):
    """
    Every human signature on one axis: does this model reproduce it?

    Each effect is divided by its across-seed SD so accuracy, RT and latent measures are
    comparable, and multiplied by the sign a human dataset shows — positive is always
    "matches humans". The count on the right is seeds with the predicted sign, which is the
    honest summary when a group mean rests on a couple of outliers.
    """
    rows = list(SIGNATURES)
    fig, ax = plt.subplots(figsize=FigSize.custom(3.4, 0.22 * len(rows) + 0.7))

    for i, (key, label, sign, _source) in enumerate(rows):
        y = len(rows) - i
        vals = _stack(effects, key) * sign
        vals = vals[~np.isnan(vals)]
        d = _effect_size(_stack(effects, key), sign)
        if len(d) < 2:
            continue
        mean, sem = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        color = (COL['pass_'] if mean - 1.96 * sem > 0 else
                 COL['fail'] if mean + 1.96 * sem < 0 else COL['null'])
        dots_with_ci(ax, y, d, color)
        ax.text(1.02, y, f'{int((vals > 0).sum())}/{len(vals)}',
                transform=ax.get_yaxis_transform(), fontsize=5, va='center', color=color)

    ax.axvline(0, color='k', linewidth=0.7, alpha=0.6)
    ax.set_yticks([len(rows) - i for i in range(len(rows))])
    ax.set_yticklabels([lbl for _, lbl, _, _ in rows], fontsize=5)
    ax.set_ylim(0.3, len(rows) + 0.9)
    ax.set_xlabel('Effect size across seeds (signed so + = matches humans)')
    ax.set_title(f'{variant}, {len(effects)} seeds')
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_7_scorecard.pdf')


# ── 8. The noise ladder ───────────────────────────────────────────────────────

def fig_noise_series(out_dir, ladder=None):
    """
    Every signature against stimulus noise, with the mechanism underneath.

    The top row is what changes behaviourally; the bottom row is why. `arrow_noise_std` sets
    how often the target slot's own samples mislead, which decides whether an error means
    "control was too low" or "that sample was unlucky". The latent update cannot tell those
    apart — it minimises this trial's prediction error — so when noise is high the common
    error teaches the model to attend the target *less*, and the post-error signatures
    invert. The bottom-left panel is the load-bearing one: where Δ focus after a noise-driven
    error crosses zero is where the post-error effects should come back.
    """
    ladder = ladder or NOISE_LADDER
    levels, per_level = [], []
    for name, noise in ladder:
        effects = collect_effects(name)
        if effects:
            levels.append(noise)
            per_level.append(effects)
    if len(levels) < 2:
        print('Noise series needs at least two levels on disk — skipping.')
        return None

    def stack(key):
        return np.array([_stack(e, key) for e in per_level])       # (levels, seeds)

    panels = [
        ('pes_BI',                  'Post-error slowing (PES)',        None),
        ('pia_BI',                  'Post-error accuracy (PIA)',       0.0),
        ('peri',                    'Interference drop (PERI)',        0.0),
        ((('dist_effect_acc_cong', 'congruent', COL['cong']),
          ('dist_effect_acc_incong', 'incongruent', COL['incong'])),
         'Accuracy: near − far', 0.0),
        ('dfocus_err_noisy',        'Δ focus after a noise-driven error', 0.0),
        ('frac_err_noisy',          'Fraction of noise-driven errors', 0.5),
        # The cell accuracies themselves rather than a contrast, congruent pair and
        # incongruent pair. A difference score that shrinks cannot say whether interference
        # fell or the task got easier; these two panels can, because the levels are visible.
        ((('acc_near_cong', 'near-congruent', COL['near_cong']),
          ('acc_far_cong', 'far-congruent', COL['far_cong'])),
         'Accuracy: congruent cells', 0.5),
        # Both incongruent cells: same idea, so a change in the near-far contrast can be
        # attributed to the near cell or the far one.
        ((('acc_near_incong', 'near-incongruent', COL['near_incong']),
          ('acc_far_incong', 'far-incongruent', COL['far_incong'])),
         'Accuracy: incongruent cells', 0.5),
    ]
    fig, axes = plt.subplots(2, 4, figsize=FigSize.grid(2, 4, panel=FigSize.wide))
    for ax, (key, ylabel, ref) in zip(axes.ravel(), panels):
        if isinstance(key, tuple):        # multiple series sharing one panel
            # Per-seed lines are left off here: two series x 20 seeds is 40 threads of
            # spaghetti, and the comparison between the two means is the point.
            for k, label, color in key:
                series(ax, levels, stack(k), label, color, seed_lines=False)
            compact_legend(ax, loc='best')
        else:
            series(ax, levels, stack(key), None, COL['incong'], seed_lines=True)
        if ref is not None:
            ax.axhline(ref, color='k', linewidth=0.6, linestyle=':', alpha=0.6)
        ax.set_xlabel('arrow_noise_std')
        ax.set_ylabel(ylabel)
        ax.set_xticks(levels)
        ax.tick_params(axis='x', labelsize=5)     # the ladder is dense once it has 7 rungs
        ax.invert_xaxis()                 # cleaner on the left, noisier on the right

    fig.suptitle('The noise ladder — behaviour above, the mechanism below')
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_8_noise_series.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def build_variant(variant, out_dir=None):
    """The seven per-variant figures."""
    from flanker_analyses import extract_trials
    from flanker_sweep import load_condition
    from flanker_sweep_config import RT_THRESHOLD

    effects, curves = collect_sessions(variant)
    if not effects:
        print(f'  [{variant}] no results on disk — skipping.')
        return
    out_dir = out_dir or out_dir_for(variant)
    fig_fingerprint(effects, out_dir, variant)
    fig_within_trial(curves, out_dir, variant)
    fig_rt(curves, effects, out_dir, variant)
    fig_history(effects, out_dir, variant)
    fig_post_error(effects, out_dir, variant)
    trials_list = [extract_trials(r['train_logger'], r['config'], rt_threshold=RT_THRESHOLD)
                   for r in load_condition(variant)]
    fig_circularity(trials_list, out_dir, variant)
    fig_scorecard(effects, out_dir, variant)


def main(variant=None, run=None):
    from flanker_sweep import use_run

    # Everything below resolves paths through RUN_NAME, so one context manager steers both
    # what is read and where the figures are written.
    with use_run(run or RUN or None) as active:
        print(f'sweep run: {active}')
        for name in ([variant] if variant else list(VARIANTS)):
            print(f'\n── {name} ──')
            build_variant(name)
        fig_noise_series(sweep_root())


def args_from_argv():
    """
    `--variant <name> --run <sweep>` from the command line.

    Inside a Jupyter / VS Code interactive window sys.argv belongs to the kernel rather
    than to this script, so anything unrecognised is ignored.
    """
    variant, run, argv = None, None, sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == '--variant' and i + 1 < len(argv):
            variant = argv[i + 1]
        elif arg == '--run' and i + 1 < len(argv):
            run = argv[i + 1]
    return variant, run


if __name__ == '__main__':
    main(*args_from_argv())
