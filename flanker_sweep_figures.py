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
    9  z_update       what each kind of trial teaches Z, and what the state buys
   10  post_conflict  the conflict twin of 5: is the model slower and better after conflict
   11  z_slot_update  the same update per slot, so the gate's own profile is visible

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

In a Jupyter / VS Code interactive window there is no command line to read — sys.argv
belongs to the kernel — so the module knobs below stand in for it: DEFAULT_VARIANT picks
one variant (None sweeps through the whole ladder) and BUILD_NOISE_SERIES decides whether
the cross-variant group_8 figure is built. Run as a script, the command line wins and both
knobs are ignored, so batch jobs keep their existing meaning.

Panel primitives and loading live in flanker_figure_utils.py; the per-seed measures in
flanker_metrics.py.
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt

from plot_style import FigSize

from flanker_figure_utils import (CELLS, COL, band, bar_grid, bar_row, bars_with_seeds,
                                  collect_effects, collect_sessions, compact_legend,
                                  dots_with_ci, exchange_panel, landing_marks,
                                  out_dir_for, plot_circularity, save,
                                  series, share_ylim, spec_post_conflict,
                                  spec_rt_by_outcome, spec_z_slot_update, sweep_root,
                                  _has, _interactive_kernel, _stack, _stack_curve)
from flanker_analyses import _timestep_ticks
from flanker_metrics import SIGNATURES
from flanker_sweep_config import DELAY_LADDER, NOISE_LADDER, VARIANTS

#: Sweep run to read and write. None follows flanker_sweep_config.RUN_NAME;
#: see flanker_sweep.SWEEP_RUNS for what each run contains.
RUN = None

#: Which variant to build when there is no command line to read — i.e. in a Jupyter /
#: VS Code interactive window. A single name ('noise09') builds just that variant, which
#: is what you usually want while iterating on a figure; None sweeps through every variant
#: in the ladder. Ignored when this file is run as a script: there `--variant` says it, and
#: passing no argument keeps meaning "every variant", which run_flanker_factorial.sh needs.
DEFAULT_VARIANT = 'noise09'

#: Whether to also build group_8_noise_series, the one cross-variant figure. It reloads
#: EVERY level on disk whatever DEFAULT_VARIANT says, so it is the slow half of a run that
#: was only meant to rebuild one variant. None decides from the variant — the ladder figure
#: is built when the whole ladder is being built anyway, and skipped otherwise. True or
#: False forces it either way.
BUILD_NOISE_SERIES = None

#: RT threshold these figures are computed at. None follows flanker_sweep_config; a number
#: overrides it here without editing the sweep config. This has to be threaded through both
#: `collect_sessions` and `extract_trials` below — setting the module variable alone used to
#: do nothing, because build_variant re-imported the config value over the top of it.
RT_THRESHOLD = 0.2

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

    # The delay is a property of the session, identical across seeds of one variant.
    delay = int(curves[0].get('target_delay', 0) or 0)
    for ax in axes:
        ax.set_xticks(_timestep_ticks(ad))
        ax.set_xlabel('Timestep within trial')
        if delay:
            # With predict_first_frame=True the target first reaches an output at
            # delay + 1. Before that line the only evidence is the flankers, so the
            # incongruent traces being on the wrong side of zero there IS the effect.
            ax.axvline(delay + 1, color='k', linewidth=0.8, linestyle='-.', alpha=0.45,
                       zorder=0)

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

    The second row splits RT by outcome *within each cell*, which the pooled panel above
    cannot do. Human flanker errors are fast — on an incongruent trial the flankers reach
    threshold before the target — so the contrast in the last panel is the signature, and
    it is the decided-only version because errors are exactly the trials that fail to
    cross: on `rt_interp` a non-response sits at the trial end and would read as a slow
    error rather than as no response at all.
    """
    fig, axes = plt.subplots(2, 3, figsize=FigSize.grid(2, 3, panel=FigSize.wide))
    ad = len(curves[0]['rt_x']) - 1
    x  = curves[0]['rt_bins'] if interpolate else curves[0]['rt_x']
    group = 'rt_density' if interpolate else 'rt_pmf'

    def _rt_axis(ax, title):
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

    for ax, keys, title in [(axes[0, 0], [('cong', 'congruent', COL['cong']),
                                          ('incong', 'incongruent', COL['incong'])],
                             'by congruency'),
                            (axes[0, 1], [('correct', 'correct', COL['correct']),
                                          ('error', 'error', COL['error'])], 'by outcome')]:
        for key, label, color in keys:
            band(ax, x, _stack_curve(curves, group, key), label, color)
        _rt_axis(ax, title)
    share_ylim(axes[0, 0], axes[0, 1])

    bars_with_seeds(axes[0, 2], [(1.0 - _stack(effects, f'dec_{k}'), lbl, COL[k])
                                 for k, lbl in CELLS],
                    'Undecided fraction', title='never crossed threshold')

    # Row 2: the same distributions per cell, correct and error side by side, then the
    # contrast the scorecard scores. Dashed = error, the line-plot counterpart of the
    # hollow bars elsewhere (plot_style.outcome_style — outcome rides on fill, not hue).
    for ax, outcome, ls, title in [(axes[1, 0], 'corr', '-', 'correct, by cell'),
                                   (axes[1, 1], 'err', '--', 'errors, by cell')]:
        for key, lbl in CELLS:
            band(ax, x, _stack_curve(curves, group, f'{key}_{outcome}'),
                 lbl.replace('\n', '-'), COL[key], linestyle=ls)
        _rt_axis(ax, title)
    share_ylim(axes[1, 0], axes[1, 1])

    groups, kw = spec_rt_by_outcome(effects, decided=True)[2]
    bars_with_seeds(axes[1, 2], groups, **kw)

    _stamp(fig, 'Reaction time, non-responses, and RT by outcome', variant, len(curves))
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

    # Accuracy panels are clipped at 0.4 with chance marked at 0.5, matching
    # run_flanker.py's single-session version of this figure. From zero the four cells
    # differ by a few percent of the axis and the conflict-adaptation step — the whole
    # point of the panel — is invisible.
    acc_ylim = (0.4, 1.02)
    fig, axes = bar_row([
        ([(_stack(effects, f'acc_{h}_to_I'), h, c) for h, c in zip(hist, shades)],
         dict(ylabel='Accuracy', connect=True, title='→ incongruent',
              baseline=0.5, ylim=acc_ylim)),
        ([(_stack(effects, f'acc_{h}_to_C'), h, c) for h, c in zip(hist, shades)],
         dict(ylabel='Accuracy', connect=True, title='→ congruent',
              baseline=0.5, ylim=acc_ylim)),
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
    if len(v) < 2:
        return np.array([])
    sd = v.std(ddof=1)
    # An effect identical in every seed has no across-seed SD to divide by. That happens
    # once a cell is on ceiling (every seed at 1.000), where the honest answer is "no
    # variance to standardise", not an infinite effect size.
    return v / sd if sd > 0 else np.array([])


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

def fig_ladder_series(out_dir, ladder, panels, xlabel, fname, title,
                      invert=True):
    """Every panel in `panels` plotted against a ladder of variants.

    `ladder` is a list of (variant_name, x_value). Each panel is either a key into the
    per-seed effects dict, or a tuple of (key, label, colour) triples sharing one axis.
    Factored out of fig_noise_series so a second ladder — the target-onset delay — can
    reuse the machinery with its own x axis and its own panel set.
    """
    levels, per_level = [], []
    for name, x in ladder:
        effects = collect_effects(name)
        if effects:
            levels.append(x)
            per_level.append(effects)
    if len(levels) < 2:
        print(f'{fname} needs at least two levels on disk — skipping.')
        return None

    def stack(key):
        return np.array([_stack(e, key) for e in per_level])       # (levels, seeds)

    n = len(panels)
    ncol = 4 if n > 4 else n
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=FigSize.grid(nrow, ncol, panel=FigSize.wide))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, ylabel, ref) in zip(axes, panels):
        if isinstance(key, tuple):        # multiple series sharing one panel
            for k, label, color in key:
                series(ax, levels, stack(k), label, color, seed_lines=False)
            compact_legend(ax, loc='best')
        else:
            series(ax, levels, stack(key), None, COL['incong'], seed_lines=True)
        if ref is not None:
            ax.axhline(ref, color='k', linewidth=0.6, linestyle=':', alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(levels)
        ax.tick_params(axis='x', labelsize=5)
        if invert:
            ax.invert_xaxis()
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    return save(fig, f'{out_dir}/{fname}')


# ── 8b. The target-onset delay ladder ─────────────────────────────────────────

#: RT first, because the delay's primary prediction is about WHEN the model responds.
#: The two RT levels share a panel deliberately: a contrast alone cannot say whether
#: congruent got faster or incongruent got slower, and with a delayed target the
#: interesting case is congruent RT barely moving because the flankers already carry
#: the answer.
DELAY_PANELS = [
    ('cong_effect_rt', 'Congruency effect on RT', 0.0),
    ((('rt_cong', 'congruent', COL['cong']),
      ('rt_incong', 'incongruent', COL['incong'])),
     'RT (timesteps from trial start)', None),
    ('cong_effect_acc', 'Congruency effect on accuracy', 0.0),
    ('undecided_frac', 'Trials that never decided', None),
    ((('acc_cong', 'congruent', COL['cong']),
      ('acc_incong', 'incongruent', COL['incong'])),
     'Accuracy', 0.5),
    ('fasterr_incong_decided', 'Errors faster than correct (inc.)', 0.0),
    ('peri', 'Interference drop (PERI)', 0.0),
    ('pes_BI', 'Post-error slowing (PES)', None),
]


def fig_delay_series(out_dir, ladder=None):
    """Every RT-relevant signature against the target-onset delay.

    The prediction the delay tests: responses are later when the target is later, but
    congruent trials are held up less than incongruent ones, because during the delay the
    flankers alone already point at the answer. So `cong_effect_rt` should grow with delay
    while congruent RT rises least — and incongruent accuracy should fall as early
    flanker-driven responses become errors.

    x is NOT inverted here: delay increases left to right, in the direction of the
    manipulation, unlike the noise ladder which reads cleanest with the clean end on the left.
    """
    return fig_ladder_series(
        out_dir, ladder or DELAY_LADDER, DELAY_PANELS,
        xlabel='target_delay (timesteps)',
        fname='group_12_delay_series.pdf',
        title='The target-onset delay — does a later target mean a later response?',
        invert=False)


#: What changes behaviourally (top row) and why (bottom row). The paired-series panels
#: give the cell levels rather than only a contrast: a difference score that shrinks cannot
#: say whether interference fell or the task simply got easier.
NOISE_PANELS = [
    ('pes_BI',                  'Post-error slowing (PES)',        None),
    ('pia_BI',                  'Post-error accuracy (PIA)',       0.0),
    ('peri',                    'Interference drop (PERI)',        0.0),
    ((('dist_effect_acc_cong', 'congruent', COL['cong']),
      ('dist_effect_acc_incong', 'incongruent', COL['incong'])),
     'Accuracy: near − far', 0.0),
    ('dfocus_err_noisy',        'Δ focus after a noise-driven error', 0.0),
    ('frac_err_noisy',          'Fraction of noise-driven errors', 0.5),
    ((('acc_near_cong', 'near-congruent', COL['near_cong']),
      ('acc_far_cong', 'far-congruent', COL['far_cong'])),
     'Accuracy: congruent cells', 0.5),
    ((('acc_near_incong', 'near-incongruent', COL['near_incong']),
      ('acc_far_incong', 'far-incongruent', COL['far_incong'])),
     'Accuracy: incongruent cells', 0.5),
]


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

    x is inverted: cleaner on the left, noisier on the right.
    """
    return fig_ladder_series(
        out_dir, ladder or NOISE_LADDER, NOISE_PANELS,
        xlabel='arrow_noise_std',
        fname='group_8_noise_series.pdf',
        title='The noise ladder — behaviour above, the mechanism below',
        invert=True)


# ── 9. What each trial teaches Z ──────────────────────────────────────────────

def fig_z_update(effects, trials_list, out_dir, variant):
    """
    The control update each kind of trial produces, and what the resulting state buys.

    `delta_focus` is the update a trial *generated*, not the state it sat in, and that
    distinction is the only reason this figure may be grouped by the trial's own
    condition at all: reading a focus level after conditioning on the outcome is circular
    — a focused Z is what made the trial correct — while the update describes the learning
    rule, so grouping it by the trial's own properties is legitimate. This is the group
    version of run_flanker.py's Result 5, which shows the same contrast in one session.

    Panels 1 and 2 hold the four congruency x distance cells and split them by outcome
    rather than pooling, because the two halves answer different questions: a correct
    trial's update is what maintains the state, an error's is the correction the model
    actually makes. The error panel is drawn hollow — the house convention for an error
    cell, so outcome never has to spend a hue.

    The two panels are deliberately NOT on a shared y scale. An error's update is three to
    five times a correct trial's, so sharing flattens the correct panel onto its baseline
    and hides what it is there to show: that congruent trials teach Z *away* from the
    target while incongruent ones do not. Read the magnitudes off the two axes, which
    differ by design; group_6's third panel is where the pooled sizes are compared on one
    scale.

    Panels 3 and 4 are the exchange rate: how much accuracy, and how much RT, a given
    inherited control state buys on incongruent trials, with the states the trial after an
    error and after a correct trial actually inherited marked on both. Panel 3 is the same
    quantity as group_6's fifth panel, repeated here so the pair reads together — a
    control account says the gap has to be paid for in speed as well as in accuracy, and a
    model that charges for it in only one of the two is not reproducing the human
    trade-off.
    """
    from flanker_metrics import event_locked

    fig, axes = plt.subplots(1, 4, figsize=FigSize.row(4, panel=FigSize.wide))

    for ax, outcome, title in [(axes[0], 'corr', 'correct trials'),
                               (axes[1], 'err', 'errors')]:
        bars_with_seeds(ax, [(_stack(effects, f'dfocus_{k}_{outcome}'), lbl, COL[k])
                             for k, lbl in CELLS],
                        'Δ Z focus (this trial\'s update)', baseline=0.0, title=title,
                        hollow=[outcome == 'err'] * len(CELLS))

    # Sessions are the replicate, as in fig_circularity: event_locked returns per-event
    # traces, averaged within a session before stacking.
    per_session = [event_locked(tr) for tr in trials_list]
    ev = {'lags': per_session[0]['lags']}
    for key in ('focus_err', 'focus_corr'):
        ev[key] = np.array([np.nanmean(s[key], axis=0) for s in per_session])
    marks = landing_marks(ev)

    for ax, key, ylabel in [(axes[2], 'curve_y',  'Accuracy, incongruent trials'),
                            (axes[3], 'curve_rt', 'RT (timesteps), incongruent trials')]:
        exchange_panel(ax, [s['curve_x'] for s in per_session],
                       [s[key] for s in per_session], ylabel, marks=marks)

    _stamp(fig, 'What each trial teaches Z, and what the state buys', variant, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_9_z_update.pdf')


# ── 10. The conflict twin of the post-error figure ────────────────────────────

def fig_post_conflict(effects, out_dir, variant):
    """
    Is the model slower and more accurate after conflict, the way a human is?

    The deliberate parallel of `fig_post_error`: same shape, same trial-B split, one
    factor changed. There trial A is an error; here trial A is *correct* and incongruent.
    That restriction is what makes this conflict adaptation rather than a second view of
    post-error adaptation — incongruent trials fail more often, so an unrestricted
    "after an incongruent trial" contrast is partly post-error slowing under another name.

    Panels are built from `spec_post_conflict`, which run_flanker.py's Result 3c also
    draws, so the workbench and the group version cannot drift apart.
    """
    fig, axes = bar_row(spec_post_conflict(effects))
    share_ylim(axes[0], axes[1])        # PCS against its decided-only companion
    _stamp(fig, 'Post-incongruent adaptation (post-correct trial A)', variant, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_10_post_conflict.pdf')


# ── 11. The update, slot by slot ──────────────────────────────────────────────

def fig_z_slot_update(effects, out_dir, variant):
    """
    What the update did to each slot, rather than to the scalar focus index.

    `delta_focus` — the measure every other Z panel uses — is centre minus the mean of the
    flankers, so it can only say whether the gate moved toward the target. These rows say
    where it went. Row 1 is the fixed geometry (centre, the near pair, the far pair); row
    2 is the role each pair played on that trial, which is not the same thing, because a
    near display leaves slots 0 and 4 empty and a far display leaves 1 and 3.

    Row 3 appears only when the run logged gradients. It matters because `delta_z` is the
    change in the *softmaxed* gate and therefore sums to ~0 across slots — the centre
    cannot rise without something else falling, so a negative bar in rows 1 and 2 is not
    on its own evidence of suppression. The raw dL/dZ carries no such constraint and is
    the honest read of what the trial's error actually asked for.

    Correct and error panels are not on a shared y scale, for the reason `fig_z_update`
    gives: an error's update is several times a correct trial's, and sharing would flatten
    the correct panel onto its baseline.
    """
    rows = [spec_z_slot_update(effects, grouping='geometry'),
            spec_z_slot_update(effects, grouping='role')]
    if _has(effects, 'zgrad_centre_incong_err'):
        rows.append(spec_z_slot_update(effects, grouping='geometry', measure='zgrad'))
    fig, axes = bar_grid(rows)
    _stamp(fig, 'What the update does to each slot', variant, len(effects))
    fig.tight_layout()
    return save(fig, f'{out_dir}/group_11_z_slot_update.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def _rt_threshold():
    """The RT threshold to read sessions at — the module knob, else the sweep's own."""
    from flanker_sweep_config import RT_THRESHOLD as SWEEP_RT_THRESHOLD
    return SWEEP_RT_THRESHOLD if RT_THRESHOLD is None else RT_THRESHOLD


def build_variant(variant, out_dir=None):
    """The ten per-variant figures."""
    from flanker_analyses import extract_trials
    from flanker_sweep import load_condition

    rt_threshold = _rt_threshold()
    effects, curves = collect_sessions(variant, rt_threshold=rt_threshold)
    if not effects:
        print(f'  [{variant}] no results on disk — skipping.')
        return
    out_dir = out_dir or out_dir_for(variant)
    fig_fingerprint(effects, out_dir, variant)
    fig_within_trial(curves, out_dir, variant)
    fig_rt(curves, effects, out_dir, variant)
    fig_history(effects, out_dir, variant)
    fig_post_error(effects, out_dir, variant)
    trials_list = [extract_trials(r['train_logger'], r['config'], rt_threshold=rt_threshold)
                   for r in load_condition(variant)]
    fig_circularity(trials_list, out_dir, variant)
    fig_scorecard(effects, out_dir, variant)
    fig_z_update(effects, trials_list, out_dir, variant)
    fig_post_conflict(effects, out_dir, variant)
    fig_z_slot_update(effects, out_dir, variant)


def main(variant=None, run=None, noise_series=None):
    from flanker_sweep import use_run

    # DEFAULT_VARIANT applies only where the command line cannot: inside a kernel, sys.argv
    # belongs to the kernel, so `--variant` can never be typed and the module knob is the
    # only way to say "just this one". Run as a script the command line stays authoritative
    # and no argument still means every variant.
    if variant is None and _interactive_kernel():
        variant = DEFAULT_VARIANT
    if noise_series is None:
        noise_series = BUILD_NOISE_SERIES
    if noise_series is None:
        noise_series = variant is None      # the ladder figure needs the whole ladder

    # Everything below resolves paths through RUN_NAME, so one context manager steers both
    # what is read and where the figures are written.
    with use_run(run or RUN or None) as active:
        names = [variant] if variant else list(VARIANTS)
        print(f'sweep run: {active}  |  variants: {", ".join(names)}  |  '
              f'rt_threshold: {_rt_threshold()}')
        for name in names:
            print(f'\n── {name} ──')
            build_variant(name)
        if noise_series:
            fig_noise_series(sweep_root())
            # Only meaningful once more than one delay level is on disk; the builder says
            # so and returns None otherwise, so calling it unconditionally is safe.
            fig_delay_series(sweep_root())
        else:
            print(f'\ngroup_8_noise_series and group_12_delay_series skipped — they reload '
                  f'every level on disk, not just {names[0]}. Set BUILD_NOISE_SERIES = True '
                  f'to build them anyway.')


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
