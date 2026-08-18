"""
flanker_sweep_figures.py — group-level figures across seeds.

Every panel shows the mean across seeds with SEM error bars, plus one dot per seed so
the spread is visible rather than hidden inside an error bar. Where a comparison is
within-subject, thin grey lines connect the same seed across conditions — that is the
contrast the statistics actually test.

Run:
    python flanker_sweep_figures.py           # reference condition (p_congruent = 0.5)
    python flanker_sweep_figures.py 0.2       # a different congruency proportion

Requires flanker_sweep.py to have been run first.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import plot_style
plot_style.set_plot_style()
from plot_style import FigSize

from flanker_analyses import extract_trials, lagged_factors
from flanker_sweep import load_condition, load_pretrain_curve, result_path
from flanker_sweep_config import P_CONGRUENT_LEVELS, RT_THRESHOLD, SEEDS
from flanker_sweep_analysis import session_effects, summarize

DEFAULT_P = 0.8

COL = dict(cong='#4393c3', incong='#d6604d',
           near_cong='#4393c3', far_cong='#a8c8e8',
           near_incong='#8b0000', far_incong='#f4a582',
           correct='#4393c3', error='#8b0000', neutral='#777777')

CELLS = [('near_cong', 'near\ncong'), ('far_cong', 'far\ncong'),
         ('near_incong', 'near\nincong'), ('far_incong', 'far\nincong')]


# ── Generic panel helpers ─────────────────────────────────────────────────────

def bars_with_seeds(ax, groups, ylabel, baseline=None, connect=False, rotation=0):
    """
    Bar chart of across-seed means with SEM, overlaid with one dot per seed.

    groups  : list of (values_per_seed, label, color)
    connect : join the same seed across bars — use for within-subject contrasts
    """
    x = np.arange(len(groups))
    means = [np.nanmean(v) for v, _, _ in groups]
    sems  = [np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v))) for v, _, _ in groups]
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
    ax.set_xticklabels([lbl for _, lbl, _ in groups], fontsize=5,
                       rotation=rotation, ha='center' if rotation == 0 else 'right')
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def band(ax, x, arr, label, color, linestyle='-'):
    """Mean ± SEM band across seeds. arr is (n_seeds, n_points)."""
    arr = np.asarray(arr, dtype=float)
    n   = np.sum(~np.isnan(arr), axis=0)
    mu  = np.nanmean(arr, axis=0)
    se  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    ax.plot(x, mu, color=color, linestyle=linestyle, linewidth=1.0, label=label)
    ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.18, linewidth=0)


# ── Per-session curves ────────────────────────────────────────────────────────

def session_curves(trials, n_bins=20, rt_bin_width=0.25):
    """Per-session curves that later get averaged across seeds."""
    f = lagged_factors(trials, n_back=2)
    cong, near = f['cong'] == 1, f['near'] == 1
    acc  = trials['correct_at_decision'].astype(float)
    rt   = trials['rt_interp']
    ad   = trials['ad']
    corr = trials['correct_at_decision']

    masks = {'near_cong': near & cong,   'far_cong': ~near & cong,
             'near_incong': near & ~cong, 'far_incong': ~near & ~cong}

    out = {'acc_by_ts': {}, 'accum': {}, 'rt_density': {}, 'learning': {}}

    for key, m in masks.items():
        out['acc_by_ts'][key] = trials['correct'][m].mean(axis=0)

    for key, m in [('correct_cong', corr & cong), ('error_cong', ~corr & cong),
                   ('correct_incong', corr & ~cong), ('error_incong', ~corr & ~cong)]:
        out['accum'][key] = (trials['signed_output'][m].mean(axis=0)
                             if m.any() else np.full(ad, np.nan))

    rt_bins = np.arange(0, ad + rt_bin_width, rt_bin_width)
    out['rt_bins'] = 0.5 * (rt_bins[:-1] + rt_bins[1:])
    for key, m in list(masks.items()) + [('cong', cong), ('incong', ~cong),
                                         ('correct', corr), ('error', ~corr)]:
        vals = rt[m]
        vals = vals[~np.isnan(vals)]
        dens, _ = np.histogram(vals, bins=rt_bins, density=True)
        out['rt_density'][key] = dens if len(vals) else np.full(len(rt_bins) - 1, np.nan)

    # Accuracy over the session, binned by trial position
    edges = np.linspace(0, trials['n_trials'], n_bins + 1).astype(int)
    out['learning_x'] = 0.5 * (edges[:-1] + edges[1:])
    for key, m in [('all', np.ones_like(cong)), ('cong', cong), ('incong', ~cong)]:
        curve = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = np.zeros(trials['n_trials'], dtype=bool)
            sel[a:b] = True
            sel &= m
            curve.append(acc[sel].mean() if sel.any() else np.nan)
        out['learning'][key] = np.array(curve)
    return out


def pretrain_curves(n_bins=25):
    """Stage-1 learning curves for every seed. Returns (x, {key: (n_seeds, n_bins)})."""
    per_seed = {'all': [], 'cong': [], 'incong': []}
    x = None
    for seed in range(SEEDS):
        rec = load_pretrain_curve(seed)
        if rec is None:
            continue
        correct = rec['correct'].astype(float)
        congruent = rec['congruent'] == 1.0
        n = len(correct)
        edges = np.linspace(0, n, n_bins + 1).astype(int)
        x = 0.5 * (edges[:-1] + edges[1:])
        for key, m in [('all', np.ones(n, bool)), ('cong', congruent), ('incong', ~congruent)]:
            curve = []
            for a, b in zip(edges[:-1], edges[1:]):
                sel = np.zeros(n, dtype=bool)
                sel[a:b] = True
                sel &= m
                curve.append(correct[sel].mean() if sel.any() else np.nan)
            per_seed[key].append(curve)
    return x, {k: np.array(v) for k, v in per_seed.items() if v}


def collect_sessions(p_congruent, rt_threshold=RT_THRESHOLD):
    """Load one congruency level; return (per-seed effects, per-seed curves)."""
    effects, curves = [], []
    for res in load_condition(p_congruent):
        trials = extract_trials(res['train_logger'], res['config'], rt_threshold=rt_threshold)
        effects.append(session_effects(trials))
        curves.append(session_curves(trials))
    return effects, curves


def _stack(effects, key):
    return np.array([e[key] for e in effects], dtype=float)


def _stack_curve(curves, group, key):
    return np.array([c[group][key] for c in curves], dtype=float)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_fingerprint(effects, out_dir, p):
    """Accuracy and RT in the four cells, plus the distance effects decomposed."""
    fig, axes = plt.subplots(1, 4, figsize=(FigSize.large[0] * 4.2, FigSize.large[1] * 1.15))

    bars_with_seeds(axes[0], [(_stack(effects, f'acc_{k}'), lbl, COL[k]) for k, lbl in CELLS],
                    'Accuracy', baseline=0.5)
    axes[0].set_title('Accuracy by condition', fontsize=6)

    bars_with_seeds(axes[1], [(_stack(effects, f'rt_{k}'), lbl, COL[k]) for k, lbl in CELLS],
                    'RT (timesteps)')
    axes[1].set_title('RT by condition', fontsize=6)

    # Congruency effect computed within each distance — the within-subject contrast
    bars_with_seeds(axes[2],
                    [(_stack(effects, 'cong_effect_acc_near'), 'near', COL['near_incong']),
                     (_stack(effects, 'cong_effect_acc_far'),  'far',  COL['far_incong'])],
                    'Congruency effect (accuracy)', baseline=0.0, connect=True)
    axes[2].set_title('Congruency effect by distance', fontsize=6)

    # The directional predictions, decomposed
    bars_with_seeds(axes[3],
                    [(_stack(effects, 'dist_effect_acc_cong'),   'congruent\n(expect +)',   COL['cong']),
                     (_stack(effects, 'dist_effect_acc_incong'), 'incongruent\n(expect −)', COL['incong'])],
                    'Near − far (accuracy)', baseline=0.0)
    axes[3].set_title('Distance effect within congruency', fontsize=6)

    fig.suptitle(f'Behavioural fingerprint — {len(effects)} seeds, p(congruent)={p}', fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, 'group_fingerprint.pdf')
    fig.savefig(path, bbox_inches='tight')
    return path


def fig_learning_and_rt(curves, out_dir, p):
    """Learning over training, accuracy over the frozen session, and RT distributions."""
    fig, axes = plt.subplots(1, 4, figsize=(FigSize.large[0] * 4.2, FigSize.large[1] * 1.15))

    # 1. Stage 1 learning curve
    x_pre, pre = pretrain_curves()
    if x_pre is not None:
        for key, label, color in [('cong', 'congruent', COL['cong']),
                                  ('incong', 'incongruent', COL['incong']),
                                  ('all', 'all trials', 'k')]:
            if key in pre:
                band(axes[0], x_pre, pre[key], label, color)
        axes[0].axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
        axes[0].set_xlabel('Training trial')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Stage 1: learning (weights plastic)', fontsize=6)
        axes[0].legend(fontsize=5)

    # 2. Stage 2 accuracy over the session — weights frozen, so only Z can drift
    x_test = curves[0]['learning_x']
    for key, label, color in [('cong', 'congruent', COL['cong']),
                              ('incong', 'incongruent', COL['incong']),
                              ('all', 'all trials', 'k')]:
        band(axes[1], x_test, _stack_curve(curves, 'learning', key), label, color)
    axes[1].axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Test trial')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Stage 2: frozen weights', fontsize=6)
    axes[1].legend(fontsize=5)

    # 3 & 4. RT distributions
    rt_x = curves[0]['rt_bins']
    for key, label, color in [('cong', 'congruent', COL['cong']),
                              ('incong', 'incongruent', COL['incong'])]:
        band(axes[2], rt_x, _stack_curve(curves, 'rt_density', key), label, color)
    axes[2].set_xlabel('RT (timesteps, interpolated)')
    axes[2].set_ylabel('Density')
    axes[2].set_title('RT by congruency', fontsize=6)
    axes[2].legend(fontsize=5)

    for key, label, color in [('correct', 'correct', COL['correct']),
                              ('error', 'error', COL['error'])]:
        band(axes[3], rt_x, _stack_curve(curves, 'rt_density', key), label, color)
    axes[3].set_xlabel('RT (timesteps, interpolated)')
    axes[3].set_ylabel('Density')
    axes[3].set_title('RT by outcome', fontsize=6)
    axes[3].legend(fontsize=5)

    fig.suptitle(f'Learning and reaction times — {len(curves)} seeds, p(congruent)={p}', fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, 'group_learning_and_rt.pdf')
    fig.savefig(path, bbox_inches='tight')
    return path


def fig_within_trial(curves, out_dir, p):
    """Accuracy and evidence accumulation across timesteps within a trial."""
    fig, axes = plt.subplots(1, 3, figsize=(FigSize.large[0] * 3.2, FigSize.large[1] * 1.15))
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
    axes[0].set_title('Accuracy builds within the trial', fontsize=6)
    axes[0].legend(fontsize=5)

    for key, lbl, color, ls in [('correct_cong', 'correct, cong', COL['cong'], '-'),
                                ('error_cong', 'error, cong', COL['cong'], '--'),
                                ('correct_incong', 'correct, incong', COL['incong'], '-'),
                                ('error_incong', 'error, incong', COL['incong'], '--')]:
        band(axes[1], ts, _stack_curve(curves, 'accum', key), lbl, color, linestyle=ls)
    axes[1].axhline(0, color='k', linewidth=0.7, alpha=0.3)
    axes[1].set_xticks(ts)
    axes[1].set_xlabel('Timestep within trial')
    axes[1].set_ylabel('Output toward final choice\n(sign-normalised)')
    axes[1].set_title('Evidence accumulation (BPL analogue)', fontsize=6)
    axes[1].legend(fontsize=5)

    # Congruency effect on accuracy as it develops within the trial
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
    axes[2].set_title('Congruency cost over time', fontsize=6)
    axes[2].legend(fontsize=5)

    fig.suptitle(f'Within-trial dynamics — {len(curves)} seeds, p(congruent)={p}', fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, 'group_within_trial.pdf')
    fig.savefig(path, bbox_inches='tight')
    return path


def fig_history(effects, out_dir, p):
    """Sequential congruency: all four history cells, and the repetition control."""
    fig, axes = plt.subplots(1, 4, figsize=(FigSize.large[0] * 4.2, FigSize.large[1] * 1.2))
    shades = ['#c6dbef', '#6baed6', '#3182bd', '#08519c']
    hist = ['CC', 'CI', 'IC', 'II']

    bars_with_seeds(axes[0], [(_stack(effects, f'acc_{h}_to_I'), h, c) for h, c in zip(hist, shades)],
                    'Accuracy', connect=True)
    axes[0].set_title('History → incongruent', fontsize=6)

    bars_with_seeds(axes[1], [(_stack(effects, f'rt_{h}_to_I'), h, c) for h, c in zip(hist, shades)],
                    'RT (timesteps)', connect=True)
    axes[1].set_title('History → incongruent', fontsize=6)

    bars_with_seeds(axes[2], [(_stack(effects, f'rt_{h}_to_C'), h, c) for h, c in zip(hist, shades)],
                    'RT (timesteps)', connect=True)
    axes[2].set_title('History → congruent', fontsize=6)

    bars_with_seeds(axes[3],
                    [(_stack(effects, 'sce_acc_switch'), 'acc\nswitch', COL['cong']),
                     (_stack(effects, 'sce_acc_repeat'), 'acc\nrepeat', COL['incong']),
                     (_stack(effects, 'sce_rt_switch'),  'RT\nswitch',  COL['far_cong']),
                     (_stack(effects, 'sce_rt_repeat'),  'RT\nrepeat',  COL['far_incong'])],
                    'Sequential congruency effect', baseline=0.0)
    axes[3].set_title('Gratton effect vs. repetition', fontsize=6)

    fig.suptitle(f'Trial history (labels are t-2,t-1; post-correct only) — '
                 f'{len(effects)} seeds, p(congruent)={p}', fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, 'group_history.pdf')
    fig.savefig(path, bbox_inches='tight')
    return path


def fig_post_error(effects, out_dir, p):
    """Post-error adaptation and what drives the control update."""
    fig, axes = plt.subplots(1, 4, figsize=(FigSize.large[0] * 4.2, FigSize.large[1] * 1.2))

    bars_with_seeds(axes[0],
                    [(_stack(effects, 'pes_BI'), 'B incong', COL['incong']),
                     (_stack(effects, 'pes_BC'), 'B cong',   COL['cong'])],
                    'Post-error slowing (RT)', baseline=0.0)
    axes[0].set_title('Post-error slowing', fontsize=6)

    bars_with_seeds(axes[1],
                    [(_stack(effects, 'pia_BI'), 'B incong', COL['incong']),
                     (_stack(effects, 'pia_BC'), 'B cong',   COL['cong'])],
                    'Post-error accuracy change', baseline=0.0)
    axes[1].set_title('Post-error accuracy', fontsize=6)

    bars_with_seeds(axes[2],
                    [(_stack(effects, 'focus_in_after_near_err'), 'after\nnear err', COL['near_incong']),
                     (_stack(effects, 'focus_in_after_far_err'),  'after\nfar err',  COL['far_incong'])],
                    'Inherited Z focus', connect=True)
    axes[2].set_title('Control state after errors', fontsize=6)

    bars_with_seeds(axes[3],
                    [(_stack(effects, 'dfocus_near_err'),  'near\nerr',  COL['near_incong']),
                     (_stack(effects, 'dfocus_far_err'),   'far\nerr',   COL['far_incong']),
                     (_stack(effects, 'dfocus_near_corr'), 'near\ncorr', COL['near_cong']),
                     (_stack(effects, 'dfocus_far_corr'),  'far\ncorr',  COL['far_cong'])],
                    'Δ Z focus (update)', baseline=0.0)
    axes[3].set_title('What drives the update', fontsize=6)

    fig.suptitle(f'Post-error effects (incongruent trial A) — {len(effects)} seeds, '
                 f'p(congruent)={p}', fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, 'group_post_error.pdf')
    fig.savefig(path, bbox_inches='tight')
    return path


def fig_proportion_congruent(out_dir):
    """Congruency effect as a function of how common congruent trials are."""
    levels, acc, rt = [], [], []
    for p in P_CONGRUENT_LEVELS:
        effects, _ = collect_sessions(p)
        if not effects:
            continue
        levels.append(p)
        acc.append(_stack(effects, 'cong_effect_acc'))
        rt.append(_stack(effects, 'cong_effect_rt'))
    if not levels:
        return None

    acc, rt = np.array(acc), np.array(rt)   # (n_levels, n_seeds)
    fig, axes = plt.subplots(1, 2, figsize=(FigSize.large[0] * 2.2, FigSize.large[1] * 1.15))
    for ax, arr, ylabel in [(axes[0], acc, 'Congruency effect (accuracy)'),
                            (axes[1], rt,  'Congruency effect (RT)')]:
        for s in range(arr.shape[1]):                       # one line per subject
            ax.plot(levels, arr[:, s], color='k', alpha=0.16, linewidth=0.5, zorder=2)
        mu = arr.mean(axis=1)
        se = arr.std(axis=1, ddof=1) / np.sqrt(arr.shape[1])
        ax.errorbar(levels, mu, yerr=se, marker='o', markersize=4, color=COL['incong'],
                    capsize=3, linewidth=1.2, zorder=3)
        ax.set_xlabel('P(congruent)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(levels)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'List-wide proportion congruent — {acc.shape[1]} seeds, lines are subjects',
                 fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, 'group_proportion_congruent.pdf')
    fig.savefig(path, bbox_inches='tight')
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main(p=DEFAULT_P):
    out_dir = os.path.dirname(os.path.dirname(result_path(0, p)))
    os.makedirs(out_dir, exist_ok=True)
    print(f'Building group figures for p_congruent={p}')

    effects, curves = collect_sessions(p)
    if not effects:
        print('No results found — run flanker_sweep.py first.')
        return

    for path in [fig_fingerprint(effects, out_dir, p),
                 fig_learning_and_rt(curves, out_dir, p),
                 fig_within_trial(curves, out_dir, p),
                 fig_history(effects, out_dir, p),
                 fig_post_error(effects, out_dir, p),
                 fig_proportion_congruent(out_dir)]:
        if path:
            print(f'Exported: {path}')


def p_from_argv(default=DEFAULT_P):
    """Congruency level from the command line, if one was given.

    Run inside a VS Code interactive window or a Jupyter kernel, sys.argv belongs to
    the kernel rather than to this script (e.g. '--f=...kernel-abc.json'), so accept
    only a bare numeric argument and otherwise fall back to the default.
    """
    for arg in sys.argv[1:]:
        try:
            return float(arg)
        except ValueError:
            continue
    return default


if __name__ == '__main__':
    main(p_from_argv())
