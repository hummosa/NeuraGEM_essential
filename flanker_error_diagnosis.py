"""
flanker_error_diagnosis.py — why the model has no post-error control adjustment.

Three human post-error signatures were meant to be here (see docs/flanker_regression.md):
PES (slower), PIA (more accurate), PERI (less interference). The model reproduces PES and
fails the other two — PIA and PERI both come out *reliably negative*, meaning performance
gets worse after an error rather than better.

This module tests the explanation, and the answer is not the optimizer:

1. Adam vs SGD. The latent optimizer was suspected because Adam's momentum smears the
   control signal across trials. It does — switching to SGD roughly doubles the lag-1
   conflict-adaptation contrast and leaves lag-2 alone, which was the other open problem.
   But PIA and PERI are unchanged, so momentum was never their cause.

2. What an error actually teaches the latent. `arrow_noise_std` is 1.3 against a signal
   of 1.0, so on a large minority of trials the *target slot's own samples* point the
   wrong way. Two different things therefore cause an error:
       - the centre slot misled (noise-driven), where attending it less genuinely lowers
         this trial's prediction error, or
       - the centre was fine and the flankers won, where attending it more is the fix.
   The latent update minimises this trial's prediction error, so it cannot tell an error
   caused by "control was too low" from one caused by "the attended slot was unlucky".
   Roughly two thirds of errors are the noise-driven kind, and they generate a *negative*
   focus update several times larger than the positive update the flanker-driven ones
   produce. The average is therefore negative, and PIA and PERI invert.

The gradient panel is the load-bearing one: dL/dZ on the centre unit flips sign between
the two kinds of error, so this is visible in the learning signal itself, not only in its
downstream consequences.

Run:
    python flanker_error_diagnosis.py                 # spatial_steep, p=0.5
    python flanker_error_diagnosis.py --p 0.5 --variant spatial_steep
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_rel

from plot_style import FigSize

from flanker_figure_utils import COL, bars_with_seeds, collect_effects, save, sweep_root, sweep_run, _stack

#: The two runs this figure contrasts, as (run name on disk, label). Unlike the other
#: scripts this one reads TWO runs at once, so it cannot follow RUN_NAME — name the pair
#: here or pass runs=[...] . Any pair works: the labels are used verbatim in the panels.
#: e.g. [('sweep_td03', 'noise 1.3'), ('sweep_td03_n10', 'noise 1.0')] to test the noise
#: account directly. See flanker_sweep.SWEEP_RUNS for what each run contains.
DEFAULT_RUNS = [('sweep_v1', 'Adam'), ('sweep_sgd', 'SGD')]
ADAM_RUN, SGD_RUN = DEFAULT_RUNS

DEFAULT_VARIANT, DEFAULT_P = 'spatial_steep', 0.5


def load_both(p=DEFAULT_P, variant=DEFAULT_VARIANT, runs=None):
    """Per-seed effects from two runs, keyed by their labels. Same seeds, same task."""
    out = {}
    for run_name, label in (runs or DEFAULT_RUNS):
        with sweep_run(run_name):
            print(f'  [{label}] reading {run_name}')
            out[label] = collect_effects(p, variant)
    return out


# ── Statistics ────────────────────────────────────────────────────────────────

def compare(effects, key):
    """Both runs on one measure, plus the paired test between them."""
    first, second = list(effects)
    a, s = _stack(effects[first], key), _stack(effects[second], key)
    ok = ~np.isnan(a) & ~np.isnan(s)
    return dict(
        adam=a, sgd=s,
        p_adam=ttest_1samp(a[~np.isnan(a)], 0)[1],
        p_sgd=ttest_1samp(s[~np.isnan(s)], 0)[1],
        p_paired=ttest_rel(s[ok], a[ok])[1],
    )


def report(effects):
    """Print the comparison that the figure summarises."""
    rows = [('pia_BI', 'PIA — post-error accuracy'),
            ('peri', 'PERI — interference drop'),
            ('pes_BI', 'PES — post-error slowing'),
            ('lag1_contrast_acc', 'lag-1 conflict adaptation (acc)'),
            ('lag2_contrast_acc', 'lag-2 conflict adaptation (acc)'),
            ('dfocus_err_noisy', 'delta focus, noise-driven error'),
            ('dfocus_err_clean', 'delta focus, flanker-driven error'),
            ('frac_err_noisy', 'fraction of errors that are noise-driven')]
    first, second = list(effects)
    print(f'  {"measure":<40}{first:>10}{"p":>9}{second:>10}{"p":>9}{"paired p":>10}')
    for key, label in rows:
        c = compare(effects, key)
        print(f'  {label:<40}{c["adam"].mean():10.4f}{c["p_adam"]:9.4f}'
              f'{c["sgd"].mean():10.4f}{c["p_sgd"]:9.4f}{c["p_paired"]:10.4f}')


def _scaled(effects, key, reference=None):
    """
    Per-seed values divided by one common SD, so two measures in different units can
    share an axis. PIA is an accuracy difference and PERI is in timesteps; without this
    the PIA bars are invisible next to PERI. Both optimizers are scaled by the same
    (reference) SD, so the comparison between them is preserved.
    """
    sd = np.nanstd(_stack(effects[reference or list(effects)[0]], key), ddof=1)
    return {label: _stack(rows, key) / sd for label, rows in effects.items()}


# ── Figure ────────────────────────────────────────────────────────────────────

def fig_error_diagnosis(effects, out_dir, p=DEFAULT_P, variant=DEFAULT_VARIANT):
    """Six panels: the failure, the optimizer test, and the mechanism behind it."""
    fig, axes = plt.subplots(2, 3, figsize=FigSize.grid(2, 3, panel=FigSize.large))
    ax = axes.ravel()
    first, second = list(effects)
    adam, sgd = effects[first], effects[second]   # first run vs second, by label order

    # A. The failure, and that it survives the optimizer change.
    pia, peri = _scaled(effects, 'pia_BI'), _scaled(effects, 'peri')
    bars_with_seeds(ax[0],
                    [(pia[first],  f'PIA\n{first}',  COL['neutral']),
                     (pia[second],  f'PIA\n{second}', COL['incong']),
                     (peri[first],  f'PERI\n{first}', COL['neutral']),
                     (peri[second], f'PERI\n{second}', COL['incong'])],
                    'Effect / SD across seeds', baseline=0.0,
                    title='Post-error signatures fail')
    ax[0].text(0.02, 0.04, 'human direction = +', transform=ax[0].transAxes,
               fontsize=4.5, color=COL['neutral'])

    # B. The optimizer did fix the other problem, which is why it is the right control.
    bars_with_seeds(ax[1],
                    [(_stack(adam, 'lag1_contrast_acc'), f'lag 1\n{first}',  COL['neutral']),
                     (_stack(sgd,  'lag1_contrast_acc'), f'lag 1\n{second}', COL['incong']),
                     (_stack(adam, 'lag2_contrast_acc'), f'lag 2\n{first}',  COL['neutral']),
                     (_stack(sgd,  'lag2_contrast_acc'), f'lag 2\n{second}', COL['incong'])],
                    'History contrast (accuracy)', baseline=0.0,
                    title='Momentum did suppress lag 1')

    # C. What actually goes wrong on an error: the target slot's own samples mislead.
    bars_with_seeds(ax[2],
                    [(_stack(sgd, 'centre_ev_incong_corr'), 'centre\ncorrect', COL['cong']),
                     (_stack(sgd, 'centre_ev_incong_err'),  'centre\nerror',   COL['incong']),
                     (_stack(sgd, 'flank_ev_incong_corr'),  'flankers\ncorrect', COL['far_cong']),
                     (_stack(sgd, 'flank_ev_incong_err'),   'flankers\nerror',   COL['far_incong'])],
                    'Evidence delivered (signed)', baseline=0.0,
                    title='Errors follow a misleading centre')

    # D. The mechanism: the update depends on which slot happened to be right.
    bars_with_seeds(ax[3],
                    [(_stack(sgd, 'dfocus_err_noisy'),  'error\ncentre bad',  COL['error']),
                     (_stack(sgd, 'dfocus_err_clean'),  'error\ncentre ok',   COL['far_incong']),
                     (_stack(sgd, 'dfocus_corr_noisy'), 'correct\ncentre bad', COL['far_cong']),
                     (_stack(sgd, 'dfocus_corr_clean'), 'correct\ncentre ok',  COL['cong'])],
                    'Δ Z focus (this trial\'s update)', baseline=0.0, rotation=30,
                    title='Only unlucky errors loosen control')

    # E. The learning signal itself, before it becomes an update.
    bars_with_seeds(ax[4],
                    [(_stack(sgd, 'zgrad_centre_err_noisy'), 'error\ncentre bad', COL['error']),
                     (_stack(sgd, 'zgrad_centre_err_clean'), 'error\ncentre ok',  COL['far_incong'])],
                    'dL/dZ at the centre unit', baseline=0.0,
                    title='The gradient flips sign')
    ax[4].text(0.5, 0.02, 'positive gradient → descent lowers centre weight',
               transform=ax[4].transAxes, fontsize=4.5, ha='center', color=COL['neutral'])

    # F. Why the average comes out negative: the bad kind is the common kind.
    # Each error type's contribution to the average update is its share times its size.
    # The noise-driven kind is both the more common and the larger, so the sum is negative.
    share = _stack(sgd, 'frac_err_noisy')
    noisy_part = share * _stack(sgd, 'dfocus_err_noisy')
    clean_part = (1.0 - share) * _stack(sgd, 'dfocus_err_clean')
    bars_with_seeds(ax[5],
                    [(noisy_part, 'noise-driven\n(64% of errors)', COL['error']),
                     (clean_part, 'flanker-driven\n(36%)', COL['far_incong']),
                     (noisy_part + clean_part, 'net update\nafter an error', COL['neutral'])],
                    'Contribution to Δ focus', baseline=0.0, rotation=30,
                    title='The common error is the wrong teacher')

    fig.suptitle(f'Why errors do not recruit control — {variant}, p(congruent)={p}, '
                 f'{first} vs {second}, {len(sgd)} seeds each', fontsize=7)
    fig.tight_layout()
    return save(fig, f'{out_dir}/error_diagnosis.pdf')


# ── The circularity: why PIA stays negative even when errors do recruit control ──

LAGS = np.arange(-2, 3)


def event_locked(run='sweep_td03_n10', p=DEFAULT_P, variant=DEFAULT_VARIANT, n_bins=8):
    """
    Focus and accuracy on the trials surrounding an incongruent error vs a correct one.

    `focus_in` is the state a trial *inherited*, so lag 0 is what the error trial itself
    started from, and lag +1 is what the next trial inherited — the step from 0 to +1
    therefore contains the error's own update. Also returns the accuracy-versus-inherited-
    focus curve, which converts a gap in control into a gap in performance.
    """
    import flanker_sweep as fs_mod
    from flanker_analyses import extract_trials
    from flanker_metrics import condition_masks

    out = {k: [] for k in ('focus_err', 'focus_corr', 'acc_err', 'acc_corr',
                           'curve_x', 'curve_y', 'start_gap', 'upd_err', 'upd_corr')}
    with sweep_run(run):
        for seed in range(10):
            res = fs_mod.load_result(seed, p, variant)
            tr = extract_trials(res['train_logger'], res['config'])
            m = condition_masks(tr)
            n, idx = tr['n_trials'], np.arange(tr['n_trials'])
            foc, acc = tr['focus_in'], tr['correct_at_decision'].astype(float)
            inside = (idx >= 2) & (idx < n - 2)
            for key, om in (('err', m['err']), ('corr', m['corr'])):
                events = idx[m['incong'] & om & inside]
                out['focus_' + key].append([np.nanmean(foc[events + l]) for l in LAGS])
                out['acc_' + key].append([np.nanmean(acc[events + l]) for l in LAGS])
            out['start_gap'].append(np.nanmean(foc[m['incong'] & m['err']])
                                    - np.nanmean(foc[m['incong'] & m['corr']]))
            out['upd_err'].append(np.nanmean(tr['delta_focus'][m['incong'] & m['err']]))
            out['upd_corr'].append(np.nanmean(tr['delta_focus'][m['incong'] & m['corr']]))

            # How much accuracy does a given inherited focus buy, on incongruent trials?
            inc = m['incong'] & ~np.isnan(foc)
            edges = np.nanquantile(foc[inc], np.linspace(0, 1, n_bins + 1))
            centres, means = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = inc & (foc >= lo) & (foc < hi)
                if sel.sum() > 20:
                    centres.append(np.nanmean(foc[sel]))
                    means.append(acc[sel].mean())
            out['curve_x'].append(centres)
            out['curve_y'].append(means)
    return {k: (v if k in ('curve_x', 'curve_y') else np.array(v)) for k, v in out.items()}


def _band(ax, x, arr, colour, label):
    mu = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    ax.errorbar(x, mu, yerr=se, marker='o', markersize=3.5, capsize=2, linewidth=1.2,
                color=colour, label=label)
    return mu


def fig_circularity(ev, out_dir, p=DEFAULT_P, variant=DEFAULT_VARIANT):
    """
    Why the model shows no post-error improvement, in four steps.

    The control deficit *precedes* the error, so the error's own correction — which is
    real, and larger than the one a correct trial produces — starts from further back and
    does not catch up within a single trial.
    """
    fig, axes = plt.subplots(1, 4, figsize=FigSize.row(4, panel=FigSize.large))
    n_seeds = ev['start_gap'].size

    # A. The state around the event. This is the whole argument in one panel.
    mu_c = _band(axes[0], LAGS, ev['focus_corr'], COL['cong'], 'after a correct trial')
    mu_e = _band(axes[0], LAGS, ev['focus_err'], COL['error'], 'after an ERROR')
    axes[0].axvline(0, color='k', linewidth=0.7, alpha=0.35)
    lo0, hi0 = axes[0].get_ylim()                 # headroom for the annotations below
    axes[0].set_ylim(lo0 - 0.30 * (hi0 - lo0), hi0)
    axes[0].set_xticks(LAGS)
    axes[0].set_xlabel('Trial, relative to the event')
    axes[0].set_ylabel('Control state inherited\n(Z focus on target)')
    axes[0].set_title('1. The deficit comes first', fontsize=6)
    axes[0].legend(fontsize=4.5, loc='upper left')
    axes[0].annotate('already low before\nthe error happens', xy=(-1.05, mu_e[1]),
                     xytext=(-2.05, mu_e[1] - 0.055), fontsize=4.5, color=COL['error'],
                     arrowprops=dict(arrowstyle='->', color=COL['error'], lw=0.6))
    axes[0].annotate('the error corrects it,\nbut lands short', xy=(1.05, mu_e[3]),
                     xytext=(1.15, mu_e[2] - 0.03), fontsize=4.5, color=COL['neutral'],
                     ha='center',
                     arrowprops=dict(arrowstyle='->', color=COL['neutral'], lw=0.6))

    # B. The same thing as two numbers: the gap before, and the gap left over.
    start = ev['start_gap'].mean()
    up_e, up_c = ev['upd_err'].mean(), ev['upd_corr'].mean()
    residual = start + up_e - up_c
    gaps = ev['start_gap']
    resid_seeds = ev['start_gap'] + ev['upd_err'] - ev['upd_corr']
    for i, (vals, label, colour) in enumerate([(gaps, 'when the error\nhappens', COL['error']),
                                               (resid_seeds, 'left over for\nthe next trial', COL['neutral'])]):
        se = vals.std(ddof=1) / np.sqrt(vals.size)
        axes[1].bar(i, vals.mean(), color=colour, alpha=0.8, width=0.55)
        axes[1].errorbar(i, vals.mean(), yerr=se, color='k', capsize=3, linewidth=1)
        axes[1].scatter(np.full(vals.size, i) + (np.random.default_rng(0).random(vals.size) - .5) * .18,
                        vals, s=5, color='k', alpha=0.45, zorder=5, linewidths=0)
    axes[1].axhline(0, color='k', linewidth=0.7, alpha=0.6)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['when the error\nhappens', 'left over for\nthe next trial'], fontsize=5)
    axes[1].set_ylabel('Control gap vs a correct trial')
    axes[1].set_title('2. The correction undershoots', fontsize=6)
    lo, hi = axes[1].get_ylim()
    axes[1].set_ylim(lo, hi + 0.45 * (hi - lo))
    axes[1].annotate('', xy=(0.78, residual), xytext=(0.22, start),
                     arrowprops=dict(arrowstyle='->', color=COL['cong'], lw=1.0))
    axes[1].text(0.5, axes[1].get_ylim()[1],
                 f'the error updates {up_e:+.3f}\n(a correct trial: {up_c:+.3f})\n'
                 f'closing only {100 * (1 - residual / start):.0f}% of the gap',
                 fontsize=4.5, ha='center', va='top', color=COL['neutral'])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # C. What that costs behaviourally — this is PIA, drawn rather than tabulated.
    keep = [0, 1, 3, 4]
    _band(axes[2], LAGS[keep], ev['acc_corr'][:, keep], COL['cong'], 'after a correct trial')
    _band(axes[2], LAGS[keep], ev['acc_err'][:, keep], COL['error'], 'after an ERROR')
    axes[2].axvspan(-0.4, 0.4, color='k', alpha=0.07)
    axes[2].set_xticks(LAGS)
    axes[2].set_xlabel('Trial, relative to the event')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('3. So the next trial is worse (PIA)', fontsize=6)
    axes[2].text(0, axes[2].get_ylim()[0], ' event', fontsize=4.5, color=COL['neutral'],
                 va='bottom', ha='center')

    # D. The exchange rate between control and accuracy, with both landing points marked.
    grid = np.linspace(0, 1, 8)
    xs = np.array([np.interp(grid, np.linspace(0, 1, len(cx)), cx) for cx in ev['curve_x']])
    ys = np.array([np.interp(grid, np.linspace(0, 1, len(cy)), cy) for cy in ev['curve_y']])
    mu_x, mu_y = xs.mean(axis=0), ys.mean(axis=0)
    se_y = ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])
    axes[3].plot(mu_x, mu_y, color=COL['neutral'], linewidth=1.2, marker='o', markersize=2.5)
    axes[3].fill_between(mu_x, mu_y - se_y, mu_y + se_y, color=COL['neutral'], alpha=0.18,
                         linewidth=0)
    top = axes[3].get_ylim()[1]
    for key, colour, label in (('focus_err', COL['error'], 'after error'),
                               ('focus_corr', COL['cong'], 'after correct')):
        state = ev[key].mean(axis=0)[3]          # what the NEXT trial inherited
        axes[3].axvline(state, color=colour, linewidth=0.9, linestyle='--')
        axes[3].text(state, top, ' ' + label, rotation=90, fontsize=4.5, color=colour,
                     va='top')
    axes[3].set_xlabel('Control state inherited')
    axes[3].set_ylabel('Accuracy, incongruent trials')
    axes[3].set_title('4. What the gap costs', fontsize=6)
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['right'].set_visible(False)

    fig.suptitle('Why an error does not improve the next trial: the control deficit precedes '
                 f'the error  —  {variant}, p(congruent)={p}, {n_seeds} seeds', fontsize=7)
    fig.tight_layout()
    return save(fig, out_dir + '/pia_circularity.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(p=DEFAULT_P, variant=DEFAULT_VARIANT, runs=None):
    effects = load_both(p, variant, runs)
    print(f'\nPost-error diagnosis — {variant}, p(congruent)={p}\n')
    report(effects)
    fig_error_diagnosis(effects, sweep_root(), p, variant)


def args_from_argv():
    p, variant, argv = DEFAULT_P, DEFAULT_VARIANT, sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == '--p' and i + 1 < len(argv):
            p = float(argv[i + 1])
        elif arg == '--variant' and i + 1 < len(argv):
            variant = argv[i + 1]
    return p, variant


if __name__ == '__main__':
    main(*args_from_argv())
