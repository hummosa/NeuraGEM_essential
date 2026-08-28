"""
flanker_rt_threshold_sweep.py — how much of the scorecard is `rt_threshold = 0.5`?

`rt_threshold` is a **post-hoc analysis parameter**, not a simulation one. It is applied by
`flanker_analyses.extract_trials()` to the `train_logger` already stored in every result
pickle, so every threshold can be re-applied to the runs on disk. Nothing here retrains,
and nothing here submits a job.

It is also not only an RT knob. The threshold defines the decision point, so it also sets
`correct_at_decision` — every one of the 11 human signatures in `flanker_metrics.SIGNATURES`
depends on it, the accuracy ones included.

What this script produces
─────────────────────────
For each (arm x noise level x seed x threshold) it recomputes `session_effects` and then,
across seeds, the same PASS / null / OPP verdict `flanker_sweep_figures.fig_scorecard`
draws. `_effect_size` is imported from that module rather than reimplemented, so the
verdict rule here is the scorecard's rule by construction and cannot drift from it.

Two threshold families are run:

    abs     a fixed absolute |output| level, 0.2 … 0.8 — what the project uses today.
    quant   a per-session level set at a fixed quantile of that session's own
            max|output| distribution, so every session is scored at the same
            *undecided rate* instead of the same absolute criterion. This is the
            robustness check for cross-arm comparisons: the jitter arms run a higher
            mean focus, so a fixed 0.5 need not be the same decision criterion in each.

Alongside the headline numbers it computes **decided-only** versions of the four RT-based
signatures. Undecided trials are all assigned `rt = arrows_duration` by convention (see
`_interpolated_rt`, which documents why dropping or extrapolating them was rejected), and
that pile-up grows with the threshold — so the decided-only column says how much of a
signature's threshold dependence is speed and how much is failure-to-respond. It is a
diagnostic, not a proposed change of convention.

Usage
─────
    .venv/bin/python flanker_rt_threshold_sweep.py pilot      # baseline/noise09, ~20 s
    .venv/bin/python flanker_rt_threshold_sweep.py sweep      # 4 arms x 5 levels, ~3 min
    .venv/bin/python flanker_rt_threshold_sweep.py report     # tables + figures from the cache
    .venv/bin/python flanker_rt_threshold_sweep.py hypotheses # the H1-H6 tables
    .venv/bin/python flanker_rt_threshold_sweep.py all        # the three above, in order

    .venv/bin/python flanker_rt_threshold_sweep.py figures            # baseline arm
    .venv/bin/python flanker_rt_threshold_sweep.py figures jit_pc52   # another arm

`figures` is the separate one: instead of putting the threshold on an axis, it redraws the
project's own scorecard, noise ladder and regression forest once per threshold into
`by_threshold/`, with the threshold in every filename, so the series can be flipped
through.

Everything is written to `exports/flanker_random/rt_threshold/`, a new folder — the
existing run directories and their figures are never touched.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_style
plot_style.set_plot_style()
from plot_style import FigSize

import flanker_sweep
from flanker_analyses import extract_trials
from flanker_figure_utils import COL, save
from flanker_metrics import SIGNATURES, condition_masks, session_effects, _mean
from flanker_sweep_config import ARMS, NOISE_LADDER, RT_THRESHOLD, SEEDS
# The scorecard's own effect-size function. Imported, not copied: the whole point of this
# script is to report the verdicts fig_scorecard would draw, so the two must not diverge.
from flanker_sweep_figures import _effect_size


# ── What is swept ─────────────────────────────────────────────────────────────

#: Absolute |output| thresholds. 0.5 is the value in `flanker_sweep_config.RT_THRESHOLD`.
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

#: The five levels the brief's pilot table used, so `pilot` reproduces it exactly.
PILOT_THRESHOLDS = (0.20, 0.35, 0.50, 0.65, 0.80)

#: Per-session thresholds placed at these quantiles of the session's own max|output|,
#: i.e. these undecided rates by construction. 0.10 is close to the baseline arm's rate
#: at an absolute 0.5, which makes it the like-for-like comparison.
QUANTILES = (0.05, 0.10, 0.20)

ARM_ORDER = ('nojit_pc52', 'jit_pc52', 'nojit_pc58', 'jit_pc58')
BASELINE_ARM = 'nojit_pc52'

OUT_DIR = './exports/flanker_random/rt_threshold'
EFFECTS_CSV = f'{OUT_DIR}/effects.csv'
VERDICTS_CSV = f'{OUT_DIR}/verdicts.csv'
FLIPS_CSV = f'{OUT_DIR}/flips.csv'

SIG_KEYS = [k for k, _, _, _ in SIGNATURES]
SIG_SIGN = {k: s for k, _, s, _ in SIGNATURES}
SIG_LABEL = {k: lbl for k, lbl, _, _ in SIGNATURES}

#: Extra per-cell measures carried alongside the signatures.
EXTRA_KEYS = ['undecided_frac', 'undecided_frac_cong', 'undecided_frac_incong',
              'acc_overall', 'focus_all', 'gate_peak']

#: The RT-based signatures, and the decided-only counterpart computed for each.
RT_SIGNATURES = ('cong_effect_rt', 'dist_effect_rt_incong', 'pes_BI', 'peri')

#: The accuracy row of `fig_effects_vs_threshold`, column-matched to RT_SIGNATURES by
#: construct wherever a counterpart exists: congruency, distance within incongruent,
#: post-error. The fourth column has no cross-family pair, so it carries whichever
#: signature in that family moves most — `peri` on RT, `dist_effect_acc_cong` on accuracy.
ACC_SIGNATURES = ('cong_effect_acc', 'dist_effect_acc_incong', 'pia_BI',
                  'dist_effect_acc_cong')


# ── Per-session measures the metrics module does not already provide ──────────

def response_amplitude(trials, config):
    """
    Per-trial max |output| inside the response window — the quantity the threshold cuts.

    `_interpolated_rt` searches `|output|` from `config.response_start_timestep` onward and
    calls a trial decided if it ever exceeds the threshold, so this maximum is a sufficient
    statistic for `decided`: a session's undecided fraction at threshold t is exactly the
    fraction of this array at or below t. That makes it the right thing to look at when
    asking whether one absolute threshold means the same thing in two arms.
    """
    a = np.abs(trials['output_traj'])[:, config.response_start_timestep:]
    return a.max(axis=1)


def decided_only_effects(trials):
    """
    The four RT-based signatures recomputed on trials that actually crossed threshold.

    Undecided trials are assigned `rt = arrows_duration` — one value above every decided
    RT — so as the threshold rises they form a growing high-RT pile-up in whichever cells
    fail most. Incongruent trials fail roughly four times more often than congruent ones,
    so that pile-up loads directly onto the congruency effect. These keys strip it out.

    `_interpolated_rt`'s docstring is explicit that dropping those trials was tried and
    rejected: censoring them biases incongruent RT downward and shrinks the effect. So the
    decided-only number is not a better estimate of the congruency effect — it is the
    lower bound that isolates how much of the threshold dependence is speed rather than
    failure to respond. The honest reading is that the truth sits between the two columns.
    """
    m = condition_masks(trials)
    rt, dec = trials['rt_interp'], m['decided']
    e = {}

    e['cong_effect_rt'] = _mean(rt, m['incong'] & dec) - _mean(rt, m['cong'] & dec)
    for cn in ('cong', 'incong'):
        e[f'dist_effect_rt_{cn}'] = (_mean(rt, m['near'] & m[cn] & dec)
                                     - _mean(rt, m['far'] & m[cn] & dec))

    # Post-error slowing and PERI, same cells as post_error_effects but decided-only.
    after_err = m['valid'] & m['p_incong'] & m['perr']
    after_cor = m['valid'] & m['p_incong'] & m['pc']
    e['pes_BI'] = (_mean(rt, after_err & m['incong'] & dec)
                   - _mean(rt, after_cor & m['incong'] & dec))
    ce_err = (_mean(rt, after_err & m['incong'] & dec)
              - _mean(rt, after_err & m['cong'] & dec))
    ce_cor = (_mean(rt, after_cor & m['incong'] & dec)
              - _mean(rt, after_cor & m['cong'] & dec))
    e['peri'] = ce_cor - ce_err
    return e


# ── The sweep ─────────────────────────────────────────────────────────────────

def session_row(trials, base):
    """One output row: every signature, the extras, and the decided-only RT columns."""
    e = session_effects(trials)
    row = dict(base)
    row.update({k: e.get(k, np.nan) for k in SIG_KEYS})
    row.update({k: e.get(k, np.nan) for k in EXTRA_KEYS})
    dec = decided_only_effects(trials)
    row.update({f'{k}__dec': dec.get(k, np.nan) for k in RT_SIGNATURES})
    return row


def sweep(arms=ARM_ORDER, ladder=NOISE_LADDER, thresholds=THRESHOLDS,
          quantiles=QUANTILES, seeds=None, verbose=True):
    """
    Recompute every signature at every threshold, for every run on disk.

    Each pickle is unpickled **once** and the threshold loop runs inside it: unpickling is
    ~0.14 s and `extract_trials` + `session_effects` together are ~0.03 s, so loading per
    threshold would cost 7x for nothing.

    Returns a long-ish wide DataFrame, one row per
    (arm, variant, seed, mode, threshold), which is the unit every later table groups over.
    """
    seeds = list(range(SEEDS)) if seeds is None else list(seeds)
    rows, t_start = [], time.time()

    for arm in arms:
        with flanker_sweep.use_run(f'factorial_{arm}'):
            for variant, noise in ladder:
                n_loaded = 0
                for seed in seeds:
                    res = flanker_sweep.load_result(seed, variant)
                    if res is None:
                        continue
                    n_loaded += 1
                    logger, config = res['train_logger'], res['config']
                    base = dict(arm=arm, variant=variant, noise=noise, seed=seed)

                    # The amplitude distribution is threshold-independent, so it is read
                    # once per session from any extraction and carried on every row.
                    trials0 = extract_trials(logger, config, rt_threshold=RT_THRESHOLD)
                    amp = response_amplitude(trials0, config)
                    amp_stats = {'amp_median': float(np.median(amp)),
                                 'amp_q10': float(np.quantile(amp, 0.10)),
                                 'amp_q25': float(np.quantile(amp, 0.25)),
                                 'amp_frac_below_0.5': float((amp <= 0.5).mean())}

                    jobs = ([('abs', t, t) for t in thresholds]
                            + [('quant', q, float(np.quantile(amp, q))) for q in quantiles])
                    for mode, nominal, thr in jobs:
                        trials = (trials0 if (mode == 'abs' and thr == RT_THRESHOLD)
                                  else extract_trials(logger, config, rt_threshold=thr))
                        row = session_row(trials, {**base, 'mode': mode,
                                                   'threshold': nominal, 'thr_abs': thr})
                        row.update(amp_stats)
                        rows.append(row)
                if verbose:
                    print(f'  {arm:12s} {variant:8s} {n_loaded:2d} seeds '
                          f'({time.time() - t_start:5.1f}s)')
    return pd.DataFrame(rows)


# ── Scoring: the verdict fig_scorecard would draw ─────────────────────────────

def verdict(values, sign):
    """
    PASS / null / OPP for one signature across seeds, by `fig_scorecard`'s exact rule.

    `_effect_size` divides by the across-seed SD and multiplies by the human sign, so the
    colour test `mean - 1.96*sem > 0` is a two-sided z-test of the across-seed mean at
    alpha = .05. Returns (verdict, d, sem, n).

    The middle verdict is spelled `n.s.` rather than `null`: `pandas.read_csv` treats the
    bare string "null" as a missing value, so a CSV written with it reads back as NaN and
    a whole row of non-significant cells silently vanishes from a pivot table.
    """
    d = _effect_size(np.asarray(values, dtype=float), sign)
    if len(d) < 2:
        return 'n/a', np.nan, np.nan, len(d)
    mean, sem = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    v = 'PASS' if mean - 1.96 * sem > 0 else 'OPP' if mean + 1.96 * sem < 0 else 'n.s.'
    return v, float(mean), float(sem), len(d)


def score(df, keys=None, suffix=''):
    """
    Collapse the per-seed table to one verdict per (arm, variant, mode, threshold, signature).

    `suffix` selects the decided-only columns (`'__dec'`) instead of the headline ones,
    so the same scoring path serves both and they cannot be scored differently.
    """
    keys = keys or SIG_KEYS
    out = []
    group_cols = ['arm', 'variant', 'noise', 'mode', 'threshold']
    for (arm, variant, noise, mode, thr), g in df.groupby(group_cols, sort=False):
        for key in keys:
            col = f'{key}{suffix}'
            if col not in g:
                continue
            vals = g[col].to_numpy(dtype=float)
            v, d, sem, n = verdict(vals, SIG_SIGN[key])
            out.append(dict(arm=arm, variant=variant, noise=noise, mode=mode,
                            threshold=thr, signature=key, label=SIG_LABEL[key],
                            verdict=v, d=d, d_sem=sem, n_seeds=n,
                            raw_mean=float(np.nanmean(vals)),
                            n_sign_match=int((vals * SIG_SIGN[key] > 0).sum()),
                            thr_abs=float(g['thr_abs'].mean()),
                            undecided=float(g['undecided_frac'].mean())))
    return pd.DataFrame(out)


def flip_table(verdicts, mode='abs'):
    """
    Every (arm, noise, signature) whose verdict is not constant over the threshold range.

    A conclusion that appears in one row of this table is threshold-dependent: the same
    seeds, the same simulation, a different post-hoc cut, a different answer.
    """
    v = verdicts[verdicts['mode'] == mode]
    rows = []
    for (arm, variant, noise, sig), g in v.groupby(['arm', 'variant', 'noise', 'signature'],
                                                   sort=False):
        g = g.sort_values('threshold')
        seq = list(g['verdict'])
        if len(set(seq)) <= 1:
            continue
        at_default = g.loc[np.isclose(g['threshold'], RT_THRESHOLD), 'verdict']
        row = dict(arm=arm, variant=variant, noise=noise, signature=sig,
                   label=SIG_LABEL[sig],
                   at_default=(at_default.iloc[0] if len(at_default) else 'n/a'),
                   verdicts='|'.join(seq),
                   # A signature that only softens (PASS -> null) is a weaker problem than
                   # one that reverses (PASS -> OPP); the two are separated here so the
                   # recommendation can weight them differently.
                   reverses=bool({'PASS', 'OPP'} <= set(seq)))
        for thr, vd in zip(g['threshold'], seq):
            row[f'thr_{thr:g}'] = vd
        rows.append(row)
    cols = ['arm', 'variant', 'noise', 'signature', 'label', 'at_default', 'reverses']
    out = pd.DataFrame(rows)
    if len(out):
        out = out[cols + [c for c in out.columns if c not in cols]]
    return out


def matched_counts(verdicts, mode='abs'):
    """
    Signatures passing / opposing at each (arm, noise, threshold) — the scorecard count.

    Restricted to the 11 registered signatures, so passing in a table that also carries
    the decided-only diagnostic rows cannot inflate the count past 11.
    """
    v = verdicts[(verdicts['mode'] == mode) & (verdicts['signature'].isin(SIG_KEYS))]
    g = v.groupby(['arm', 'variant', 'noise', 'threshold'], sort=False)
    out = g.apply(lambda x: pd.Series({
        'matched': int((x['verdict'] == 'PASS').sum()),
        'opp': int((x['verdict'] == 'OPP').sum()),
        'ns': int((x['verdict'] == 'n.s.').sum()),
        'undecided': float(x['undecided'].mean()),
        'thr_abs': float(x['thr_abs'].mean()),
    }), include_groups=False)
    return out.reset_index()


# ── Figures ───────────────────────────────────────────────────────────────────
#
# The 2x2 of arms borrows the flanker palette's own logic: hue carries the knob under
# test (blue = no jitter, red = jitter), shade carries p_corr[2]. So a figure about
# jitter reads as blue-vs-red at a glance, which is the comparison every arm panel is
# actually about.

ARM_COLORS = {'nojit_pc52': '#2166ac', 'nojit_pc58': '#92c5de',
              'jit_pc52': '#b2182b', 'jit_pc58': '#f4a582'}
ARM_LABELS = {'nojit_pc52': 'no jitter, .52', 'nojit_pc58': 'no jitter, .58',
              'jit_pc52': 'jitter, .52', 'jit_pc58': 'jitter, .58'}
NOISE_COLORS = plt.cm.viridis(np.linspace(0.05, 0.85, len(NOISE_LADDER)))


def _mark_default(ax):
    """A thin rule at the threshold the project currently uses."""
    ax.axvline(RT_THRESHOLD, color='k', linewidth=0.6, alpha=0.35, zorder=0)


def _line(ax, x, mu, sem, color, label, linestyle='-'):
    ax.plot(x, mu, color=color, linewidth=1.0, linestyle=linestyle, label=label)
    if sem is not None:
        ax.fill_between(x, mu - sem, mu + sem, color=color, alpha=0.18, linewidth=0)


def _seed_stats(df, col, group):
    """Mean and SEM across seeds of `col`, ordered by threshold."""
    g = df.groupby(group, sort=True)[col]
    return g.mean().index.to_numpy(dtype=float), g.mean().to_numpy(), \
        (g.std(ddof=1) / np.sqrt(g.count())).to_numpy()


def fig_undecided(df, out_dir):
    """
    The mechanism: how many trials the threshold pushes out of the decided set.

    Undecided trials are not dropped — they are pinned at `rt = arrows_duration`, above
    every decided RT. So the left panel is also a picture of how large a high-RT pile-up
    each threshold manufactures, and the right panel is whether the four arms get the
    same-sized pile-up from the same absolute cut. They do not.
    """
    d = df[df['mode'] == 'abs']
    fig, axes = plt.subplots(1, 2, figsize=FigSize.row(2))

    base = d[d['arm'] == BASELINE_ARM]
    for (variant, noise), color in zip(NOISE_LADDER, NOISE_COLORS):
        sub = base[base['variant'] == variant]
        x, mu, sem = _seed_stats(sub, 'undecided_frac', 'threshold')
        _line(axes[0], x, mu, sem, color, f'{noise}')
    axes[0].set_title(f'{BASELINE_ARM}, by stimulus noise', fontsize=6)
    axes[0].legend(title='arrow noise', fontsize=4.5, title_fontsize=4.5,
                   frameon=False, loc='upper left')

    for arm in ARM_ORDER:
        sub = d[(d['arm'] == arm) & (d['variant'] == 'noise09')]
        if not len(sub):
            continue
        x, mu, sem = _seed_stats(sub, 'undecided_frac', 'threshold')
        _line(axes[1], x, mu, sem, ARM_COLORS[arm], ARM_LABELS[arm])
    axes[1].set_title('noise09, by arm', fontsize=6)
    axes[1].legend(fontsize=4.5, frameon=False, loc='upper left')

    for ax in axes:
        _mark_default(ax)
        ax.set_xlabel('rt_threshold (|output|)')
        ax.set_ylabel('undecided fraction')
    fig.tight_layout()
    return save(fig, f'{out_dir}/fig_1_undecided.pdf')


def fig_signatures(verdicts, out_dir, variant='noise09'):
    """
    Every signature's effect size against threshold, one line per arm.

    The grey band is the region where the 95% CI would still contain zero at this n, so a
    line inside it is a `null` verdict and a line above it is `PASS`. Reading the crossings
    off this figure is the same operation `fig_scorecard` performs at a single threshold.
    """
    v = verdicts[(verdicts['mode'] == 'abs') & (verdicts['variant'] == variant)]
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=FigSize.grid(rows, cols), sharex=True)
    for ax, key in zip(axes.ravel(), SIG_KEYS):
        for arm in ARM_ORDER:
            sub = v[(v['arm'] == arm) & (v['signature'] == key)].sort_values('threshold')
            if not len(sub):
                continue
            x = sub['threshold'].to_numpy(dtype=float)
            _line(ax, x, sub['d'].to_numpy(), 1.96 * sub['d_sem'].to_numpy(),
                  ARM_COLORS[arm], ARM_LABELS[arm])
        # The decision boundary, drawn once from the baseline arm's n.
        n = float(v['n_seeds'].max())
        ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color=COL['null'],
                   alpha=0.15, linewidth=0, zorder=0)
        ax.axhline(0, color='k', linewidth=0.6, alpha=0.6, zorder=1)
        _mark_default(ax)
        ax.set_title(SIG_LABEL[key], fontsize=5)
    for ax in axes.ravel()[len(SIG_KEYS):]:
        ax.axis('off')
    axes.ravel()[len(SIG_KEYS)].legend(*axes.ravel()[0].get_legend_handles_labels(),
                                       fontsize=5, frameon=False, loc='center')
    for ax in axes[-1]:
        ax.set_xlabel('rt_threshold')
    for ax in axes[:, 0]:
        ax.set_ylabel("Cohen's d (+ = human)")
    fig.suptitle(f'{variant}: signature effect size vs rt_threshold', fontsize=7)
    fig.tight_layout()
    return save(fig, f'{out_dir}/fig_2_signatures.pdf')


def fig_matched(counts, out_dir):
    """
    How many signatures the model matches, as a function of the analysis cut.

    If the count peaks at the value the project already uses, that is worth saying out
    loud: a post-hoc parameter tuned — even unintentionally — to maximise a scorecard is
    a researcher degree of freedom, and the honest report is the whole curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=FigSize.row(2))

    for arm in ARM_ORDER:
        sub = counts[(counts['arm'] == arm) & (counts['variant'] == 'noise09')]
        if not len(sub):
            continue
        sub = sub.sort_values('threshold')
        axes[0].plot(sub['threshold'], sub['matched'], marker='o', markersize=3,
                     color=ARM_COLORS[arm], linewidth=1.0, label=ARM_LABELS[arm])
    axes[0].set_title('noise09, by arm', fontsize=6)
    axes[0].legend(fontsize=4.5, frameon=False, loc='lower center')

    base = counts[counts['arm'] == BASELINE_ARM]
    for (variant, noise), color in zip(NOISE_LADDER, NOISE_COLORS):
        sub = base[base['variant'] == variant].sort_values('threshold')
        axes[1].plot(sub['threshold'], sub['matched'], marker='o', markersize=3,
                     color=color, linewidth=1.0, label=f'{noise}')
    axes[1].set_title(f'{BASELINE_ARM}, by stimulus noise', fontsize=6)
    axes[1].legend(title='arrow noise', fontsize=4.5, title_fontsize=4.5,
                   frameon=False, loc='lower center')

    for ax in axes:
        _mark_default(ax)
        ax.set_xlabel('rt_threshold')
        ax.set_ylabel(f'signatures matched (of {len(SIG_KEYS)})')
        ax.set_ylim(0, len(SIG_KEYS) + 0.5)
    fig.tight_layout()
    return save(fig, f'{out_dir}/fig_3_matched.pdf')


def fig_effects_vs_threshold(df, out_dir, variant='noise09'):
    """
    Every RT and accuracy signature against the threshold, each on its own scale.

    Two rows because the two families answer H1 differently, and the difference is the
    point: over 0.2–0.8 the RT effects move by more than their own size while the accuracy
    effects barely stir. Columns are matched by construct where a counterpart exists —
    congruency, distance within incongruent, post-error — so reading down a column compares
    the same manipulation measured two ways.

    Effects are plotted in their own units rather than standardised: the whole point is
    that the accuracy row's y-range is narrow, and normalising would hide it.
    """
    d = df[(df['mode'] == 'abs') & (df['variant'] == variant)
           & (df['arm'] == BASELINE_ARM)]
    families = [(RT_SIGNATURES, 'RT effect (timesteps)', COL['incong']),
                (ACC_SIGNATURES, 'Accuracy effect (proportion)', COL['cong'])]

    fig, axes = plt.subplots(2, len(RT_SIGNATURES),
                             figsize=FigSize.grid(2, len(RT_SIGNATURES),
                                                  panel=FigSize.small))
    for (keys, ylabel, color), axrow in zip(families, axes):
        for ax, key in zip(axrow, keys):
            x, mu, sem = _seed_stats(d, key, 'threshold')
            _line(ax, x, mu, sem, color, None)
            ax.axhline(0, color='k', linewidth=0.6, alpha=0.6)
            _mark_default(ax)
            ax.set_title(SIG_LABEL[key], fontsize=5)
        axrow[0].set_ylabel(ylabel, fontsize=5)
    for ax in axes[-1]:
        ax.set_xlabel('rt_threshold')

    fig.suptitle(f'{BASELINE_ARM}, {variant}: every signature against the analysis '
                 'threshold — RT above, accuracy below', fontsize=6)
    fig.tight_layout()
    return save(fig, f'{out_dir}/fig_4_effects_vs_threshold.pdf')


def fig_amplitude(df, verdicts, out_dir):
    """
    H5: is one absolute threshold the same criterion in all four arms?

    Left: where each arm's decision variable actually lives, summarised by the fraction
    of trials whose max|output| never reaches 0.5. If the arms differ there, a fixed
    absolute cut scores them at different points of their own distributions and every
    cross-arm comparison inherits that difference.

    Right: the same signatures scored at a per-session *quantile* threshold instead, so
    every session is held at one undecided rate. Verdicts that survive that swap are
    about the model; verdicts that do not are about the criterion.
    """
    fig, axes = plt.subplots(1, 2, figsize=FigSize.row(2))

    d = df[(df['mode'] == 'abs') & np.isclose(df['threshold'], RT_THRESHOLD)]
    for i, arm in enumerate(ARM_ORDER):
        sub = d[d['arm'] == arm]
        if not len(sub):
            continue
        for (variant, noise), color in zip(NOISE_LADDER, NOISE_COLORS):
            s = sub[sub['variant'] == variant]['amp_frac_below_0.5'].to_numpy()
            if not len(s):
                continue
            axes[0].errorbar(i + 0.16 * (list(dict(NOISE_LADDER).values()).index(noise) - 2),
                             s.mean(), yerr=s.std(ddof=1) / np.sqrt(len(s)),
                             marker='o', markersize=2.5, color=color, capsize=1.5,
                             linewidth=0.8)
    axes[0].set_xticks(range(len(ARM_ORDER)))
    axes[0].set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=4.5, rotation=20)
    axes[0].set_ylabel('P(max|output| < 0.5)')
    axes[0].set_title('Where 0.5 sits in each arm', fontsize=6)

    q = verdicts[(verdicts['mode'] == 'quant') & (verdicts['variant'] == 'noise09')]
    for arm in ARM_ORDER:
        sub = q[(q['arm'] == arm) & (q['signature'] == 'peri')].sort_values('threshold')
        if not len(sub):
            continue
        _line(axes[1], sub['threshold'].to_numpy(dtype=float), sub['d'].to_numpy(),
              1.96 * sub['d_sem'].to_numpy(), ARM_COLORS[arm], ARM_LABELS[arm])
    n = float(q['n_seeds'].max()) if len(q) else float(SEEDS)
    axes[1].axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color=COL['null'],
                    alpha=0.15, linewidth=0)
    axes[1].axhline(0, color='k', linewidth=0.6, alpha=0.6)
    axes[1].set_xlabel('undecided rate held per session')
    axes[1].set_ylabel("PERI, Cohen's d")
    axes[1].set_title('PERI at a matched undecided rate', fontsize=6)
    axes[1].legend(fontsize=4.5, frameon=False)
    fig.tight_layout()
    return save(fig, f'{out_dir}/fig_5_amplitude.pdf')


# ── Tables printed to the console ─────────────────────────────────────────────

def pilot_table(df):
    """The brief's pilot table: baseline arm, noise09, the five thresholds it used."""
    d = df[(df['arm'] == BASELINE_ARM) & (df['variant'] == 'noise09')
           & (df['mode'] == 'abs')]
    v = score(d)
    counts = matched_counts(v)
    rows = []
    for thr in sorted(d['threshold'].unique()):
        s = d[np.isclose(d['threshold'], thr)]
        c = counts[np.isclose(counts['threshold'], thr)]
        rows.append({
            'thr': thr,
            'undecided': round(float(s['undecided_frac'].mean()), 3),
            'matched': int(c['matched'].iloc[0]), 'opp': int(c['opp'].iloc[0]),
            'cong_acc': round(float(s['cong_effect_acc'].mean()), 4),
            'cong_rt': round(float(s['cong_effect_rt'].mean()), 4),
            'pes_BI': round(float(s['pes_BI'].mean()), 4),
            'pia_BI': round(float(s['pia_BI'].mean()), 4),
            'peri': round(float(s['peri'].mean()), 4),
        })
    return pd.DataFrame(rows)


def verdict_matrix(verdicts, arm=BASELINE_ARM, variant='noise09', mode='abs'):
    """Signature x threshold verdicts for one cell, as a readable table."""
    v = verdicts[(verdicts['mode'] == mode) & (verdicts['arm'] == arm)
                 & (verdicts['variant'] == variant)]
    return v.pivot_table(index='signature', columns='threshold', values='verdict',
                         aggfunc='first').reindex(SIG_KEYS)


# ── Entry points ──────────────────────────────────────────────────────────────

def run_pilot():
    """Baseline arm, noise09, the brief's five thresholds — the correctness check."""
    os.makedirs(OUT_DIR, exist_ok=True)
    df = sweep(arms=(BASELINE_ARM,), ladder=[('noise09', 0.9)],
               thresholds=PILOT_THRESHOLDS, quantiles=())
    print('\nPilot — factorial_nojit_pc52, noise09, 20 seeds\n')
    print(pilot_table(df).to_string(index=False))
    print('\nVerdicts (signature x threshold):\n')
    print(verdict_matrix(score(df)).to_string())
    return df


def run_sweep():
    """The full grid, cached to CSV so `report` can be re-run without re-reading pickles."""
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Sweeping {len(ARM_ORDER)} arms x {len(NOISE_LADDER)} noise levels x {SEEDS} '
          f'seeds x {len(THRESHOLDS)} absolute + {len(QUANTILES)} quantile thresholds')
    df = sweep()
    df.to_csv(EFFECTS_CSV, index=False)
    print(f'\nWrote {EFFECTS_CSV}  ({len(df)} rows)')
    return df


def run_report(df=None):
    """Score the cached sweep, write the tables and the figures, print the summary."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if df is None:
        df = pd.read_csv(EFFECTS_CSV)
    v = score(df)
    v_dec = score(df, keys=list(RT_SIGNATURES), suffix='__dec')
    v_dec['signature'] = v_dec['signature'] + '__dec'
    pd.concat([v, v_dec]).to_csv(VERDICTS_CSV, index=False)

    flips = flip_table(v)
    flips.to_csv(FLIPS_CSV, index=False)
    counts = matched_counts(v)
    counts.to_csv(f'{OUT_DIR}/matched_counts.csv', index=False)

    print(f'\nWrote {VERDICTS_CSV} and {FLIPS_CSV}')
    print(f'\nVerdict flips over {THRESHOLDS[0]}–{THRESHOLDS[-1]}: '
          f'{len(flips)} of {len(v.groupby(["arm", "variant", "signature"]))} '
          f'(arm, noise, signature) cells; {int(flips["reverses"].sum()) if len(flips) else 0} '
          f'reverse sign (PASS <-> OPP).\n')
    if len(flips):
        print(flips[['arm', 'variant', 'signature', 'at_default', 'verdicts']]
              .to_string(index=False))

    print('\nSignatures matched vs threshold (rows = arm x noise):\n')
    print(counts.pivot_table(index=['arm', 'variant'], columns='threshold',
                             values='matched').to_string())

    fig_undecided(df, OUT_DIR)
    fig_signatures(v, OUT_DIR)
    fig_matched(counts, OUT_DIR)
    fig_effects_vs_threshold(df, OUT_DIR)
    fig_amplitude(df, v, OUT_DIR)
    return df, v, flips, counts


# ── The project's own figures, re-rendered at each threshold ──────────────────
#
# The figures above put the threshold on an axis. These do the opposite: they redraw the
# figures the project already reads — the scorecard, the noise ladder, the regression
# forest — once per threshold, into one folder, with the threshold in every filename. Sort
# the folder by name and you can flip through the series.

FIG_DIR = f'{OUT_DIR}/by_threshold'


def _effects_by_threshold(arm, ladder=NOISE_LADDER, thresholds=THRESHOLDS, seeds=None):
    """
    `{threshold: {variant: [per-seed effects]}}`, unpickling each result exactly once.

    The whole cache is per-seed effect dicts, a few kB each, so all 7 thresholds x 5
    variants x 20 seeds fit in memory comfortably and no pickle is read twice.
    """
    seeds = list(range(SEEDS)) if seeds is None else list(seeds)
    cache = {t: {} for t in thresholds}
    with flanker_sweep.use_run(f'factorial_{arm}'):
        for variant, _ in ladder:
            per_thr = {t: [] for t in thresholds}
            for seed in seeds:
                res = flanker_sweep.load_result(seed, variant)
                if res is None:
                    continue
                for t in thresholds:
                    trials = extract_trials(res['train_logger'], res['config'],
                                            rt_threshold=t)
                    per_thr[t].append(session_effects(trials))
            for t in thresholds:
                cache[t][variant] = per_thr[t]
            print(f'  {arm:12s} {variant:8s} {len(per_thr[thresholds[0]]):2d} seeds')
    return cache


@contextlib.contextmanager
def _figures_at(threshold, effects_by_variant):
    """
    Make `flanker_sweep_figures` draw at one threshold instead of the global default.

    Two names are rebound for the duration. `collect_effects` is what `fig_noise_series`
    calls internally, and it takes the threshold from `flanker_sweep_config.RT_THRESHOLD`
    as a default bound at definition time — so serving it from the cache is the only way
    to steer it without editing the config, which the brief rules out. `save` is wrapped
    only to stamp the threshold in the corner, so a figure pulled out of the series still
    says which one it is.
    """
    import flanker_sweep_figures as figs
    original_save, original_collect = figs.save, figs.collect_effects

    def stamped_save(fig, path, note=None):
        fig.text(0.995, 0.005, f'rt_threshold = {threshold:.2f}', ha='right', va='bottom',
                 fontsize=5, color=COL['neutral'])
        return original_save(fig, path, note)

    figs.save = stamped_save
    figs.collect_effects = lambda variant, **kw: effects_by_variant.get(variant, [])
    try:
        yield
    finally:
        figs.save, figs.collect_effects = original_save, original_collect


def run_figures(arm=BASELINE_ARM, thresholds=THRESHOLDS, regression_variant='noise09'):
    """
    Redraw the scorecard, the noise ladder and the regression forest at every threshold.

    Writes to `exports/flanker_random/rt_threshold/by_threshold/`, never into the run
    directories the existing figures live in.

    The regression is run for one variant only. Fitting is ~0.9 s per session per spec, so
    the full ladder would be ~20 minutes against ~4 for one level; pass
    `regression_variant=None` to skip it or another variant name to move it.
    """
    from flanker_sweep_figures import fig_noise_series, fig_scorecard

    os.makedirs(FIG_DIR, exist_ok=True)
    print(f'Building the effect cache for {arm}:')
    cache = _effects_by_threshold(arm, thresholds=thresholds)

    written = []
    for thr in thresholds:
        with _figures_at(thr, cache[thr]):
            for variant, _ in NOISE_LADDER:
                effects = cache[thr][variant]
                if not effects:
                    continue
                path = fig_scorecard(effects, FIG_DIR,
                                     f'{arm} · {variant} · rt_threshold {thr:.2f}')
                written.append(_rename(path, f'scorecard_{arm}_{variant}_thr{thr:.2f}.pdf'))
            path = fig_noise_series(FIG_DIR)
            if path:
                written.append(_rename(path, f'noise_series_{arm}_thr{thr:.2f}.pdf'))

    if regression_variant:
        written += run_regression_series(arm, regression_variant, thresholds)
    print(f'\n{len(written)} figures in {FIG_DIR}')
    return written


def _rename(path, name):
    """Move a figure the project's `save` just wrote to its threshold-tagged filename."""
    final = os.path.join(os.path.dirname(path), name)
    os.replace(path, final)
    return final


def run_regression_series(arm, variant, thresholds=THRESHOLDS, specs=('M2', 'M3')):
    """
    The trial-history regression forest at each threshold, for one variant.

    `flanker_regression.group_report` extracts trials at the default threshold and cannot
    be told otherwise, so this reproduces its two steps — extract, then `fit_sessions` per
    spec — with the threshold passed through, and hands the result to the same figure.
    The RT signatures here are coefficients on log RT, so they inherit the threshold's
    effect on RT directly; PERI is the `incong:prev_error` term.
    """
    import matplotlib.pyplot as plt
    import flanker_regression as reg

    os.makedirs(FIG_DIR, exist_ok=True)
    with flanker_sweep.use_run(f'factorial_{arm}'):
        results = [r for r in (flanker_sweep.load_result(s, variant)
                               for s in range(SEEDS)) if r is not None]
    if not results:
        print(f'No sessions for {arm}/{variant} — skipping the regression series.')
        return []

    written, tidy = [], []
    for thr in thresholds:
        trials = [extract_trials(r['train_logger'], r['config'], rt_threshold=thr)
                  for r in results]
        summaries = {}
        for spec in specs:
            _, summary = reg.fit_sessions(trials, spec=spec)
            summaries[spec] = summary.set_index(['dv', 'term'])
            tidy.append(summary.assign(threshold=thr, spec=spec, arm=arm, variant=variant))
        path = f'{FIG_DIR}/regression_{arm}_{variant}_thr{thr:.2f}.pdf'
        fig = reg.fig_group_coefficients(
            summaries, path=path,
            title=f'Trial-history regression — {arm} · {variant} · '
                  f'rt_threshold {thr:.2f}')
        plt.close(fig)
        written.append(path)

    # The coefficients as a table too — seven forest plots are hard to read a trend off,
    # and this is the regression's independent verdict on the same question the scorecard
    # answers with cell contrasts.
    coef_path = f'{OUT_DIR}/regression_coefficients.csv'
    coefs = pd.concat(tidy, ignore_index=True)
    coefs.to_csv(coef_path, index=False)
    print(f'\nWrote {coef_path}')
    key_terms = ['incong', 'incong:far', 'prev_error', 'incong:prev_error']
    for dv in ('acc', 'rt'):
        sub = coefs[(coefs['spec'] == specs[0]) & (coefs['dv'] == dv)
                    & coefs['term'].isin(key_terms)]
        if not len(sub):
            continue
        print(f'\n  t across sessions, {dv}, spec {specs[0]} '
              f'(|t| > 1.96 is significant):')
        print(sub.pivot_table(index='term', columns='threshold', values='t')
              .reindex(key_terms).to_string(float_format=lambda x: f'{x: .2f}'))
    return written


def run_hypotheses(df=None):
    """
    The tables `docs/rt_threshold_findings.md` quotes, so its numbers stay re-derivable.

    Six questions, in the order the brief asks them: is accuracy more robust than RT (H1),
    is the RT dependence the undecided pile-up (H2), does PERI's verdict move and does the
    cross-arm PERI conclusion move with it (H3), is 0.5 a maximum of the scorecard (H4), is
    one absolute threshold the same criterion in every arm (H5), and where along the noise
    ladder does the choice matter (H6).
    """
    from scipy import stats

    if df is None:
        df = pd.read_csv(EFFECTS_CSV)
    v = pd.read_csv(VERDICTS_CSV)
    A = df[df['mode'] == 'abs']
    counts = pd.read_csv(f'{OUT_DIR}/matched_counts.csv')

    def head(n, text):
        print(f'\n{"=" * 78}\n{n} — {text}\n{"=" * 78}')

    head('H1', 'accuracy robust, RT not: |v(0.8) - v(0.2)| / |v(0.5)|, median over cells')
    rows = []
    for key in SIG_KEYS:
        at = {t: A[np.isclose(A['threshold'], t)].groupby(['arm', 'variant'])[key].mean()
              for t in (0.2, 0.5, 0.8)}
        rel = (at[0.8] - at[0.2]).abs() / at[0.5].abs().replace(0, np.nan)
        rows.append(dict(signature=key,
                         kind='RT' if ('_rt' in key or key in ('pes_BI', 'peri')) else 'acc',
                         v_0p2=at[0.2].mean(), v_0p5=at[0.5].mean(), v_0p8=at[0.8].mean(),
                         rel_swing=float(rel.median())))
    h1 = pd.DataFrame(rows).sort_values('rel_swing')
    print(h1.to_string(index=False, float_format=lambda x: f'{x: .4f}'))
    print('\nmedian relative swing by kind:')
    print(h1.groupby('kind')['rel_swing'].median().to_string())

    head('H2', 'how much of the RT threshold-dependence is the undecided pile-up?')
    base = A[A['arm'] == BASELINE_ARM]
    for key in RT_SIGNATURES:
        a, d = (base.groupby('threshold')[c].mean() for c in (key, f'{key}__dec'))
        sw_a, sw_d = a.loc[0.8] - a.loc[0.2], d.loc[0.8] - d.loc[0.2]
        print(f'{key:24s} all {a.loc[0.2]: .3f}->{a.loc[0.8]: .3f}   '
              f'decided-only {d.loc[0.2]: .3f}->{d.loc[0.8]: .3f}   '
              f'pile-up accounts for {1 - abs(sw_d) / abs(sw_a):.0%}')
    print('\nDecided-only verdicts, baseline arm, noise09:')
    vd = v[(v['mode'] == 'abs') & v['signature'].str.endswith('__dec')
           & (v['arm'] == BASELINE_ARM) & (v['variant'] == 'noise09')]
    print(vd.pivot_table(index='signature', columns='threshold', values='verdict',
                         aggfunc='first').to_string())

    head('H3', 'PERI: the verdict, and the cross-arm contrast the conclusion rests on')
    p = v[(v['mode'] == 'abs') & (v['signature'] == 'peri')]
    print(p.pivot_table(index=['variant', 'arm'], columns='threshold', values='verdict',
                        aggfunc='first').to_string())
    print('\njitter\'s effect on PERI (nojit_pc52 - jit_pc52), Welch t across seeds:')
    rows = []
    for variant, _ in NOISE_LADDER:
        for thr in THRESHOLDS:
            sel = (A['variant'] == variant) & np.isclose(A['threshold'], thr)
            a = A[sel & (A['arm'] == 'nojit_pc52')]['peri'].to_numpy()
            b = A[sel & (A['arm'] == 'jit_pc52')]['peri'].to_numpy()
            _, pv = stats.ttest_ind(a, b, equal_var=False)
            rows.append(dict(variant=variant, thr=thr, diff=a.mean() - b.mean(),
                             sig='yes' if pv < 0.05 else 'no'))
    h3 = pd.DataFrame(rows)
    print(h3.pivot_table(index='variant', columns='thr', values='diff').to_string(
        float_format=lambda x: f'{x: .3f}'))
    print(h3.pivot_table(index='variant', columns='thr', values='sig',
                         aggfunc='first').to_string())

    head('H4', 'is 0.5 at a maximum of the scorecard?  (summed over all 20 arm x noise cells)')
    print(counts.groupby('threshold')[['matched', 'opp']].sum().to_string())

    head('H5', 'is a fixed absolute threshold the same criterion in every arm?')
    at5 = A[np.isclose(A['threshold'], RT_THRESHOLD)]
    print('undecided fraction at an absolute 0.5, and the threshold each arm would need')
    print('for a 10% undecided rate:')
    lhs = at5.groupby('arm')['undecided_frac'].mean().rename('undecided_at_0.5')
    rhs = (df[(df['mode'] == 'quant') & np.isclose(df['threshold'], 0.10)]
           .groupby('arm')['thr_abs'].mean().rename('thr_for_10pct'))
    print(pd.concat([lhs, rhs], axis=1).to_string(float_format=lambda x: f'{x: .3f}'))
    print('\nPERI rescored at a per-session quantile threshold (matched undecided rate):')
    pq = v[(v['mode'] == 'quant') & (v['signature'] == 'peri')]
    print(pq.pivot_table(index=['variant', 'arm'], columns='threshold', values='verdict',
                         aggfunc='first').to_string())

    head('H6', 'where along the noise ladder does the choice matter?')
    flips = pd.read_csv(FLIPS_CSV)
    order = [x for x, _ in NOISE_LADDER]
    print('flips per noise level (of 44 = 4 arms x 11 signatures):')
    print(flips.groupby('variant').size().reindex(order).fillna(0).astype(int).to_string())
    print('\nflips per signature (of 20 = 4 arms x 5 noise):')
    print(flips.groupby('signature').size().sort_values(ascending=False).to_string())

    head('CLAIMS', 'the sentences in flanker_sweep_config.py, checked across the range')
    piv = counts.pivot_table(index='variant', columns=['threshold', 'arm'], values='matched')
    diff = (piv.xs('nojit_pc52', level='arm', axis=1)
            - piv.xs('jit_pc52', level='arm', axis=1)).reindex(order)
    print('matched(baseline) - matched(jitter), per noise x threshold:')
    print(diff.to_string())
    print(f'\n"jitter never matches MORE than the baseline": violated in '
          f'{int((diff < 0).sum().sum())} of {diff.size} cells; '
          f'{int((diff[RT_THRESHOLD] < 0).sum())} of {len(diff)} at the default 0.5.')
    for key in ('dist_effect_rt_incong', 'pes_BI'):
        w = v[(v['mode'] == 'abs') & (v['variant'] == 'noise09') & (v['signature'] == key)]
        print(f'\n{key}, noise09:')
        print(w.pivot_table(index='arm', columns='threshold', values='verdict',
                            aggfunc='first').to_string())
    return df


def main(mode='all', arm=BASELINE_ARM):
    if mode == 'pilot':
        return run_pilot()
    if mode == 'figures':
        return run_figures(arm)
    df = run_sweep() if mode in ('sweep', 'all') else None
    if mode in ('report', 'all'):
        df = run_report(df)[0]
    if mode in ('hypotheses', 'all'):
        return run_hypotheses(df)
    return df


if __name__ == '__main__':
    # `figures` takes an optional arm: `... figures jit_pc52`.
    main(*(sys.argv[1:3] or ['all']))
