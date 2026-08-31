"""
flanker_sweep_analysis.py — across-subject statistics for the flanker sweep.

Each seed is a synthetic subject, so every effect is computed *within* a seed and
then tested across seeds with a one-sample t-test on the per-seed values. This is
the summary-statistics design used on the human data, and it is the error bar that
matters: a difference that is huge across trials within one model but inconsistent
across seeds is a property of that model instance, not of the mechanism.

Run:
    python flanker_sweep_analysis.py

Requires flanker_sweep.py to have been run first.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import plot_style
plot_style.set_plot_style()
from plot_style import FigSize

from flanker_analyses import extract_trials
from flanker_metrics import session_effects
from flanker_sweep import load_condition, result_path
from flanker_sweep_config import RT_THRESHOLD, SEEDS, VARIANTS

import os


# ── Per-session effects ───────────────────────────────────────────────────────
# The measures themselves live in flanker_metrics.py, grouped by question (behaviour,
# RT by outcome, history, post-error, control). This file is about what to do with them
# across seeds: t-tests, paired contrasts, and the printed tables.


def collect_by_seed(variant, rt_threshold=RT_THRESHOLD):
    """Load every seed of one variant. Returns {seed: effects dict}."""
    out = {}
    for res in load_condition(variant):
        trials = extract_trials(res['train_logger'], res['config'], rt_threshold=rt_threshold)
        out[res['seed']] = session_effects(trials)
    return out


def collect(variant, rt_threshold=RT_THRESHOLD):
    """Load every seed of one variant and stack per-seed effects."""
    by_seed = collect_by_seed(variant, rt_threshold)
    if not by_seed:
        return {}
    rows = [by_seed[s] for s in sorted(by_seed)]
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


# ── Across-seed statistics ────────────────────────────────────────────────────

def summarize(values, popmean=0.0):
    """Mean, SEM, t and p for a per-seed effect against popmean."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return dict(mean=np.nan, sem=np.nan, t=np.nan, p=np.nan, n=n)
    mean = v.mean()
    sem  = v.std(ddof=1) / np.sqrt(n)
    try:
        from scipy.stats import ttest_1samp
        t, p = ttest_1samp(v, popmean)
    except Exception:
        t = (mean - popmean) / sem if sem > 0 else np.nan
        p = np.nan
    return dict(mean=mean, sem=sem, t=float(t), p=float(p), n=n)


def report(effects, keys, title, popmean=0.0):
    """Print a labelled block of across-seed tests."""
    print(f'\n{title}')
    print(f'  {"effect":<32} {"mean":>9} {"SEM":>8} {"t":>7} {"p":>9}   n')
    for key, label in keys:
        s = summarize(effects[key], popmean) if key in effects else dict(
            mean=np.nan, sem=np.nan, t=np.nan, p=np.nan, n=0)
        star = ''
        if not np.isnan(s['p']):
            star = '***' if s['p'] < 0.001 else '**' if s['p'] < 0.01 else '*' if s['p'] < 0.05 else ''
        print(f'  {label:<32} {s["mean"]:9.4f} {s["sem"]:8.4f} {s["t"]:7.2f} {s["p"]:9.5f} {star:>4} {s["n"]}')


def paired(effects, key_a, key_b, label):
    """Across-seed paired test on the difference between two per-seed effects."""
    if key_a not in effects or key_b not in effects:
        return
    s = summarize(effects[key_a] - effects[key_b])
    star = ''
    if not np.isnan(s['p']):
        star = '***' if s['p'] < 0.001 else '**' if s['p'] < 0.01 else '*' if s['p'] < 0.05 else ''
    print(f'  {label:<32} {s["mean"]:9.4f} {s["sem"]:8.4f} {s["t"]:7.2f} {s["p"]:9.5f} {star:>4} {s["n"]}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(variant=None):
    variant = variant or next(iter(VARIANTS))
    print('=' * 84)
    print(f'ACROSS-SUBJECT SUMMARY  (each seed = one subject, variant={variant})')
    print('=' * 84)

    eff = collect(variant)
    if not eff:
        print('No results found — run flanker_sweep.py first.')
        return

    report(eff, [('acc_overall', 'overall accuracy'),
                 ('undecided_frac', 'undecided fraction'),
                 ('focus_all', 'Z focus (control state)'),
                 # The gate's own behaviour. off_centre is the fraction of trials whose
                 # inherited gate peaks away from the target slot — the mechanism behind a
                 # spurious distance effect on congruent trials. gate_peak is comparable to
                 # the Stage-1 oracle peak (0.405 at softmax_temp 1, five slots): well above
                 # it means Stage 2 is running the weights at a sharpness Stage 1 never
                 # trained, unless config.oracle_gate_jitter is set.
                 ('gate_off_centre', 'gate peaks off target (fraction)'),
                 ('gate_peak', 'gate peak (median)')],
           'Session quality')

    report(eff, [('acc_near_cong',   'accuracy, near-congruent'),
                 ('acc_far_cong',    'accuracy, far-congruent'),
                 ('acc_near_incong', 'accuracy, near-incongruent'),
                 ('acc_far_incong',  'accuracy, far-incongruent'),
                 ('rt_near_cong',    'RT, near-congruent'),
                 ('rt_far_cong',     'RT, far-congruent'),
                 ('rt_near_incong',  'RT, near-incongruent'),
                 ('rt_far_incong',   'RT, far-incongruent')],
           'RESULT 1a — the four condition cells')

    report(eff, [('cong_effect_acc', 'congruency effect, accuracy'),
                 ('cong_effect_rt',  'congruency effect, RT'),
                 ('cong_effect_acc_near', '  ... accuracy, near'),
                 ('cong_effect_acc_far',  '  ... accuracy, far'),
                 ('interaction_acc', 'near - far  (accuracy)'),
                 ('cong_effect_rt_near', '  ... RT, near'),
                 ('cong_effect_rt_far',  '  ... RT, far'),
                 ('interaction_rt',  'near - far  (RT)')],
           'RESULT 1b — congruency effect, and its interaction with distance')

    # The directional predictions: near flankers should help on congruent trials and
    # hurt on incongruent ones, more than far flankers do.
    report(eff, [('dist_effect_acc_cong',   'accuracy, congruent   (expect +)'),
                 ('dist_effect_acc_incong', 'accuracy, incongruent (expect -)'),
                 ('dist_effect_rt_cong',    'RT, congruent   (expect -)'),
                 ('dist_effect_rt_incong',  'RT, incongruent (expect +)')],
           'RESULT 1c — simple effects of distance (near minus far), within congruency')

    report(eff, [('acc_CC_to_I', 'accuracy CC->I'),
                 ('acc_CI_to_I', 'accuracy CI->I'),
                 ('acc_IC_to_I', 'accuracy IC->I'),
                 ('acc_II_to_I', 'accuracy II->I'),
                 ('lag1_contrast_acc', 'lag-1 contrast (accuracy)'),
                 ('lag2_contrast_acc', 'lag-2 contrast (accuracy)'),
                 ('lag1_contrast_rt',  'lag-1 contrast (RT)'),
                 ('lag2_contrast_rt',  'lag-2 contrast (RT)')],
           'RESULT 3 — sequential congruency, post-correct')
    print('  --- which lag dominates? ---')
    paired(eff, 'lag1_contrast_acc', 'lag2_contrast_acc', 'lag1 - lag2 (accuracy)')
    paired(eff, 'lag1_contrast_rt',  'lag2_contrast_rt',  'lag1 - lag2 (RT)')

    report(eff, [('sce_rt_all',    'RT SCE, all trials'),
                 ('sce_rt_switch', 'RT SCE, response switched'),
                 ('sce_rt_repeat', 'RT SCE, response repeated'),
                 ('sce_acc_all',    'acc SCE, all trials'),
                 ('sce_acc_switch', 'acc SCE, response switched'),
                 ('sce_acc_repeat', 'acc SCE, response repeated')],
           'RESULT 3b — does the Gratton effect survive repetition control?')
    print('  --- repetition dependence ---')
    paired(eff, 'sce_rt_repeat', 'sce_rt_switch', 'RT SCE: repeat - switch')
    paired(eff, 'sce_acc_repeat', 'sce_acc_switch', 'acc SCE: repeat - switch')

    report(eff, [('pes_BI', 'post-error slowing, B incong'),
                 ('pes_BC', 'post-error slowing, B cong'),
                 ('pia_BI', 'post-error accuracy, B incong'),
                 ('pia_BC', 'post-error accuracy, B cong'),
                 ('peri',   'PERI (congruency effect drop)'),
                 ('focus_in_diff_BI', 'focus_in: err - corr, B incong')],
           'RESULT 4 — post-error effects (incongruent trial A)')

    report(eff, [('dfocus_near_err',  'delta focus, near-incong error'),
                 ('dfocus_far_err',   'delta focus, far-incong error'),
                 ('dfocus_near_corr', 'delta focus, near-incong correct'),
                 ('dfocus_far_corr',  'delta focus, far-incong correct'),
                 ('dfocus_near_minus_far_err', 'near - far (errors)')],
           'RESULT 5 — what drives the control update')


if __name__ == '__main__':
    # `python flanker_sweep_analysis.py noise04` for one variant; no argument uses the
    # first in flanker_sweep_config.VARIANTS.
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
