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

from flanker_analyses import extract_trials, lagged_factors
from flanker_sweep import load_condition, result_path
from flanker_sweep_config import P_CONGRUENT_LEVELS, RT_THRESHOLD, SEEDS

import os


# ── Per-session effect computation ────────────────────────────────────────────

def _mean(vals, mask):
    """NaN-safe mean over a boolean mask; NaN if the cell is empty."""
    v = vals[mask]
    v = v[~np.isnan(v)]
    return v.mean() if len(v) else np.nan


def session_effects(trials):
    """Compute every scalar effect of interest for one session. Returns dict."""
    f = lagged_factors(trials, n_back=2)

    acc = trials['correct_at_decision'].astype(float)
    rt  = trials['rt_interp']
    foc_in, dfoc = trials['focus_in'], trials['delta_focus']

    cong,   incong  = f['cong'] == 1,      f['cong'] == 0
    near,   far     = f['near'] == 1,      f['near'] == 0
    valid           = f['valid']
    pc, perr        = f['correct_1'] == 1, f['correct_1'] == 0
    p2c             = f['correct_2'] == 1
    p_cong, p_incong   = f['cong_1'] == 1, f['cong_1'] == 0
    p2_cong, p2_incong = f['cong_2'] == 1, f['cong_2'] == 0
    p_near, p_far   = f['near_1'] == 1,    f['near_1'] == 0
    rep, sw         = f['resp_rep'] == 1,  f['resp_rep'] == 0
    corr_dec        = trials['correct_at_decision']

    e = {}
    e['censored_frac'] = float(np.isnan(rt).mean())
    e['acc_overall']   = acc.mean()

    # Control state actually in force during each trial: the Z focus carried in from
    # the previous trial. This is the level the list composition moves, and it gates
    # how much the flankers get into the decision at all.
    e['focus_all']    = float(np.nanmean(foc_in))
    e['focus_cong']   = _mean(foc_in, cong)
    e['focus_incong'] = _mean(foc_in, incong)

    # 1. Congruency x distance
    e['cong_effect_acc'] = _mean(acc, cong) - _mean(acc, incong)
    e['cong_effect_rt']  = _mean(rt, incong) - _mean(rt, cong)
    for dn, dm in [('near', near), ('far', far)]:
        e[f'cong_effect_acc_{dn}'] = _mean(acc, cong & dm) - _mean(acc, incong & dm)
        e[f'cong_effect_rt_{dn}']  = _mean(rt, incong & dm) - _mean(rt, cong & dm)
    e['interaction_acc'] = e['cong_effect_acc_near'] - e['cong_effect_acc_far']
    e['interaction_rt']  = e['cong_effect_rt_near']  - e['cong_effect_rt_far']

    # Cell means, so the four conditions can be read directly and not only as contrasts.
    for cn, cm in [('cong', cong), ('incong', incong)]:
        for dn, dm in [('near', near), ('far', far)]:
            e[f'acc_{dn}_{cn}'] = _mean(acc, cm & dm)
            e[f'rt_{dn}_{cn}']  = _mean(rt,  cm & dm)

    # Simple effects of distance *within* each congruency. This is the prediction that
    # actually has a direction: near flankers should help when they agree with the
    # target (+ on accuracy) and hurt when they conflict (- on accuracy). The
    # interaction is the sum of the two and hides which half, if either, is present.
    e['dist_effect_acc_cong']   = e['acc_near_cong']   - e['acc_far_cong']
    e['dist_effect_acc_incong'] = e['acc_near_incong'] - e['acc_far_incong']
    e['dist_effect_rt_cong']    = e['rt_near_cong']    - e['rt_far_cong']
    e['dist_effect_rt_incong']  = e['rt_near_incong']  - e['rt_far_incong']

    # 2. Sequential congruency, post-correct (both prior trials correct)
    ok = valid & pc & p2c
    for lbl, c2m, c1m in [('CC', p2_cong, p_cong), ('CI', p2_cong, p_incong),
                          ('IC', p2_incong, p_cong), ('II', p2_incong, p_incong)]:
        e[f'acc_{lbl}_to_I'] = _mean(acc, ok & c2m & c1m & incong)
        e[f'rt_{lbl}_to_I']  = _mean(rt,  ok & c2m & c1m & incong)
        e[f'acc_{lbl}_to_C'] = _mean(acc, ok & c2m & c1m & cong)
        e[f'rt_{lbl}_to_C']  = _mean(rt,  ok & c2m & c1m & cong)

    # Which lag drives the history effect? Both contrasts on current-incongruent trials.
    e['lag1_contrast_acc'] = _mean(acc, ok & p_incong & incong)  - _mean(acc, ok & p_cong & incong)
    e['lag2_contrast_acc'] = _mean(acc, ok & p2_incong & incong) - _mean(acc, ok & p2_cong & incong)
    e['lag1_contrast_rt']  = _mean(rt,  ok & p_cong & incong)    - _mean(rt,  ok & p_incong & incong)
    e['lag2_contrast_rt']  = _mean(rt,  ok & p2_cong & incong)   - _mean(rt,  ok & p2_incong & incong)

    # 3. Gratton interaction (lag 1), with and without the response-repetition control
    for rname, rmask in [('all', np.ones_like(rep)), ('switch', sw), ('repeat', rep)]:
        base = valid & pc & rmask
        ce_rt_c = _mean(rt, base & p_cong & incong)   - _mean(rt, base & p_cong & cong)
        ce_rt_i = _mean(rt, base & p_incong & incong) - _mean(rt, base & p_incong & cong)
        ce_ac_c = _mean(acc, base & p_cong & cong)    - _mean(acc, base & p_cong & incong)
        ce_ac_i = _mean(acc, base & p_incong & cong)  - _mean(acc, base & p_incong & incong)
        e[f'sce_rt_{rname}']  = ce_rt_c - ce_rt_i
        e[f'sce_acc_{rname}'] = ce_ac_c - ce_ac_i

    # 4. Post-error, incongruent trial A only
    for bn, bm in [('I', incong), ('C', cong)]:
        a_e, a_c = valid & p_incong & perr & bm, valid & p_incong & pc & bm
        e[f'pia_B{bn}'] = _mean(acc, a_e) - _mean(acc, a_c)
        e[f'pes_B{bn}'] = _mean(rt,  a_e) - _mean(rt,  a_c)
        e[f'focus_in_diff_B{bn}'] = _mean(foc_in, a_e) - _mean(foc_in, a_c)
    ce_err  = _mean(rt, valid & p_incong & perr & incong) - _mean(rt, valid & p_incong & perr & cong)
    ce_corr = _mean(rt, valid & p_incong & pc   & incong) - _mean(rt, valid & p_incong & pc   & cong)
    e['peri'] = ce_corr - ce_err

    # 5. What drives the update — measured on the trial itself, not the one after
    e['dfocus_near_err']  = _mean(dfoc, near & incong & ~corr_dec)
    e['dfocus_far_err']   = _mean(dfoc, far  & incong & ~corr_dec)
    e['dfocus_near_corr'] = _mean(dfoc, near & incong &  corr_dec)
    e['dfocus_far_corr']  = _mean(dfoc, far  & incong &  corr_dec)
    e['dfocus_near_minus_far_err'] = e['dfocus_near_err'] - e['dfocus_far_err']
    e['focus_in_after_near_err'] = _mean(foc_in, valid & p_incong & p_near & perr)
    e['focus_in_after_far_err']  = _mean(foc_in, valid & p_incong & p_far  & perr)
    return e


def collect_by_seed(p_congruent, rt_threshold=RT_THRESHOLD):
    """Load every seed at one congruency level. Returns {seed: effects dict}."""
    out = {}
    for res in load_condition(p_congruent):
        trials = extract_trials(res['train_logger'], res['config'], rt_threshold=rt_threshold)
        out[res['seed']] = session_effects(trials)
    return out


def collect(p_congruent, rt_threshold=RT_THRESHOLD):
    """Load every seed at one congruency level and stack per-seed effects."""
    by_seed = collect_by_seed(p_congruent, rt_threshold)
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

def main(reference_p=0.8):
    export_dir = os.path.dirname(os.path.dirname(result_path(0, reference_p)))
    print('=' * 84)
    print(f'ACROSS-SUBJECT SUMMARY  (each seed = one subject, reference p_congruent={reference_p})')
    print('=' * 84)

    eff = collect(reference_p)
    if not eff:
        print('No results found — run flanker_sweep.py first.')
        return

    report(eff, [('acc_overall', 'overall accuracy'),
                 ('censored_frac', 'censored RT fraction'),
                 ('focus_all', 'Z focus (control state)')],
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

    # ── Proportion-congruent manipulation ─────────────────────────────────────
    print('\n' + '=' * 84)
    print('LIST-WIDE PROPORTION CONGRUENT  (Aim 2 prediction: effects shrink as C gets common)')
    print('=' * 84)
    levels, acc_m, acc_s, rt_m, rt_s = [], [], [], [], []
    by_level = {}
    for p in P_CONGRUENT_LEVELS:
        by_level[p] = collect_by_seed(p)
        ep = collect(p)
        if not ep:
            continue
        sa, sr = summarize(ep['cong_effect_acc']), summarize(ep['cong_effect_rt'])
        levels.append(p)
        acc_m.append(sa['mean']); acc_s.append(sa['sem'])
        rt_m.append(sr['mean']);  rt_s.append(sr['sem'])
        print(f'  p_congruent={p}:  accuracy effect {sa["mean"]:.4f} ± {sa["sem"]:.4f}   '
              f'RT effect {sr["mean"]:.4f} ± {sr["sem"]:.4f}   (n={sa["n"]} seeds)  '
              f'censored {summarize(ep["censored_frac"])["mean"]:.3f}')

    # The same pretrained model is tested at every level, so this is a within-subject
    # manipulation and the paired contrast is the statistic that matters.
    if len(levels) >= 2:
        lo, hi = min(levels), max(levels)
        shared = sorted(set(by_level[lo]) & set(by_level[hi]))
        print(f'\n  Paired within-subject contrast, p={hi} minus p={lo}  (n={len(shared)} seeds)')
        print(f'  {"effect":<32} {"mean":>9} {"SEM":>8} {"t":>7} {"p":>9}')
        for key, label in [('cong_effect_acc', 'congruency effect, accuracy'),
                           ('cong_effect_rt',  'congruency effect, RT')]:
            diff = np.array([by_level[hi][s][key] - by_level[lo][s][key] for s in shared])
            s = summarize(diff)
            star = '***' if s['p'] < 0.001 else '**' if s['p'] < 0.01 else '*' if s['p'] < 0.05 else ''
            print(f'  {label:<32} {s["mean"]:9.4f} {s["sem"]:8.4f} {s["t"]:7.2f} {s["p"]:9.5f} {star:>4}')
        print('  Positive = larger congruency effect in the mostly-congruent list, i.e. the '
              'classic\n  list-wide proportion-congruent direction (control relaxes when '
              'conflict is rare).')

    if levels:
        fig, axes = plt.subplots(1, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1]))
        for ax, mu, se, ylabel in [(axes[0], acc_m, acc_s, 'Congruency effect (accuracy)'),
                                   (axes[1], rt_m, rt_s, 'Congruency effect (RT)')]:
            ax.errorbar(levels, mu, yerr=se, marker='o', markersize=4,
                        color='#4393c3', capsize=3, linewidth=0.9)
            ax.set_xlabel('P(congruent)')
            ax.set_ylabel(ylabel)
            ax.set_xticks(levels)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        fig.suptitle(f'List-wide proportion congruent — mean ± SEM across {SEEDS} seeds', fontsize=7)
        fig.tight_layout()
        out = os.path.join(export_dir, 'flanker_proportion_congruent.pdf')
        fig.savefig(out, bbox_inches='tight')
        print(f'\nExported: {out}')


if __name__ == '__main__':
    main()
