"""
flanker_regression.py — trial-level regression on accuracy and RT, mirroring the
behavioural analysis used on the human flanker data.

Reference implementation: `code_shared/STA_BH/Regression_on_Behavior.m` (Fischer et al.
2018), fit per participant with MATLAB `fitglm`, then one-sample t-tested across
participants by `helper/AGFHK_summary.m`. Here each *session* plays the role of a
participant.

Why a regression when we already have cell contrasts
────────────────────────────────────────────────────
The cell contrasts in `flanker_metrics.py` each condition on one factor at a time, so
they cannot separate effects that move together. An error is followed by a slower trial,
but errors also happen mostly on incongruent trials, after slow trials, and later in a
session — the regression estimates all of those simultaneously and reports what is left
of each. It also turns the control signatures into single coefficients:

    PES  = prev_error            on log RT      (positive = slower after an error)
    PIA  = prev_error            on accuracy    (positive = more accurate after an error)
    PERI = incong:prev_error     on both        (see EXPECTED for the sign in each)

Human coding conventions copied from STA_BH
───────────────────────────────────────────
- The accuracy DV is **ACC (1 = correct)**, not error probability. A positive
  coefficient means *more accurate*. (Hans's DBM package models Error instead, so its
  signs are flipped relative to these.)
- The RT model **keeps error trials** and controls for accuracy with an `is_error`
  regressor, rather than dropping them. There is no RT trimming.
- Binary factors enter as 0/1 indicators. MATLAB's `fitglm` with `'CategoricalVars'`
  dummy-codes the ±1 factors, which gives exactly this contrast: each coefficient is the
  difference between the two levels, and main effects are simple effects at the reference
  level (congruent / near / post-correct).
- `Trnr` is the trial index divided by the session length (the code comment calls it
  log-scaled; the data is linear).
- Rows with an undefined lag are dropped, as MATLAB's `nanrem` does.

What we cannot mirror: the human `RSI` factor (short/long response-stimulus interval) and
block resets. The model has neither, so PES here is comparable to the human result in
direction but not in magnitude.

Run
───
Interactively (the intended way — set the run once, then call):

    import flanker_regression as reg
    reg.describe_runs()                     # what is on disk and how each run differs
    reg.RUN = 'sweep_td03_n10'              # see SWEEP_RUNS below for the options

    summaries = reg.group_report()          # signatures + M2 vs M3, across sessions
    fig = reg.fig_group_coefficients(summaries)

From a shell:

    python flanker_regression.py --variant spatial_steep --p 0.5 [--run sweep_sgd]

Both fit every session in one sweep condition and t-test each coefficient across
sessions. `run_flanker.py` fits a single session, which is one synthetic subject and
should not be read as evidence on its own.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from flanker_analyses import lagged_factors

# ── Which sweep to read ───────────────────────────────────────────────────────
#
# Set this once at the top of an interactive session and every function below follows it:
#
#     import flanker_regression as reg
#     reg.RUN = 'sweep_sgd'
#     summaries = reg.group_report()          # prints the tables
#     fig = reg.fig_group_coefficients(summaries)
#
# `describe_runs()` reads the configs off disk, so it reports what a run actually
# contains rather than what these notes claim.

#: The registry lives in flanker_sweep, beside RUN_NAME. Re-exported here so an
#: interactive session only has to import this module.
from flanker_sweep import SWEEP_RUNS, describe_runs, use_run          # noqa: E402

#: Default run for every function here. None follows flanker_sweep_config.RUN_NAME.
RUN = None


# ── Model specifications ──────────────────────────────────────────────────────
#
# M1 is the Fischer 2018 design, minus RSI. M2 adds the lab's own extended terms
# (documented in the header comment of helper/AGFHK_Eval_Model.m) plus lag-2 history and
# target repetition. M3 adds the model's own control state.
#
# NOTE on a collinearity trap: `cong_rep` (their CongRepeat) is an exact linear
# combination of {1, incong, prev_incong, incong:prev_incong}, so it cannot appear
# alongside the Gratton interaction. Their extended model uses CongRepeat *instead of*
# the interaction; we keep the interaction, because that is the term the conflict-
# adaptation prediction is about, and drop CongRepeat.

_M1_ACC = 'acc ~ incong + far + prev_error + trial_progress + incong:prev_error'
_M1_RT  = ('logrt ~ incong + is_error + far + prev_error + trial_progress '
           '+ incong:prev_error')

_M2_EXTRA = ('+ prev_incong + prev_far + prev_logrt_c + resp_rep + target_rep + resp_side '
             '+ prev2_incong + prev2_error '
             '+ incong:far + incong:prev_incong + prev_error:resp_rep')

MODELS = {
    'M1': dict(
        label='M1 — Fischer 2018 core (minus RSI)',
        acc=_M1_ACC,
        rt=_M1_RT,
    ),
    'M2': dict(
        label='M2 — extended sequential',
        acc=_M1_ACC + _M2_EXTRA,
        rt=_M1_RT + _M2_EXTRA,
    ),
    'M3': dict(
        label='M3 — extended + inherited Z focus',
        acc=_M1_ACC + _M2_EXTRA + ' + focus_in_z',
        rt=_M1_RT + _M2_EXTRA + ' + focus_in_z',
    ),
}

# ── What each term means, and which way it should point in humans ─────────────
# (label, expected sign on accuracy, expected sign on log RT). None = no prior.
# Signs follow from: accuracy DV is P(correct), reference level is congruent / near /
# post-correct, and near flankers are the ones that interfere most.

TERMS = {
    'incong':              ('Congruency effect',                    -1, +1),
    'far':                 ('Distance (far vs near), congruent',    -1, +1),
    'incong:far':          ('Distance x congruency',                +1, -1),
    'prev_error':          ('Post-error: PIA (acc) / PES (RT)',     +1, +1),
    'incong:prev_error':   ('PERI — post-error interference drop',  +1, -1),
    'prev_incong':         ('Previous trial incongruent',         None, None),
    'incong:prev_incong':  ('Gratton — conflict adaptation',        +1, -1),
    'prev2_incong':        ('Lag-2 congruency',                   None, None),
    'prev2_error':         ('Lag-2 error',                        None, None),
    'resp_rep':            ('Response repetition (priming)',        +1, -1),
    'target_rep':          ('Target repetition (Yu & Cohen)',       +1, -1),
    'prev_error:resp_rep': ('Post-error x repetition',            None, None),
    'prev_far':            ('Previous trial far',                 None, None),
    'prev_logrt_c':        ('Previous log RT',                    None, None),
    'resp_side':           ('Response side (nuisance)',           None, None),
    'trial_progress':      ('Time on task',                       None, None),
    'is_error':            ('Error trial (RT model only)',        None, None),
    'focus_in_z':          ('Inherited Z focus',                    +1, -1),
    'Intercept':           ('Intercept',                          None, None),
}

#: The rows that answer "does the model reproduce the human signature?"
SIGNATURE_TERMS = ['incong', 'incong:far', 'prev_error', 'incong:prev_error',
                   'incong:prev_incong', 'resp_rep', 'target_rep']

#: What the summary view plots. Everything else in the design is a nuisance regressor:
#: it is there to be controlled for, not to be read.
SUMMARY_TERMS = SIGNATURE_TERMS + ['focus_in_z']

#: History terms that focus_in should absorb if the latent is what carries trial history.
MEDIATION_TERMS = ['prev_error', 'incong:prev_error', 'prev_incong',
                   'incong:prev_incong', 'prev2_incong', 'prev2_error']


def term_label(term):
    return TERMS.get(term, (term, None, None))[0]


def expected_sign(term, dv):
    """+1, -1 or None — the direction a human dataset shows for this term."""
    entry = TERMS.get(term)
    if entry is None:
        return None
    return entry[1] if dv == 'acc' else entry[2]


# ── Design matrix ─────────────────────────────────────────────────────────────

def build_design(trials, n_back=2):
    """
    One row per trial, every regressor the specs above can ask for.

    Rows with an undefined lag carry NaN and are dropped at fit time (`nanrem`).
    Continuous regressors are centered, as in the human code; `focus_in` is z-scored so
    its coefficient is per SD of control state and is comparable across sessions.
    """
    f = lagged_factors(trials, n_back=max(n_back, 2))
    n = trials['n_trials']

    rt = np.asarray(trials['rt_interp'], dtype=float)
    acc = np.asarray(trials['correct_at_decision'], dtype=float)
    focus_in = np.asarray(trials['focus_in'], dtype=float)

    # A trial whose decision variable is already past threshold at the first eligible
    # timestep gets rt exactly 0, and log(0) is -inf. That is rare (21 trials in 50k
    # across the SGD sweep, all in one seed) but one infinity poisons the centering and
    # then the whole design matrix. Treat it as missing, which is what the human
    # pipeline's `nanrem` does with an undefined row.
    with np.errstate(divide='ignore', invalid='ignore'):
        logrt = np.log(np.where(rt > 0, rt, np.nan))
        prev_logrt = np.log(np.where(f['rt_1'] > 0, f['rt_1'], np.nan))

    df = pd.DataFrame({
        # outcomes
        'acc':            acc,
        'is_error':       1.0 - acc,
        'logrt':          logrt,
        'rt':             rt,
        # current trial
        'incong':         1.0 - f['cong'],
        'far':            1.0 - f['near'],
        'resp_side':      f['side'],
        'trial_progress': np.arange(n, dtype=float) / max(n, 1),
        # previous trial
        'prev_error':     1.0 - f['correct_1'],
        'prev_incong':    1.0 - f['cong_1'],
        'prev_far':       1.0 - f['near_1'],
        'prev_logrt_c':   prev_logrt - np.nanmean(prev_logrt),
        # two back
        'prev2_incong':   1.0 - f['cong_2'],
        'prev2_error':    1.0 - f['correct_2'],
        # sequence structure
        'resp_rep':       f['resp_rep'],
        'target_rep':     f['target_rep'],
        'cong_rep':       f['cong_rep'],
        # the model's own control state, as inherited by this trial
        'focus_in_z':     (focus_in - np.nanmean(focus_in)) / np.nanstd(focus_in),
        # bookkeeping, not regressors
        'decided':        np.asarray(trials['decided'], dtype=float),
        'trial_idx':      np.arange(n, dtype=float),
    })
    return df


# ── Fitting ───────────────────────────────────────────────────────────────────

def _tidy(result, dv, n_used, converged=True, extra=None, formula=''):
    """statsmodels result -> tidy DataFrame, one row per coefficient."""
    table = pd.DataFrame({
        'term':  result.params.index,
        'coef':  result.params.values,
        'se':    result.bse.values,
        'stat':  result.tvalues.values,      # z for Logit, t for OLS
        'p':     result.pvalues.values,
    })
    table['label'] = [term_label(t) for t in table['term']]
    table['expected'] = [expected_sign(t, dv) for t in table['term']]

    # Without the interaction, `far` is a single pooled slope across both congruencies,
    # not the congruent simple effect — and the two point opposite ways here, because
    # far flankers barely help when congruent but help a lot when incongruent. Do not
    # score it against a directional prediction it is not estimating.
    if 'incong:far' not in formula.replace(' ', ''):
        mask = table['term'] == 'far'
        table.loc[mask, 'expected'] = None
        table.loc[mask, 'label'] = 'Distance (far vs near), pooled'
    table['dv'] = dv
    table['n'] = n_used
    table['converged'] = converged
    for key, value in (extra or {}).items():
        table[key] = value
    return table


def fit_session(trials, spec='M1', decided_only=False, design=None):
    """
    Fit one session's accuracy and RT models.

    decided_only : restrict to trials that crossed threshold inside the window. The
                   default (False) matches the human data, where every trial has a
                   response; our never-crossed trials carry an extrapolated RT.

    Returns {'acc': tidy table, 'rt': tidy table, 'design': the DataFrame used}.
    """
    import statsmodels.formula.api as smf

    if spec not in MODELS:
        raise ValueError(f'Unknown spec {spec!r}. Options: {list(MODELS)}')
    df = build_design(trials) if design is None else design
    if decided_only:
        df = df[df['decided'] == 1.0]

    out = {'design': df}

    # Accuracy: logistic on P(correct), all trials with a defined history.
    acc_formula = MODELS[spec]['acc']
    converged = True
    try:
        fit = smf.logit(acc_formula, data=df).fit(disp=0)
        converged = bool(getattr(fit, 'mle_retvals', {}).get('converged', True))
    except Exception as exc:                      # separation in a thin cell
        print(f'  [{spec}] logistic fit fell back to regularized: {exc}')
        fit = smf.logit(acc_formula, data=df).fit_regularized(disp=0)
        converged = False
    out['acc'] = _tidy(fit, 'acc', int(fit.nobs), converged,
                       extra={'pseudo_r2': float(getattr(fit, 'prsquared', np.nan)),
                              'bic': float(getattr(fit, 'bic', np.nan))},
                       formula=acc_formula)

    # RT: OLS on log RT, error trials kept and controlled for (STA_BH convention).
    rt_formula = MODELS[spec]['rt']
    rt_fit = smf.ols(rt_formula, data=df).fit()
    out['rt'] = _tidy(rt_fit, 'rt', int(rt_fit.nobs), True,
                      extra={'pseudo_r2': float(rt_fit.rsquared),
                             'bic': float(rt_fit.bic)},
                      formula=rt_formula)

    # STA_BH excludes a participant whose fit is degenerate; flag rather than drop.
    for dv in ('acc', 'rt'):
        if np.nanmax(np.abs(out[dv]['stat'].values)) > 1500:
            print(f'  [{spec}/{dv}] WARNING: |t| > 1500 — STA_BH would exclude this subject.')
    return out


def fit_sessions(trials_list, spec='M1', decided_only=False):
    """
    Fit every session, then test each coefficient across sessions.

    This is the group level: each session is a subject, exactly as each participant is in
    `AGFHK_summary.m`. Returns (per_session, summary) where summary carries the mean
    coefficient, SEM, t, p and the count of sessions with the expected sign.
    """
    from scipy.stats import ttest_1samp

    rows = []
    for i, trials in enumerate(trials_list):
        fits = fit_session(trials, spec=spec, decided_only=decided_only)
        for dv in ('acc', 'rt'):
            table = fits[dv].copy()
            table['session'] = i
            rows.append(table)
    per_session = pd.concat(rows, ignore_index=True)

    out = []
    for (dv, term), group in per_session.groupby(['dv', 'term'], sort=False):
        v = group['coef'].values
        v = v[~np.isnan(v)]
        t, p = ttest_1samp(v, 0.0) if len(v) > 1 else (np.nan, np.nan)
        exp = expected_sign(term, dv)
        n_ok = int(np.sum(np.sign(v) == exp)) if exp is not None else np.nan
        out.append(dict(dv=dv, term=term, label=term_label(term), mean=v.mean(),
                        sem=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                        t=t, p=p, n_sessions=len(v), expected=exp, n_expected_sign=n_ok))
    return per_session, pd.DataFrame(out)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(table, title=None, terms=None):
    """Print one tidy coefficient table, labelled by meaning rather than by term name."""
    from flanker_analyses import _console_safe

    rows = table if terms is None else table[table['term'].isin(terms)]
    if title:
        n = int(rows['n'].iloc[0]) if len(rows) else 0
        r2 = rows['pseudo_r2'].iloc[0] if len(rows) else np.nan
        print(_console_safe(f'\n{title}   (n={n} trials, R2={r2:.3f})'))
        print(f'  {"term":<22}{"meaning":<38}{"coef":>9}{"SE":>8}{"stat":>8}{"p":>10}')
    for _, r in rows.iterrows():
        if r['term'] == 'Intercept':
            continue
        star = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else ''
        mark = ''
        if r['expected'] is not None and not (isinstance(r['expected'], float) and np.isnan(r['expected'])):
            mark = ' OK' if np.sign(r['coef']) == r['expected'] else ' <-- opposite'
        print(_console_safe(f'  {r["term"]:<22}{r["label"]:<38}{r["coef"]:9.4f}{r["se"]:8.4f}'
                            f'{r["stat"]:8.2f}{r["p"]:10.5f} {star}{mark}'))


def signature_report(fits):
    """
    One row per human signature: the coefficient, its direction, and the verdict.

    This is the single-session analogue of the group scorecard — the summary of what the
    model reproduces and what it does not.
    """
    rows = []
    for dv in ('acc', 'rt'):
        table = fits[dv].set_index('term')
        for term in SIGNATURE_TERMS:
            if term not in table.index:
                continue
            r = table.loc[term]
            exp = r['expected']
            if exp is None or (isinstance(exp, float) and np.isnan(exp)):
                continue
            matches = np.sign(r['coef']) == exp
            if r['p'] >= 0.05:
                verdict = 'null'
            else:
                verdict = 'MATCH' if matches else 'WRONG DIRECTION'
            rows.append(dict(dv='accuracy' if dv == 'acc' else 'log RT',
                             signature=r['label'], coef=r['coef'], p=r['p'],
                             expected='+' if exp > 0 else '-', verdict=verdict))
    return pd.DataFrame(rows)


def print_signatures(fits):
    """
    One line per signature, accuracy and RT side by side.

    `signature_report` returns the same content in tidy long form (one row per
    signature x outcome), which is the right shape for further analysis and the wrong
    shape for reading — it doubles the number of lines for no extra information.
    """
    from flanker_analyses import _console_safe

    report = signature_report(fits)
    wide = {}
    for _, r in report.iterrows():
        wide.setdefault(r['signature'], {})[r['dv']] = r
    print(f'  {"signature":<36}{"accuracy":>22}{"log RT":>22}')
    for name, cells in wide.items():
        line = f'  {name:<36}'
        for dv in ('accuracy', 'log RT'):
            if dv in cells:
                r = cells[dv]
                mark = {'MATCH': 'match', 'null': 'null', 'WRONG DIRECTION': 'WRONG WAY'}[r['verdict']]
                line += f'{r["coef"]:>11.3f}  {mark:<9}'
            else:
                line += ' ' * 22
        print(_console_safe(line))


def report_session(reg, detail=False, design=None):
    """
    Print the regression results at one of two levels of detail.

    reg    : {spec name: fit_session output}, e.g. {'M2': ..., 'M3': ...}
    detail : False prints only the human signatures and the mediation — the ~6 rows that
             are results. True adds every model's full coefficient table and the VIFs,
             including the nuisance regressors that exist to be controlled for rather
             than read. Use it when auditing a fit or comparing against the human
             coefficients, not on every run.
    design : the design matrix, needed only for the VIF block in detail mode.
    """
    if detail:
        for spec, fits in reg.items():
            print_table(fits['acc'], title=f'{MODELS[spec]["label"]} — ACCURACY (1 = correct)')
            print_table(fits['rt'],  title=f'{MODELS[spec]["label"]} — log RT')
        if design is not None and 'M3' in reg:
            vif = collinearity(design, 'M3').sort_values('vif', ascending=False)
            print('\nHighest VIFs in M3 (>5 would make the mediation unreadable):')
            print(vif.head(5).to_string(index=False))
    elif 'M2' in reg:
        print('\nHuman signatures, from the M2 coefficients (everything else controlled):')
        print_signatures(reg['M2'])

    if 'M2' in reg and 'M3' in reg:
        print('\nDoes the inherited control state absorb trial history? (M2 -> M3)')
        print(f'  {"term":<22}{"accuracy M2":>12}{"M3":>10}   {"log RT M2":>11}{"M3":>10}')
        for term in MEDIATION_TERMS:
            row = f'  {term:<22}'
            for dv in ('acc', 'rt'):
                a = reg['M2'][dv].set_index('term'); b = reg['M3'][dv].set_index('term')
                if term in a.index and term in b.index:
                    row += f'{a.loc[term, "coef"]:>12.4f}{b.loc[term, "coef"]:>10.4f}   '
            print(row)
        focus = reg['M3']['rt'].set_index('term')
        if 'focus_in_z' in focus.index:
            r = focus.loc['focus_in_z']
            print(f'  focus_in itself on log RT: {r["coef"]:+.4f}  (t={r["stat"]:.1f}, '
                  f'p={r["p"]:.2g}) per SD of control state')


def collinearity(design, spec='M2'):
    """
    Variance inflation factors for one spec's regressors.

    STA_BH stores the design correlation matrix per subject for the same reason: when
    `focus_in_z` is added in M3 it competes with the history terms that generated it, and
    a mediation that looks strong can be collinearity instead.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    y, X = np.asarray([]), None
    model = smf.ols(MODELS[spec]['rt'], data=design)
    X = model.exog
    names = model.exog_names
    vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return pd.DataFrame({'term': names, 'vif': vifs}).query('term != "Intercept"')


# ── Group level ───────────────────────────────────────────────────────────────

def group_signatures(summary, spec_label='M2'):
    """
    The digestible table, at group level: one row per human signature.

    Same content as `signature_report` for a single session, but the statistic is the
    one-sample t-test across sessions rather than the trial-level p-value, and it carries
    the count of sessions with the predicted sign — the honest summary when a group mean
    rests on one or two outliers.
    """
    rows = []
    for dv, dv_label in (('acc', 'accuracy'), ('rt', 'log RT')):
        for term in SIGNATURE_TERMS:
            if (dv, term) not in summary.index:
                continue
            r = summary.loc[(dv, term)]
            exp = r['expected']
            if exp is None or (isinstance(exp, float) and np.isnan(exp)):
                continue
            if r['p'] >= 0.05:
                verdict = 'null'
            else:
                verdict = 'MATCH' if np.sign(r['mean']) == exp else 'WRONG DIRECTION'
            rows.append(dict(signature=term_label(term), dv=dv_label, mean=r['mean'],
                             sem=r['sem'], p=r['p'],
                             signs=f'{int(r["n_expected_sign"])}/{int(r["n_sessions"])}'
                                   if not np.isnan(r['n_expected_sign']) else '',
                             verdict=verdict))
    return pd.DataFrame(rows)


def print_group_signatures(summary, title=None):
    """One line per signature, accuracy and RT side by side, group statistics."""
    from flanker_analyses import _console_safe

    table = group_signatures(summary)
    wide = {}
    for _, r in table.iterrows():
        wide.setdefault(r['signature'], {})[r['dv']] = r
    if title:
        print(_console_safe(f'\n{title}'))
    print(f'  {"signature":<34}{"accuracy":>26}{"log RT":>26}')
    for name, cells in wide.items():
        line = f'  {name[:32]:<34}'
        for dv in ('accuracy', 'log RT'):
            if dv in cells:
                r = cells[dv]
                mark = {'MATCH': 'match', 'null': 'null', 'WRONG DIRECTION': 'WRONG WAY'}[r['verdict']]
                line += f'{r["mean"]:>9.3f} {mark:<10}{r["signs"]:>6} '
            else:
                line += ' ' * 26
        print(_console_safe(line))
    print('  (mean across sessions; "signs" = sessions with the human-predicted direction)')


def group_report(p_congruent=0.5, variant='spatial_steep', run=None, specs=('M2', 'M3'),
                 terms=None, signatures=True):
    """
    Fit every session in one condition, then t-test each coefficient across sessions.

    This is the across-seed analysis: one session plays the role of one participant, and
    the error bar that matters is the spread between sessions, not the trial-level SE
    inside one. With two specs it also reports which terms change significance when the
    control state enters the model.

    run : sweep RUN_NAME to read; defaults to whatever flanker_sweep_config names.
    """
    import flanker_sweep
    from flanker_analyses import extract_trials

    with use_run(run or RUN or flanker_sweep.RUN_NAME) as run_label:
        results = flanker_sweep.load_condition(p_congruent, variant=variant)
        trials = [extract_trials(r['train_logger'], r['config']) for r in results]

    if not trials:
        print('No sessions found — check the run name, variant and p_congruent.')
        return {}

    summaries = {}
    for spec in specs:
        _, summary = fit_sessions(trials, spec=spec)
        summaries[spec] = summary.set_index(['dv', 'term'])

    terms = terms or (SIGNATURE_TERMS + ['prev_incong', 'prev2_incong', 'prev2_error'])
    star = lambda q: '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else 'ns'
    head, extra = specs[0], specs[1] if len(specs) > 1 else None

    print(f'\n{variant}, p(congruent)={p_congruent}, run={run_label}, '
          f'{len(trials)} sessions x {trials[0]["n_trials"]} trials')
    if signatures and head in summaries:
        print_group_signatures(summaries[head],
                               title=f'Human signatures, {MODELS[head]["label"]}')
        print()
    for dv, dv_label in (('acc', 'ACCURACY (1 = correct)'), ('rt', 'log RT')):
        print(f'\n  {dv_label}')
        cols = f'{head:>18}' + (f'{extra:>18}   change' if extra else '')
        print(f'  {"term":<34}{cols}')
        for term in terms:
            if (dv, term) not in summaries[head].index:
                continue
            a = summaries[head].loc[(dv, term)]
            line = f'  {term_label(term)[:32]:<34}{a["mean"]:11.4f} {star(a["p"]):<6}'
            if extra and (dv, term) in summaries[extra].index:
                b = summaries[extra].loc[(dv, term)]
                sa, sb = star(a['p']), star(b['p'])
                change = ('LOST when Z added' if sa != 'ns' and sb == 'ns' else
                          'GAINED when Z added' if sa == 'ns' and sb != 'ns' else '')
                line += f'{b["mean"]:11.4f} {sb:<6}   {change}'
            print(line)
        if extra and (dv, 'focus_in_z') in summaries[extra].index:
            r = summaries[extra].loc[(dv, 'focus_in_z')]
            print(f'  {"[Z focus itself]":<34}{"—":>11} {"":<6}{r["mean"]:11.4f} '
                  f'{star(r["p"]):<6}   t={r["t"]:.1f}')
    return summaries


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_coefficients(ax, table, title=None, terms=None, color='#4393c3', metric='stat',
                      order=None, y_offset=0.0):
    """
    Forest plot, one row per term.

    metric='stat' plots the t/z value with dashed guides at +/-1.96, which is what
    STA_BH plots and the only sensible choice when the two panels are in different units
    (log-odds for accuracy, log timesteps for RT) and one coefficient — the congruency
    effect — is an order of magnitude larger than the rest. metric='coef' plots the raw
    coefficient with its 95% CI instead; the printed tables always carry both.

    order : term names, top to bottom. Rows missing from `table` are left blank, so two
            models with different term sets can be overlaid on the same axis.
    y_offset : nudge this series vertically. Two overlaid models otherwise plot at the
            same height, and where they agree — which is most rows — one hides the other
            and the panel looks like it is missing a series.
    """
    rows = table if terms is None else table[table['term'].isin(terms)]
    rows = rows[rows['term'] != 'Intercept'].set_index('term')
    order = list(rows.index) if order is None else [t for t in order if t != 'Intercept']

    y_all = np.arange(len(order))[::-1]
    x, xerr, y = [], [], []
    for term, y_pos in zip(order, y_all):
        if term not in rows.index:
            continue
        r = rows.loc[term]
        x.append(r['stat'] if metric == 'stat' else r['coef'])
        xerr.append(0.0 if metric == 'stat' else 1.96 * r['se'])
        y.append(y_pos + y_offset)

    ax.errorbar(x, y, xerr=xerr if metric == 'coef' else None, fmt='o', markersize=3,
                color=color, capsize=2, linewidth=0.9, linestyle='none')
    ax.axvline(0, color='k', linewidth=0.7, alpha=0.6)
    if metric == 'stat':
        for edge in (-1.96, 1.96):
            ax.axvline(edge, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
    ax.set_yticks(y_all)
    ax.set_yticklabels([term_label(t) for t in order], fontsize=5)
    ax.set_ylim(-0.7, len(order) - 0.3)
    if title:
        ax.set_title(title, fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def fig_group_coefficients(summaries, terms=None, title=None, path=None):
    """
    The forest plot at group level: across-session t for each term, one marker per spec.

    Takes the {spec: summary} dict `group_report` returns. The statistic plotted is the
    one-sample t across sessions, so +/-1.96 is roughly the significance boundary — the
    same convention as the single-session figure, but the error bar that matters.
    """
    import matplotlib.pyplot as plt
    from plot_style import FigSize

    terms = terms or SUMMARY_TERMS
    specs = list(summaries)
    colors = ['#d6604d', '#777777', '#4393c3'][:len(specs)]
    offsets = np.linspace(0.16, -0.16, len(specs)) if len(specs) > 1 else [0.0]

    fig, axes = plt.subplots(1, 2, figsize=FigSize.custom(6.6, 0.28 * len(terms) + 1.0))
    for ax, dv, name in [(axes[0], 'acc', 'Accuracy (1 = correct)'),
                         (axes[1], 'rt', 'log RT')]:
        for spec, color, dy in zip(specs, colors, offsets):
            rows = summaries[spec].loc[dv].reset_index()
            # Reuse the session-level plotter by handing it the group statistics.
            table = pd.DataFrame({'term': rows['term'], 'coef': rows['mean'],
                                  'se': rows['sem'], 'stat': rows['t']})
            plot_coefficients(ax, table, title=name, order=terms, color=color, y_offset=dy)
        ax.set_xlabel('t across sessions  (dotted = ±1.96)')
    axes[0].legend(handles=[plt.Line2D([], [], color=c, marker='o', markersize=3,
                                       linewidth=0.9, label=MODELS[s]['label'].split(' — ')[0]
                                       + (' (no Z)' if s != 'M3' else ' (+ Z focus)'))
                            for s, c in zip(specs, colors)], fontsize=5, loc='lower left')
    fig.suptitle(title or 'Trial-history regression — across sessions', fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches='tight')
        print(f'Exported: {path}')
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def _argv_options():
    opts = {'p_congruent': 0.5, 'variant': 'spatial_steep', 'run': None}
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == '--p' and i + 1 < len(argv):
            opts['p_congruent'] = float(argv[i + 1])
        elif arg == '--variant' and i + 1 < len(argv):
            opts['variant'] = argv[i + 1]
        elif arg == '--run' and i + 1 < len(argv):
            opts['run'] = argv[i + 1]
    return opts


if __name__ == '__main__':
    group_report(**_argv_options())
