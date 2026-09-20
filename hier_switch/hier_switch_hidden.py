"""
hier_switch_hidden.py — what the hidden state X carries, and how fast it gets there.

The PFC side of the comparison with Lam et al. 2025 (the latent Z is the MD side; see
hier_switch_analyses.py). Everything here runs on a recorded session — hidden states are
saved only when a session is recorded with `record_hidden` (hier_switch_test_inference.py).

What the paper found in PFC, and what each function here measures.

* **Three cell classes.** CueS: a transient, context-specific cue response. CueL: a
  sustained, context-invariant one. Rule: opposite cue preference in the two contexts.
  `unit_classes` classifies each hidden unit the same way, from the sign agreement of its
  cue selectivity across contexts (cue-like vs rule-like) and from how much of that
  selectivity falls in the second half of the cue period (CueS vs CueL).
* **Cue and rule axes.** Within a block cue and rule are perfectly collinear
  (rule = cue × ctx_sign), so any cue-vs-rule fit must pool both contexts: `fit_axes` runs
  one regression of the hidden state on [cue, rule] over trials from both. P5 predicts the
  cue axis is the same in both contexts while the rule axis flips; `fit_axes(per_context=
  True)` gives the two fits to check that.
* **The integration index.** The paper's population measure of "is this a rule-driven or an
  input-driven regime": load +1 on every CueL and Rule neuron, orthogonalise against the
  CueS axis, and take the ratio of activity along that dimension in the second half of the
  cue period to the first. High during rule-based behaviour, low in early exploration.
  `integration` is the same construction on the fitted axes. **Cue velocity** is the
  paper's other measure: the fastest rise along the cue axis in the first half of the cue
  period. Both are per condition, not per trial.
* **Decoding.** `decode` is cross-validated logistic decoding of cue, rule, context and
  conflict from the hidden state at each timestep: what X carries, and when. The build-up
  time (first timestep at 0.75) is the integration-speed measure.

Timing (predict_first_frame=True): the output and hidden state at step t have seen input
frames < t. Pulses are frames 0-15, so steps 1-16 are the cue period; 1-8 is its first half
and 9-16 its second. Targets appear at frame 18, so steps 19-24 are the response window.

    .venv/bin/python hier_switch/hier_switch_hidden.py <session> [<session> ...]
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from hier_switch_analyses import (PRIMARY, STEADY, EARLY_N, load_session, select,
                                  trial_labels, z_updates)

CUE_PERIOD = (1, 16)            # hidden-state steps that have seen pulse frames
EARLY_HALF = (1, 8)
LATE_HALF = (9, 16)
CUE_S_WINDOW = (1, 4)           # the transient, CueS-like part of the cue period
BUILDUP_LEVEL = 0.75            # decoding accuracy that counts as "the code is there"


def _steps(window):
    return np.arange(window[0], window[1] + 1)


def _hidden(sess):
    if 'hidden' not in sess:
        raise ValueError('no hidden states in this session: re-record it with record_hidden '
                         '(hier_switch_test_inference.py does)')
    return np.asarray(sess['hidden'], dtype=np.float32)


def steady_mask(sess, primary=PRIMARY, aligned=True):
    """Steady-state trials (since ≥ 11) on which the axes are fitted; aligned Z by default.

    A session with no context axis — the RNN baseline, whose Z never moves, and any clamped
    session — has no aligned trials at all, so the aligned filter is dropped there rather
    than leaving an empty mask.
    """
    m = select(sess, phase=primary, since=(STEADY, 10 ** 9))
    if aligned and sess.get('axis') is not None:
        m = m & sess['aligned']
    return m


def fit_axes(sess, mask=None, per_context=False, window=CUE_PERIOD):
    """Per-timestep OLS of the hidden state on [cue, rule, 1], pooled over both contexts.

    Returns beta_cue / beta_rule, shape (trial_len, hidden), the unit axes averaged over the
    cue period, and the angle between them. With per_context, the same fit is run inside each
    context (where cue and rule are collinear, so only the cue column is identifiable) and
    the two cue axes are compared: P5 says they agree, while the rule axis flips.
    """
    h = _hidden(sess)
    m = steady_mask(sess) if mask is None else mask
    cue, rule = sess['cue'][m], sess['rule'][m]
    X = np.c_[cue, rule, np.ones(m.sum())]
    beta, *_ = np.linalg.lstsq(X, h[m].reshape(m.sum(), -1), rcond=None)
    beta = beta.reshape(3, h.shape[1], h.shape[2])
    res = dict(beta_cue=beta[0], beta_rule=beta[1], beta_const=beta[2], n=int(m.sum()))
    ts = _steps(window)
    res['u_cue'] = _unit(beta[0][ts].mean(axis=0))
    res['u_rule'] = _unit(beta[1][ts].mean(axis=0))
    res['u_cue_early'] = _unit(beta[0][_steps(CUE_S_WINDOW)].mean(axis=0))
    res['u_cue_late'] = _unit(beta[0][_steps(LATE_HALF)].mean(axis=0))
    res['angle_cue_rule'] = _angle(res['u_cue'], res['u_rule'])
    if per_context:
        per = []
        for c in (0, 1):
            mc = m & (sess['context'] == c)
            Xc = np.c_[sess['cue'][mc], np.ones(mc.sum())]
            b, *_ = np.linalg.lstsq(Xc, h[mc].reshape(mc.sum(), -1), rcond=None)
            per.append(_unit(b.reshape(2, h.shape[1], h.shape[2])[0][ts].mean(axis=0)))
        res['u_cue_ctx'] = per
        res['angle_cue_across_contexts'] = _angle(per[0], per[1])
        # The rule axis is the part that flips: cue axes agree, rule axes are opposite.
        res['cos_cue_across_contexts'] = float(per[0] @ per[1])
    return res


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _angle(a, b):
    c = float(np.clip(a @ b, -1, 1))
    return float(np.degrees(np.arccos(abs(c)))), float(c)


def integration(sess, axes, mask, window=CUE_PERIOD):
    """The paper's integration index and cue velocity for one set of trials.

    index  mean |projection| on the CueL-and-Rule dimension over the second half of the cue
           period, divided by the same over the first half. The dimension is
           normalise(u_cue_late + u_rule) with the transient CueS axis projected out, the
           paper's "+1 on every CueL and Rule neuron, orthogonalised against CueS".
    cue_velocity  the largest rise along the cue axis per timestep within the first half,
           measured from the start of the cue period.
    """
    h = _hidden(sess)
    if mask.sum() < 5:
        return dict(n=int(mask.sum()), index=np.nan, cue_velocity=np.nan)
    late = _unit(axes['u_cue_late'] + axes['u_rule'])
    late = _unit(late - (late @ axes['u_cue_early']) * axes['u_cue_early'])
    proj_late = np.abs(h[mask] @ late)                       # (n, trial_len)
    first = proj_late[:, _steps(EARLY_HALF)].mean()
    second = proj_late[:, _steps(LATE_HALF)].mean()
    # Cue velocity: signed by the trial's cue, so the two cues do not cancel.
    proj_cue = (h[mask] @ axes['u_cue']) * sess['cue'][mask][:, None]
    t0 = window[0]
    base = proj_cue[:, t0]
    vel = [(proj_cue[:, t] - base).mean() / (t - t0) for t in range(t0 + 1, EARLY_HALF[1] + 1)]
    return dict(n=int(mask.sum()), index=float(second / first) if first > 0 else np.nan,
                first_half=float(first), second_half=float(second),
                cue_velocity=float(np.max(vel)), cue_velocity_t=int(np.argmax(vel) + t0 + 1),
                proj_cue=proj_cue.mean(axis=0).tolist(), proj_late=proj_late.mean(axis=0).tolist())


def integration_aligned(sess, axes, window=(-5, 15), blocks=None, primary=PRIMARY,
                        min_trials=8):
    """The integration index and cue velocity at each trial offset from a reversal.

    The paper plots both across the exploration phase (Fig 3c); this is the same cut on the
    model. Both measures are population quantities defined over a *set* of trials, not per
    trial, so each offset pools the trials at that offset over every reversal block — the
    same block selection `reversal_aligned` uses, so the x axis matches the behavioural
    panels exactly. An offset with fewer than `min_trials` trials is NaN rather than noise.
    """
    from hier_switch_analyses import _blocks
    _, starts, ends = _blocks(sess)
    n, phase = sess['n'], sess['phase']
    keep = sess['reversal'] & ~sess['transient'] & (phase == primary)
    sel = [b for b in range(len(starts)) if keep[starts[b]] and (blocks is None or blocks[b])]
    ks = np.arange(window[0], window[1] + 1)
    out = dict(k=ks.tolist(), index=[], cue_velocity=[], n=[])
    for k in ks:
        m = np.zeros(n, bool)
        for b in sel:
            a, e = starts[b], ends[b]
            i = a + k - 1
            lo = starts[b - 1] if b > 0 else 0
            if lo <= i < min(e, n) and phase[i] == phase[a]:
                m[i] = True
        if m.sum() < min_trials:
            out['index'].append(np.nan)
            out['cue_velocity'].append(np.nan)
        else:
            r = integration(sess, axes, m)
            out['index'].append(r['index'])
            out['cue_velocity'].append(r['cue_velocity'])
        out['n'].append(int(m.sum()))
    return out


def decode(sess, mask=None, targets=('cue', 'rule', 'context', 'conflict'), folds=5, seed=0):
    """Cross-validated logistic decoding of each target from the hidden state, per timestep.

    Returns {target: dict(acc=[...per timestep], buildup=first step at BUILDUP_LEVEL, n)}.
    `conflict` is high (levels 3-4) against low (levels 0-1); the middle level is dropped.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    h = _hidden(sess)
    base = steady_mask(sess) if mask is None else mask
    out = {}
    for name in targets:
        m = base.copy()
        if name == 'conflict':
            y_all = np.where(sess['conf_level'] >= 3, 1, 0)
            m &= np.isin(sess['conf_level'], [0, 1, 3, 4])
        elif name == 'context':
            y_all = sess['context'].astype(int)
        else:
            y_all = (sess[name] > 0).astype(int)
        y = y_all[m]
        if m.sum() < 4 * folds or len(np.unique(y)) < 2 or np.bincount(y).min() < folds:
            out[name] = dict(acc=[np.nan] * h.shape[1], buildup=None, n=int(m.sum()))
            continue
        X = h[m]
        cv = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X[:, 0], y))
        acc = []
        for t in range(h.shape[1]):
            sc = []
            for tr, te in cv:
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, C=1.0))
                clf.fit(X[tr, t], y[tr])
                sc.append(float((clf.predict(X[te, t]) == y[te]).mean()))
            acc.append(float(np.mean(sc)))
        hit = [t for t, a in enumerate(acc) if a >= BUILDUP_LEVEL]
        out[name] = dict(acc=acc, buildup=int(hit[0]) if hit else None, n=int(m.sum()),
                         chance=float(max(np.mean(y), 1 - np.mean(y))))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# What is encoded where: the hidden state ("PFC") against Z and its gradient ("MD"/"ACC")
# ══════════════════════════════════════════════════════════════════════════════

ENCODING_VARS = ('cue', 'rule', 'context', 'conflict', 'outcome',
                 'rule_uncertainty', 'cue_uncertainty')
#: Which variables are binary (decoded by logistic regression, scored as balanced accuracy)
#: and which are continuous (ridge, scored as cross-validated R²).
_BINARY_VARS = ('cue', 'rule', 'context', 'outcome')


def encoding_variables(sess, obs=None):
    """The task and uncertainty variables every source is tested against.

    Returns {name: (values, is_binary)}. `context` is `ctx_rel` — relative to the context
    the model saw last in training, not the raw label. The two uncertainties are the
    paper's pair (Fig 2o-p), taken from the ideal observer rather than from the model, so
    they are properties of the trial sequence and not of what we are decoding from:

      rule uncertainty  1 − |2b − 1|, on the observer's predictive context belief b (before
                        it sees this trial's feedback). 1 = it has no idea which context.
      cue uncertainty   1 − max(q, 1 − q) on the observer's cue posterior. 1/2 = the pulses
                        were uninformative. This is the graded version of cue conflict.
    """
    ctx_rel = sess.get('ctx_rel', sess['context'])
    out = {'cue': ((sess['cue'] > 0).astype(float), True),
           'rule': ((sess['rule'] > 0).astype(float), True),
           'context': (np.asarray(ctx_rel, dtype=float), True),
           'conflict': (sess['conflict'].astype(float), False),
           'outcome': (sess['err'].astype(float), True)}
    n = sess['n']
    if obs is not None and 'b' in obs:
        b = np.asarray(obs['b'], dtype=float)
        out['rule_uncertainty'] = (1.0 - np.abs(2.0 * b - 1.0), False)
        q = np.asarray(obs['q'], dtype=float)
        out['cue_uncertainty'] = (1.0 - np.maximum(q, 1.0 - q), False)
    else:
        out['rule_uncertainty'] = (np.full(n, np.nan), False)
        out['cue_uncertainty'] = (np.full(n, np.nan), False)
    return out


#: Sources with at most this many columns are given a magnitude channel — see `_augment`.
LOW_DIM = 8


def _contrast_gain(v):
    """A 2-unit latent signal in its meaningful coordinates: (contrast, gain).

    A rotation of the raw pair, but the one that names things: the contrast is what the
    softmax acts on and what selects the context, the gain is the common mode.
    """
    v = np.asarray(v, dtype=float)
    return np.c_[0.5 * (v[:, 0] - v[:, 1]), v.mean(axis=1)]


def _augment(X):
    """A low-dimensional signal plus the absolute value of each of its columns.

    Both uncertainty variables are *magnitudes* (rule uncertainty is 1 − |2b − 1|, cue
    uncertainty is 1 − max(q, 1 − q)), and a linear read-out of a signed two-dimensional
    vector cannot form a magnitude. Without this channel, "Z carries no rule uncertainty"
    would be a statement about the decoder rather than about Z. Every source of at most
    LOW_DIM columns gets it, so the comparison between them stays symmetric; the 64-unit
    hidden state is left raw, having units enough to carry such a channel itself.
    """
    X = np.asarray(X, dtype=float)
    return np.c_[X, np.abs(X)] if X.shape[1] <= LOW_DIM else X


def encoding_sources(sess, mask):
    """The per-trial feature matrices we ask "what does this carry?" of.

    The model's own PFC/MD split, plus a dimension-matched control. A 64-unit hidden state
    will out-decode a 2-unit Z on almost anything simply by having 32× the dimensions, so
    `hidden_pc2` — the top two principal components of the same hidden state — is carried
    alongside it: any hidden-vs-Z difference that survives against `hidden_pc2` is not a
    unit-count effect.

      hidden_t16  the hidden state at the end of the cue period (the PFC read-out)
      hidden_t24  the hidden state at the end of the trial
      hidden_pc2  the top 2 PCs of hidden_t16 (the dimension-matched control)
      z_in        the latent the trial actually ran under — the persistent context code
      step        z − z_in, the trial's own latent update: the transient switch signal
      grad        the error gradient on Z

    The four low-dimensional sources are all in (contrast, gain) coordinates and all carry
    a magnitude channel (`_augment`), so none of them is handicapped on the magnitude
    variables relative to the others.

    Returns {name: (X, rows)} where `rows` is the subset of `mask` with finite features.
    """
    out = {}
    if 'hidden' in sess:
        h = _hidden(sess)
        last = h.shape[1] - 1
        out['hidden_t16'] = h[:, min(CUE_PERIOD[1], last), :].astype(float)
        out['hidden_t24'] = h[:, last, :].astype(float)
    z_in, z = sess['z_in'].astype(float), sess['z'].astype(float)
    out['z_in'] = _contrast_gain(z_in)
    out['step'] = _contrast_gain(z - z_in)
    if 'grad' in sess:
        out['grad'] = _contrast_gain(sess['grad'].astype(float))
    res = {}
    for name, X in out.items():
        X = _augment(np.atleast_2d(X))
        rows = mask & np.isfinite(X).all(axis=1)
        if rows.sum() < 20:
            continue
        # Drop dead columns: under the softmax the gain is identically zero, and a constant
        # column makes the standardiser divide by ~0.
        keep = np.ptp(X[rows], axis=0) > 1e-12
        if not keep.any():
            continue
        res[name] = (X[:, keep], rows)
    # The control is fitted on the same rows the full hidden state uses.
    if 'hidden_t16' in res:
        X, rows = res['hidden_t16']
        Xc = X[rows] - X[rows].mean(axis=0)
        # Two leading PCs; economy SVD on the centred, masked block.
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        res['hidden_pc2'] = (_augment((X - X[rows].mean(axis=0)) @ vt[:2].T), rows)
    return res


def _standardise(X):
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)


def _cv_logistic(X, y, folds=5, seed=0):
    """Cross-validated balanced accuracy of a logistic decoder. NaN if it cannot be fitted."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or np.bincount(y).min() < folds:
        return np.nan
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    sc = []
    for tr, te in cv.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        sc.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(sc))


def _cv_ridge(X, y, folds=5, seed=0):
    """Cross-validated R² of a ridge decoder, clipped at 0 (a worse-than-mean fit is 'no
    information', and a small negative R² is only noise about that)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y, dtype=float)
    if np.std(y) < 1e-12:
        return np.nan
    sc = []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        mdl.fit(X[tr], y[tr])
        p = mdl.predict(X[te])
        ss = np.sum((y[te] - y[te].mean()) ** 2)
        sc.append(1.0 - np.sum((y[te] - p) ** 2) / ss if ss > 0 else np.nan)
    return float(np.clip(np.nanmean(sc), 0.0, 1.0))


def _r2_multi(Y, X):
    """Total R² of an OLS fit of every column of Y on X (with intercept), pooled over
    columns as 1 − ΣSS_res / ΣSS_tot."""
    A = np.c_[X, np.ones(len(X))]
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = Y - A @ beta
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
    return float(1.0 - np.sum(res ** 2) / ss_tot) if ss_tot > 0 else np.nan


def demixed_variance(Y, variables, n_perm=0, seed=0):
    """How much of a source's variance each variable uniquely explains.

    The paper's Fig 2k ("MD neurons encode one variable, PFC neurons several") asked at the
    population level and without a neuron-count confound: fit every column of the source
    jointly on all the variables at once, then drop one variable and see how much total R²
    falls. That drop is the variance *uniquely* attributable to it — variance two variables
    share is not credited to either. What is left over is reported honestly as `shared`
    (explained, but not uniquely) and `residual` (not explained), so the parts sum to 1.

    With `n_perm > 0`, also returns a per-column count of how many variables each column is
    tuned to, against a null in which that variable's column is shuffled: the Fig 2k
    histogram.
    """
    Y, X = _standardise(Y), _standardise(np.column_stack(variables[1]))
    names = list(variables[0])
    full = _r2_multi(Y, X)
    unique = {}
    for j, nm in enumerate(names):
        unique[nm] = float(full - _r2_multi(Y, np.delete(X, j, axis=1)))
    res = dict(r2_full=float(full), unique=unique,
               shared=float(full - sum(unique.values())), residual=float(1.0 - full))
    if n_perm:
        rng = np.random.default_rng(seed)
        A = np.c_[X, np.ones(len(X))]
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)          # per column
        base = np.sum((Y - A @ beta) ** 2, axis=0)
        drop = np.zeros((len(names), Y.shape[1]))
        for j in range(len(names)):
            Aj = np.c_[np.delete(X, j, axis=1), np.ones(len(X))]
            bj, *_ = np.linalg.lstsq(Aj, Y, rcond=None)
            drop[j] = (np.sum((Y - Aj @ bj) ** 2, axis=0) - base) / np.maximum(ss_tot, 1e-12)
        null = np.zeros((len(names), n_perm, Y.shape[1]))
        for j in range(len(names)):
            for p in range(n_perm):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                Ap, Apj = np.c_[Xp, np.ones(len(X))], np.c_[np.delete(Xp, j, axis=1), np.ones(len(X))]
                bp, *_ = np.linalg.lstsq(Ap, Y, rcond=None)
                bpj, *_ = np.linalg.lstsq(Apj, Y, rcond=None)
                null[j, p] = (np.sum((Y - Apj @ bpj) ** 2, axis=0)
                              - np.sum((Y - Ap @ bp) ** 2, axis=0)) / np.maximum(ss_tot, 1e-12)
        tuned = drop > np.percentile(null, 95, axis=1)               # (vars, columns)
        cnt = tuned.sum(axis=0)
        res['n_vars_per_unit'] = cnt.tolist()
        res['n_vars_hist'] = [int((cnt == k).sum()) for k in range(len(names) + 1)]
        res['mean_vars_per_unit'] = float(cnt.mean())
        res['frac_multi'] = float((cnt >= 2).mean())
    return res


def encoding_table(sess, obs=None, primary=PRIMARY, folds=5, seed=0, n_perm=50):
    """What each source carries about each variable: decoding, and demixed variance.

    Runs on every trial of the primary phase, not only the steady state: the trials right
    after a reversal are where the latent update and its gradient do their work, and a
    steady-state-only table would leave the error signal with almost nothing to carry. The
    steady-state version is returned alongside as `decoding_steady`.

    Two measures per (source, variable):
      decoding   cross-validated balanced accuracy (binary variables) or R² (continuous),
                 with a label-shuffle null per cell so "above chance" is measured, not assumed
      variance   the drop-one unique variance of `demixed_variance`

    No prediction here is enforced: the hidden state's context decoding is expected to be
    near ceiling *because Z gates it*, and the gradient's cue decoding is expected to be at
    chance because along the context axis the gradient's sign carries the context and never
    the cue. Both are reported as they come out.
    """
    mask = select(sess, phase=primary)
    if mask.sum() < 50:
        return dict(note='too few trials', n=int(mask.sum()))
    variables = encoding_variables(sess, obs)
    sources = encoding_sources(sess, mask)
    steady = mask & (sess['since'] >= STEADY)
    rng = np.random.default_rng(seed)
    res = dict(n=int(mask.sum()), sources=sorted(sources), variables=list(ENCODING_VARS),
               decoding={}, decoding_null={}, decoding_steady={}, variance={})
    for sname, (X, rows) in sources.items():
        res['decoding'][sname], res['decoding_null'][sname] = {}, {}
        res['decoding_steady'][sname] = {}
        Xr = X[rows]
        for vname in ENCODING_VARS:
            if vname not in variables:
                continue
            y, binary = variables[vname]
            yr = np.asarray(y, dtype=float)[rows]
            good = np.isfinite(yr)
            if good.sum() < 5 * folds:
                res['decoding'][sname][vname] = np.nan
                continue
            Xg, yg = Xr[good], yr[good]
            if vname == 'conflict':
                # Decoded as the paper contrasts it: the two most ambiguous levels against
                # the two least, dropping the middle one. The continuous version is what
                # the variance decomposition uses.
                cl = sess['conf_level'][rows][good]
                keep = np.isin(cl, [0, 1, 3, 4])
                Xg, yg, binary = Xg[keep], (cl[keep] >= 3).astype(float), True
            fn = _cv_logistic if binary else _cv_ridge
            res['decoding'][sname][vname] = fn(Xg, yg, folds, seed)
            res['decoding_null'][sname][vname] = fn(Xg, rng.permutation(yg), folds, seed)
            st = steady[rows][good] if vname != 'conflict' else steady[rows][good][keep]
            if st.sum() >= 5 * folds:
                res['decoding_steady'][sname][vname] = fn(Xg[st], yg[st], folds, seed)
        # Demixed variance: every variable at once, on the rows where all of them are finite.
        vnames = [v for v in ENCODING_VARS if v in variables]
        V = np.column_stack([np.asarray(variables[v][0], dtype=float) for v in vnames])
        good = rows & np.isfinite(V).all(axis=1)
        if good.sum() >= 50:
            res['variance'][sname] = demixed_variance(
                X[good], (vnames, [V[good][:, j] for j in range(V.shape[1])]),
                n_perm=n_perm if sname.startswith('hidden_t') else 0, seed=seed)
            res['variance'][sname]['n'] = int(good.sum())
    return res


def unit_classes(sess, mask=None, n_perm=100, seed=0):
    """CueS / CueL / Rule classification of every hidden unit, the paper's three classes.

    A unit's cue selectivity is fitted inside each context (where cue and rule coincide).
    Signs that agree across contexts mean the unit tracks the *cue*; signs that disagree
    mean it tracks the *rule*. The fraction of |selectivity| in the second half of the cue
    period splits the cue units into CueS (transient, < 0.5) and CueL (sustained). Units
    whose selectivity never exceeds a label-permutation null are unclassified.
    """
    h = _hidden(sess)
    m = steady_mask(sess) if mask is None else mask
    ts = _steps(CUE_PERIOD)
    sel = np.stack([_selectivity(h, m & (sess['context'] == c), sess['cue']) for c in (0, 1)])
    peak = np.abs(sel[:, ts]).max(axis=1)                    # (2, hidden)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        cue = sess['cue'].copy()
        idx = np.flatnonzero(m)
        cue[idx] = rng.permutation(cue[idx])
        s = np.stack([_selectivity(h, m & (sess['context'] == c), cue) for c in (0, 1)])
        null.append(np.abs(s[:, ts]).max(axis=1))
    thr = np.percentile(np.stack(null), 95, axis=0)          # (2, hidden)
    tuned = (peak > thr).all(axis=0)
    agree = np.sign(sel[0][ts].mean(axis=0)) == np.sign(sel[1][ts].mean(axis=0))
    late_frac = (np.abs(sel).mean(axis=0)[_steps(LATE_HALF)].sum(axis=0)
                 / np.maximum(np.abs(sel).mean(axis=0)[ts].sum(axis=0), 1e-12))
    label = np.full(h.shape[2], 'none', dtype=object)
    label[tuned & ~agree] = 'Rule'
    label[tuned & agree & (late_frac < 0.5)] = 'CueS'
    label[tuned & agree & (late_frac >= 0.5)] = 'CueL'
    counts = {k: int((label == k).sum()) for k in ('CueS', 'CueL', 'Rule', 'none')}
    return dict(label=label.tolist(), counts=counts, late_frac=late_frac.tolist(),
                tuned=int(tuned.sum()), n_units=int(h.shape[2]),
                frac_of_tuned={k: (counts[k] / max(int(tuned.sum()), 1))
                               for k in ('CueS', 'CueL', 'Rule')})


def _selectivity(h, mask, cue):
    """Mean difference between the two cues, per timestep and unit."""
    a, b = mask & (cue > 0), mask & (cue < 0)
    if a.sum() < 3 or b.sum() < 3:
        return np.zeros(h.shape[1:], dtype=np.float32)
    return h[a].mean(axis=0) - h[b].mean(axis=0)


def z_side_table(sess, upd=None, primary=PRIMARY):
    """The Z half of the same table: what the latent carries, against what X carries.

    Z is fixed within a trial, so single-trial conflict cannot be in it — the honest
    statement is that conflict lives in X (and in the *update* of Z), not in Z. This
    measures both: the context separation of z_in (d′ and decoding), and whether |Δz_err|
    carries the conflict of the trial that produced it.
    """
    m = select(sess, phase=primary, since=(STEADY, 10 ** 9))
    z = sess['z_in']
    res = dict(n=int(m.sum()))
    if sess.get('axis') is None:            # Z never moved (the RNN, or a clamped session)
        res.update(context_dprime=0.0, context_decoding=np.nan, conflict_from_z_r=np.nan,
                   note='Z is constant in this session: no context axis')
    else:
        p0 = sess['proj_in'][m & (sess['context'] == 0)]
        p1 = sess['proj_in'][m & (sess['context'] == 1)]
        sd = np.sqrt(0.5 * (np.nanvar(p0) + np.nanvar(p1)))
        res['context_dprime'] = float(abs(np.nanmean(p1) - np.nanmean(p0)) / max(sd, 1e-12))
        res['context_decoding'] = float(np.nanmean(sess['aligned'][m]))
        # Conflict from Z itself: nothing to find, by construction — reported, not hidden.
        contrast = np.nan_to_num(sess['contrast_in'][m])
        res['conflict_from_z_r'] = (float(np.corrcoef(contrast, sess['conflict'][m])[0, 1])
                                    if np.std(contrast) > 0 else np.nan)
    if upd is not None:
        ok = upd['ok'] & (sess['phase'] == primary)
        mag = np.linalg.norm(np.nan_to_num(upd['dz_err']), axis=1)
        for label, mm in (('error', ok & sess['err']), ('correct', ok & ~sess['err'])):
            if mm.sum() > 10:
                res[f'conflict_from_dz_{label}_r'] = float(
                    np.corrcoef(mag[mm], sess['conflict'][mm])[0, 1])
                res[f'dz_by_conflict_{label}'] = [
                    float(mag[mm & (sess['conf_level'] == c)].mean())
                    if (mm & (sess['conf_level'] == c)).any() else np.nan for c in range(5)]
    return res


def hidden_report(sess, primary=PRIMARY, n_perm=100, with_decoding=True, obs=None):
    """Every hidden-state number for one session, as one json-able dict.

    Conditions for the integration index and cue velocity: steady aligned trials (the
    rule-driven regime), the first 5 trials after a reversal (exploration), stale vs
    aligned, and each conflict level. `obs` (an ideal-observer result) is optional and only
    supplies the two uncertainty variables to the encoding table.
    """
    base = steady_mask(sess, primary)
    if base.sum() < 20:
        return dict(n_steady=int(base.sum()), note='too few steady-state trials to fit axes')
    axes = fit_axes(sess, base, per_context=True)
    res = dict(n_steady=int(base.sum()),
               angle_cue_rule=axes['angle_cue_rule'],
               cos_cue_across_contexts=axes.get('cos_cue_across_contexts'),
               beta_cue_norm=[float(np.linalg.norm(v)) for v in axes['beta_cue']],
               beta_rule_norm=[float(np.linalg.norm(v)) for v in axes['beta_rule']])
    m = select(sess, phase=primary)
    conds = {'steady_aligned': base,
             'early': m & (sess['since'] <= EARLY_N) & sess['reversal'],
             'aligned': m & sess['aligned'], 'stale': m & sess['stale']}
    conds.update({f'conf{c}': base & (sess['conf_level'] == c) for c in range(5)})
    res['integration'] = {k: integration(sess, axes, v) for k, v in conds.items()}
    res['unit_classes'] = unit_classes(sess, base, n_perm=n_perm)
    # The paper's Fig 3c cut: both population measures trial by trial around a reversal.
    res['integration_reversal'] = integration_aligned(sess, axes, primary=primary)
    if with_decoding:
        res['decoding'] = {k: decode(sess, mask=v) for k, v in
                           (('steady_aligned', base), ('early', conds['early']))}
        res['encoding'] = encoding_table(sess, obs, primary=primary)
    # C3: the two axes' magnitude around a reversal.
    from hier_switch_analyses import reversal_aligned
    h = _hidden(sess)
    ts = _steps(CUE_PERIOD)
    for name, ax in (('cue', axes['u_cue']), ('rule', axes['u_rule'])):
        sign = sess['cue'] if name == 'cue' else sess['rule']
        proj = (h[:, ts, :] @ ax).mean(axis=1) * sign
        ks, mu, sem, _ = reversal_aligned(sess, proj, primary=primary)
        res[f'{name}_axis_reversal'] = dict(k=ks.tolist(), mean=mu.tolist(), sem=sem.tolist())
    return res


def _self_test(seed=0):
    """Check the encoding measures on data whose answer is known.

    A source built as a known linear mix of two variables should have its variance credited
    to those two and not to the others, and a variable the source does not contain should
    decode at chance. Runs in a second; no model or session needed.
    """
    from hier_switch_analyses import synthetic_session, trial_labels
    sess, _ = synthetic_session(blen=40, n_blocks=12, seed=seed)
    trial_labels(sess)
    n = sess['n']
    rng = np.random.default_rng(seed)
    cue = (sess['cue'] > 0).astype(float)
    ctx = sess['context'].astype(float)
    # A 10-unit source that is 70% cue, 30% context, plus noise; and a 10-unit source that
    # is noise alone.
    mix = (0.7 * cue[:, None] * rng.normal(size=(1, 10))
           + 0.3 * ctx[:, None] * rng.normal(size=(1, 10))
           + 0.25 * rng.normal(size=(n, 10)))
    noise = rng.normal(size=(n, 10))
    variables = (['cue', 'context', 'conflict'],
                 [cue, ctx, sess['conflict'].astype(float)])
    dm = demixed_variance(mix, variables)
    dn = demixed_variance(noise, variables)
    u = dm['unique']
    ok = (u['cue'] > 0.2 and u['context'] > 0.02 and u['conflict'] < 0.01
          and max(dn['unique'].values()) < 0.01)
    print(f"{'OK ' if ok else 'FAIL'} demixed_variance credits the mix to cue "
          f"({u['cue']:.2f}) and context ({u['context']:.2f}), not conflict "
          f"({u['conflict']:.3f}); pure noise gets {max(dn['unique'].values()):.3f}")
    a_cue = _cv_logistic(mix, cue, seed=seed)
    a_shuf = _cv_logistic(mix, rng.permutation(cue), seed=seed)
    a_noise = _cv_logistic(noise, cue, seed=seed)
    ok2 = a_cue > 0.8 and abs(a_shuf - 0.5) < 0.08 and abs(a_noise - 0.5) < 0.08
    print(f"{'OK ' if ok2 else 'FAIL'} decoding: cue from the mix {a_cue:.2f}, "
          f"from shuffled labels {a_shuf:.2f}, from pure noise {a_noise:.2f}")
    return ok and ok2


if __name__ == '__main__':
    if not sys.argv[1:]:
        raise SystemExit(0 if _self_test() else 1)
    for path in sys.argv[1:]:
        sess = trial_labels(load_session(path))
        upd = z_updates(sess) if sess['meta']['lu_steps'] > 0 else None
        rep = hidden_report(sess)
        zt = z_side_table(sess, upd)
        print(f"\n{path}  ({sess['meta'].get('condition')}, {sess['meta']['model_type']})")
        print(f"  axes: cue-rule angle {rep['angle_cue_rule'][0]:.1f}°, cue axis across "
              f"contexts cos {rep['cos_cue_across_contexts']:+.2f}")
        print(f"  unit classes {rep['unit_classes']['counts']} of {rep['unit_classes']['n_units']}")
        for k, v in rep['integration'].items():
            if k in ('steady_aligned', 'early', 'stale', 'aligned'):
                print(f"  {k:15s} index {v['index']:.2f}  cue velocity {v['cue_velocity']:.3f}  n {v['n']}")
        for k, d in rep.get('decoding', {}).items():
            print(f"  decoding {k}: " + '  '.join(
                f"{t} {v['acc'][16]:.2f} (build {v['buildup']})" for t, v in d.items()))
        print(f"  Z side: context d′ {zt['context_dprime']:.2f}, "
              f"|Δz_err| vs conflict r (errors) {zt.get('conflict_from_dz_error_r', float('nan')):+.2f}")
