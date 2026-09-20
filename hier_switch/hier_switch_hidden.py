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


def hidden_report(sess, primary=PRIMARY, n_perm=100, with_decoding=True):
    """Every hidden-state number for one session, as one json-able dict.

    Conditions for the integration index and cue velocity: steady aligned trials (the
    rule-driven regime), the first 5 trials after a reversal (exploration), stale vs
    aligned, and each conflict level.
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
    if with_decoding:
        res['decoding'] = {k: decode(sess, mask=v) for k, v in
                           (('steady_aligned', base), ('early', conds['early']))}
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


if __name__ == '__main__':
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
