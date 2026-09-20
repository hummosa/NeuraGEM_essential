"""
hier_switch_group.py — every model × condition × seed, one report each, then the group table.

Single sessions are where an effect first shows up; the flanker project's lesson was that
single-session effects repeatedly failed to survive across seeds, so every number here is
reported per seed, with **how many seeds carry the predicted sign** next to the mean.

    ./hier_switch/run_sessions.sh                       # SLURM array: one task per model
    .venv/bin/python hier_switch/hier_switch_group.py list
    SLURM_ARRAY_TASK_ID=0 .venv/bin/python hier_switch/hier_switch_group.py task
    .venv/bin/python hier_switch/hier_switch_group.py analyse <session dir> ...
    .venv/bin/python hier_switch/hier_switch_group.py aggregate

`task` records one model's conditions (hier_switch_test_inference) and writes a
results.json beside each session. `analyse` does the second half alone, for sessions that
are already on disk. `aggregate` collects every results.json into
exports/hier_switch/group/group.json and prints the prediction table.

The models: the six v13/v15 NG seeds that discovered the contexts, and the ten v16 RNN
baselines (backprop only — no latent update, weights plastic at test, which is its only
route to adaptation).
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from hier_switch_analyses import (behaviour, latent, load_session, normative_table,
                                  trial_labels, z_update_table, z_updates)
from hier_switch_hidden import hidden_report, z_side_table
from hier_switch_observer import ideal_observer

EXPORTS = os.path.join(_ROOT, 'exports', 'hier_switch')
NG_SEEDS = (0, 1, 3, 5, 6, 9)            # the v13 seeds that discovered the contexts
RNN_SEEDS = tuple(range(10))
# Paired forced-conflict triplets are the backbone: same trials, only the first 5 trials of
# each block differ. The sigmoid ladder is where gain is a live axis (B5's Δgain).
NG_CONDITIONS = ['softmax_rc_none', 'softmax_rc_low', 'softmax_rc_high',
                 'sigmoid_zlr30000', 'sigmoid_zlr30000_wd1e-05', 'sigmoid_zlr30000_wd3e-05']
RNN_CONDITIONS = ['rnn_rc_none', 'rnn_rc_low', 'rnn_rc_high']


def models():
    """[(model_type, seed, model.pt, conditions)] — the group, in array-task order."""
    out = [('NG', s, os.path.join(EXPORTS, 'tune_v15', f'NG_s{s}', 'model.pt'), NG_CONDITIONS)
           for s in NG_SEEDS]
    out += [('RNN', s, os.path.join(EXPORTS, 'tune_v16', f'RNN_s{s}', 'model.pt'), RNN_CONDITIONS)
            for s in RNN_SEEDS]
    return out


def session_dirs(model_type, seed, conditions):
    tag = f"tune_{'v15' if model_type == 'NG' else 'v16'}_{model_type}_s{seed}"
    return [os.path.join(EXPORTS, 'inference_tests', tag, c) for c in conditions]


# ── One session ───────────────────────────────────────────────────────────────

def session_report(path, with_hidden=True):
    """Every phase-2 number for one recorded session, as a json-able dict."""
    sess = trial_labels(load_session(path))
    meta = sess['meta']
    has_lu = meta['lu_steps'] > 0
    obs = ideal_observer(sess)
    rep = dict(path=path, condition=meta.get('condition'), model_type=meta['model_type'],
               seed=meta.get('seed'), activation=meta['latent_activation'],
               reversal_conflict=meta.get('reversal_conflict'),
               Z_lr=meta['Z_lr'], Z_decay=meta['Z_decay'],
               z_lr_decay=float(meta['Z_lr'] * meta['Z_decay']))
    rep['behaviour'] = behaviour(sess)
    rep['observer'] = {k: v for k, v in obs.items()
                       if k in ('acc', 'acc_given_context', 'p_c', 'p_cue_mean')}
    rep['observer']['switch'] = float(np.nanmean(obs['switch_trial']))
    if has_lu:
        upd = z_updates(sess)
        rep['z_updates'] = {'__'.join(map(str, k)): v for k, v in
                            z_update_table(sess, upd, by=('state', 'err', 'conf_level')).items()}
        rep['z_updates_gain'] = {'__'.join(map(str, k)): v for k, v in
                                 z_update_table(sess, upd, by=('err', 'conf_level')).items()}
        rep['latent'] = latent(sess, obs, upd)
        rep['normative'] = normative_table(sess, upd, obs)
        rep['z_side'] = z_side_table(sess, upd)
    else:
        rep['z_side'] = z_side_table(sess)
    if with_hidden and 'hidden' in sess:
        rep['hidden'] = hidden_report(sess)
    return rep


def analyse(paths, with_hidden=True):
    for p in paths:
        rep = session_report(p, with_hidden=with_hidden)
        out = os.path.join(p, 'results.json')
        with open(out, 'w') as f:
            json.dump(rep, f, indent=1, default=_jsonable)
        b = rep['behaviour']
        print(f"{rep['model_type']} s{rep['seed']} {rep['condition']:26s} acc {b['acc']:.3f} "
              f"steady {b['acc_steady']:.3f} switch {b['switch']['all']['switch']:.2f} "
              f"→ {out}")


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return float(o)


# ── The group ─────────────────────────────────────────────────────────────────

def collect():
    """Every results.json on disk, grouped as {(model_type, condition): {seed: report}}."""
    out = {}
    for model_type, seed, _, conditions in models():
        for path in session_dirs(model_type, seed, conditions):
            f = os.path.join(path, 'results.json')
            if os.path.exists(f):
                with open(f) as fh:
                    out.setdefault((model_type, os.path.basename(path)), {})[seed] = json.load(fh)
    return out


def _per_seed(group, key, fn):
    """fn(report) per seed, as (seeds, values)."""
    seeds = sorted(group)
    vals = []
    for s in seeds:
        try:
            v = fn(group[s])
        except (KeyError, TypeError, IndexError):
            v = np.nan
        vals.append(np.nan if v is None else float(v))
    return np.array(seeds), np.array(vals)


CONFLICT_VALUES = np.array([0.0, 0.125, 2 / 7, 0.5, 0.8])


def _cell_slope(cells, state, err, key='s'):
    """Slope of a z_update_table quantity against conflict, weighted by the cells' trials."""
    xs, ys, ws = [], [], []
    for c in range(len(CONFLICT_VALUES)):
        cell = cells.get(f'{state}__{err}__{c}')
        if cell and np.isfinite(cell[key]):
            xs.append(CONFLICT_VALUES[c])
            ys.append(cell[key])
            ws.append(cell['n'])
    if len(xs) < 3:
        return np.nan
    w = np.asarray(ws, float)
    A = np.c_[np.asarray(xs), np.ones(len(xs))] * np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(A, np.asarray(ys) * np.sqrt(w), rcond=None)
    return float(beta[0])


def _sign_row(name, seeds, vals, predicted, note=''):
    ok = np.isfinite(vals)
    n_sign = int(np.sum(np.sign(vals[ok]) == np.sign(predicted))) if ok.any() else 0
    return dict(name=name, mean=float(np.nanmean(vals)) if ok.any() else np.nan,
                sem=float(np.nanstd(vals[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else np.nan,
                n_seeds=int(ok.sum()), n_predicted_sign=n_sign, predicted=predicted,
                per_seed=dict(zip(map(int, seeds), [None if not np.isfinite(v) else float(v)
                                                    for v in vals])), note=note)


def predictions(data):
    """P1-P7 as one table: per-seed values, the mean, and how many seeds carry the sign.

    Each row is a *difference* or a *slope* whose sign the prediction fixes, so
    n_predicted_sign / n_seeds is the headline number.
    """
    rows = []
    ng_none = data.get(('NG', 'softmax_rc_none'), {})
    ng_low = data.get(('NG', 'softmax_rc_low'), {})
    ng_high = data.get(('NG', 'softmax_rc_high'), {})
    rnn_none = data.get(('RNN', 'rnn_rc_none'), {})

    if ng_none:
        seeds, v = _per_seed(ng_none, None, lambda r: r['behaviour']['psychometric']['acc'][0]
                             - r['behaviour']['psychometric']['acc'][4])
        rows.append(_sign_row('P1 accuracy falls with conflict (level 0 − level 4)', seeds, v, +1))
        seeds, v = _per_seed(ng_none, None, lambda r: abs(
            np.nanmean(r['behaviour']['psychometric']['acc_ctx'][0])
            - np.nanmean(r['behaviour']['psychometric']['acc_ctx'][1])))
        rows.append(_sign_row('P1 |context A − context B| accuracy (should be ~0)', seeds, v, 0,
                              'no sign predicted; report the size'))

    # P2: the paired forced-conflict sessions, three switch criteria + the observer's
    for key, label in (('switch', 'behaviour'), ('z_switch', 'Z side'), ('dec_switch', 'decided')):
        if ng_low and ng_high:
            seeds = sorted(set(ng_low) & set(ng_high))
            v = np.array([ng_high[s]['behaviour']['switch']['all'][key]
                          - ng_low[s]['behaviour']['switch']['all'][key] for s in seeds])
            rows.append(_sign_row(f'P2 switch latency, high − low early conflict ({label})',
                                  np.array(seeds), v, +1))
    if ng_low and ng_high:
        seeds = sorted(set(ng_low) & set(ng_high))
        v = np.array([ng_high[s]['observer']['switch'] - ng_low[s]['observer']['switch'] for s in seeds])
        rows.append(_sign_row('P2 the same for the ideal observer (the normative size)',
                              np.array(seeds), v, +1))

    if ng_none:
        seeds, v = _per_seed(ng_none, None,
                             lambda r: r['latent']['uncertainty']['all']['peak']
                             - r['latent']['uncertainty']['all']['baseline'])
        rows.append(_sign_row('P3 Z uncertainty rises after a reversal (peak − baseline)', seeds, v, +1))
    if ng_low and ng_high:
        seeds = sorted(set(ng_low) & set(ng_high))
        v = np.array([ng_high[s]['latent']['uncertainty']['all']['width_half']
                      - ng_low[s]['latent']['uncertainty']['all']['width_half'] for s in seeds])
        rows.append(_sign_row('P3 uncertainty lasts longer after high-conflict reversals '
                              '(width, high − low)', np.array(seeds), v, +1))

    if ng_none:
        seeds, v = _per_seed(ng_none, None, lambda r: r['latent']['eps_cw']['r_window'])
        rows.append(_sign_row('P4 |dL/dZ| over 5 trials tracks ε_CW (r)', seeds, v, +1))
        seeds, v = _per_seed(ng_none, None, lambda r: r['normative']['r'])
        rows.append(_sign_row('P4/B5 mean update s against the observer’s Δlogit (r)', seeds, v, +1))
        seeds, v = _per_seed(ng_none, None, lambda r: r['z_updates']['1__1__0']['s']
                             - r['z_updates']['1__1__4']['s'])
        rows.append(_sign_row('B5 stale-error update falls with conflict (level 0 − level 4)',
                              seeds, v, +1))
        seeds, v = _per_seed(ng_none, None, lambda r: r['z_updates']['1__1__0']['tipped']
                             - r['z_updates']['1__1__4']['tipped'])
        rows.append(_sign_row('B5 P(tipped | error) falls with conflict', seeds, v, +1))
        # The endpoint difference above rests on the two thinnest cells (a stale error on an
        # unambiguous cue is rare once the model switches fast). These two rows use all the
        # trials instead: the slope across the five levels, and whether adding the cue's
        # reliability to a plain error term explains more of the update.
        seeds, v = _per_seed(ng_none, None, lambda r: _cell_slope(r['z_updates'], state=1, err=1))
        rows.append(_sign_row('B5 stale-error update against conflict: slope over all 5 levels',
                              seeds, v, -1))
        # Conflict weighting is a *reduction* of the error's effect, so the test is the
        # interaction term of s ~ err + err × conflict, not a regressor that multiplies the
        # error by the cue's log-odds (heavy-tailed, and the relation saturates, so a linear
        # fit on it can do worse than a plain error dummy — it does, for 5 of 6 seeds).
        seeds, v = _per_seed(ng_none, None,
                             lambda r: r['latent']['update_rule']['err_plus_interaction']['beta'][1])
        rows.append(_sign_row('B3 error × conflict interaction on the update (negative = '
                              'conflict weighting)', seeds, v, -1))
        seeds, v = _per_seed(ng_none, None,
                             lambda r: r['latent']['update_rule']['err_plus_interaction']['r2']
                             - r['latent']['update_rule']['err']['r2'])
        rows.append(_sign_row('B3 that interaction adds explanatory power (ΔR² over error alone)',
                              seeds, v, +1))

        seeds, v = _per_seed(ng_none, None,
                             lambda r: r['hidden']['decoding']['steady_aligned']['rule']['acc'][16]
                             - r['hidden']['decoding']['early']['rule']['acc'][16])
        rows.append(_sign_row('P5 rule decoding drops after a reversal (steady − early)', seeds, v, +1))
        seeds, v = _per_seed(ng_none, None,
                             lambda r: r['hidden']['decoding']['steady_aligned']['cue']['acc'][16]
                             - r['hidden']['decoding']['early']['cue']['acc'][16])
        rows.append(_sign_row('P5 cue decoding holds after a reversal (steady − early, ~0)',
                              seeds, v, 0, 'predicted ~0, smaller than the rule drop'))
        seeds, v = _per_seed(ng_none, None, lambda r: r['hidden']['cos_cue_across_contexts'])
        rows.append(_sign_row('P5 cue axis agrees across contexts (cos)', seeds, v, +1))

        seeds, v = _per_seed(ng_none, None, lambda r: r['behaviour']['psychometric']['rt'][4]
                             - r['behaviour']['psychometric']['rt'][0])
        rows.append(_sign_row('P6 RT rises with conflict (level 4 − level 0)', seeds, v, +1))
        seeds, v = _per_seed(ng_none, None, lambda r: max(r['behaviour']['rt']['by_since'][2:5])
                             - r['behaviour']['rt']['by_since'][0])
        rows.append(_sign_row('P6 reversal RT peaks after the perseverative trials '
                              '(peak k3-5 − k1)', seeds, v, +1))
        seeds, v = _per_seed(ng_none, None, lambda r: r['behaviour']['rt']['err_steady']
                             - r['behaviour']['rt']['err_early'])
        rows.append(_sign_row('P6 early errors are faster than steady-state errors', seeds, v, +1))

        seeds, v = _per_seed(ng_none, None, lambda r: r['hidden']['integration']['steady_aligned']['index']
                             - r['hidden']['integration']['early']['index'])
        rows.append(_sign_row('Integration index is lower in exploration (steady − early)',
                              seeds, v, +1, "the paper's direction"))
        seeds, v = _per_seed(ng_none, None, lambda r: r['hidden']['integration']['early']['cue_velocity']
                             - r['hidden']['integration']['steady_aligned']['cue_velocity'])
        rows.append(_sign_row('Cue velocity is higher in exploration (early − steady)',
                              seeds, v, +1, "the paper's direction"))

    if ng_none and rnn_none:
        for key, name in ((('z_side', 'context_dprime'), 'context separation in Z (d′)'),):
            s1, v1 = _per_seed(ng_none, None, lambda r: r[key[0]][key[1]])
            s2, v2 = _per_seed(rnn_none, None, lambda r: r[key[0]][key[1]])
            rows.append(dict(name=f'NG vs RNN: {name}', mean=float(np.nanmean(v1)),
                             rnn_mean=float(np.nanmean(v2)), n_seeds=int(np.isfinite(v1).sum()),
                             n_rnn_seeds=int(np.isfinite(v2).sum()), predicted=None,
                             per_seed=dict(zip(map(int, s1), v1.tolist())),
                             rnn_per_seed=dict(zip(map(int, s2), v2.tolist())), note=''))
        for name, fn in (('accuracy (steady)', lambda r: r['behaviour']['acc_steady']),
                         ('rule decoding at t=16', lambda r: r['hidden']['decoding']['steady_aligned']['rule']['acc'][16]),
                         ('cue decoding at t=16', lambda r: r['hidden']['decoding']['steady_aligned']['cue']['acc'][16]),
                         ('switch latency', lambda r: r['behaviour']['switch']['all']['switch'])):
            s1, v1 = _per_seed(ng_none, None, fn)
            s2, v2 = _per_seed(rnn_none, None, fn)
            rows.append(dict(name=f'NG vs RNN: {name}', mean=float(np.nanmean(v1)),
                             rnn_mean=float(np.nanmean(v2)), n_seeds=int(np.isfinite(v1).sum()),
                             n_rnn_seeds=int(np.isfinite(v2).sum()), predicted=None,
                             per_seed=dict(zip(map(int, s1), v1.tolist())),
                             rnn_per_seed=dict(zip(map(int, s2), v2.tolist())), note=''))

    # The gain axis: only live without the softmax.
    for cond in ('sigmoid_zlr30000', 'sigmoid_zlr30000_wd1e-05', 'sigmoid_zlr30000_wd3e-05'):
        g = data.get(('NG', cond), {})
        if not g:
            continue
        seeds, v = _per_seed(g, None, lambda r: r['z_updates_gain']['1__4']['dgain']
                             - r['z_updates_gain']['0__4']['dgain'])
        rows.append(_sign_row(f'Gain ({cond}): errors move the gain down relative to correct '
                              'trials (high conflict)', seeds, v, -1))
    sm = data.get(('NG', 'softmax_rc_none'), {})
    if sm:
        seeds, v = _per_seed(sm, None, lambda r: max(abs(c['dgain']) for c in r['z_updates'].values()))
        rows.append(_sign_row('Softmax: |Δgain| is identically 0 (max over cells)', seeds, v, 0,
                              'the check that the softmax has no gain direction'))
    return rows


def aggregate(out_dir=None):
    data = collect()
    rows = predictions(data)
    out_dir = out_dir or os.path.join(EXPORTS, 'group')
    os.makedirs(out_dir, exist_ok=True)
    summary = dict(sessions={f'{k[0]}/{k[1]}': sorted(v) for k, v in data.items()},
                   predictions=rows)
    with open(os.path.join(out_dir, 'group.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=_jsonable)
    print(f'{"":<62} {"mean":>8} {"sem":>7} {"seeds w/ sign":>14}')
    for r in rows:
        if r.get('predicted') is None:
            print(f"{r['name']:<62} {r['mean']:8.3f} {'':>7}   RNN {r['rnn_mean']:.3f} "
                  f"(n {r['n_seeds']} / {r['n_rnn_seeds']})")
        else:
            sign = '' if r['predicted'] == 0 else f"{r['n_predicted_sign']} / {r['n_seeds']}"
            print(f"{r['name']:<62} {r['mean']:8.3f} {r['sem']:7.3f} {sign:>14}"
                  + (f"   [{r['note']}]" if r['note'] else ''))
    print(f"\nWrote {os.path.join(out_dir, 'group.json')}")
    return summary


def task(index=None):
    """One SLURM array task: record a model's conditions, then analyse them."""
    from hier_switch_test_inference import main as record
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0) if index is None else index)
    model_type, seed, path, conditions = models()[index]
    if not os.path.exists(path):
        raise SystemExit(f'no model at {path}')
    print(f'--- task {index}: {model_type} seed {seed}: {", ".join(conditions)}')
    record(path, conditions)
    analyse(session_dirs(model_type, seed, conditions))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'aggregate'
    if mode == 'list':
        for i, (mt, sd, p, c) in enumerate(models()):
            print(i, mt, sd, 'model' if os.path.exists(p) else 'MISSING', ' '.join(c))
    elif mode == 'task':
        task(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif mode == 'analyse':
        analyse(sys.argv[2:])
    elif mode == 'aggregate':
        aggregate()
    else:
        raise SystemExit(__doc__)
