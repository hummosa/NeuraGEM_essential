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
    HIER_SWITCH_PROBE=1 ./hier_switch/run_sessions.sh   # the noise-0.6 probe, then:
    .venv/bin/python hier_switch/hier_switch_group.py probe

`task` records one model's conditions (hier_switch_test_inference) and writes a
results.json beside each session. `analyse` does the second half alone, for sessions that
are already on disk. `aggregate` collects every results.json into
exports/hier_switch/group/group.json and prints the prediction table.

The models: the six v13/v15 NG seeds that discovered the contexts, and the ten v17 RNN
baselines (backprop only — no latent update, weights plastic at test, which is its only
route to adaptation). The RNN is tested on 250-350-trial blocks at WU_lr 3e-3, the shortest
blocks a plastic RNN re-learns inside; on the paper's 30-60 it hedges at every rate (the v16
sessions, `rnn_rc_*`, are that record). Its steady state and reversal window scale with its
blocks (hier_switch_analyses.block_scale).
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
                                  switch_by_early_conf, trial_labels, z_update_table,
                                  z_updates)
from hier_switch_hidden import hidden_report, z_side_table
from hier_switch_observer import ideal_observer

EXPORTS = os.path.join(_ROOT, 'exports', 'hier_switch')

#: One cue-noise level = a pair of tune tags, a seed range and an output directory.
#:
#: Nothing physical is namespaced by this table. The tune tag is what keeps the levels
#: apart on disk, and it does so for free: it is derived from the model's own path
#: (hier_switch_test_inference.main, hier_switch_perturb.clamp_grid), so tune_v18 models
#: write to inference_tests/tune_v18_NG_s*/ and clamp/tune_v18_NG_s*/ without any of the
#: writers knowing a level exists. That matters because the alternative — a root that a
#: forgotten environment variable silently drops — would write real GPU-hours of sessions
#: into the wrong tree and then skip them as already recorded.
#:
#: HIER_SWITCH_LEVEL therefore only chooses which tags to *read* and where the group table
#: and figures land. Getting it wrong costs a few minutes of regeneration and is obvious
#: on sight (a six-seed table with twelve seeds in it).
LEVELS = {
    'n05': dict(noise=0.5, ng_tag='v15', rnn_tag='v17', rnn_seeds=range(10), out=('group',)),
    'n06': dict(noise=0.6, ng_tag='v18', rnn_tag='v19', rnn_seeds=range(20), out=('group', 'n06')),
}
LEVEL = os.environ.get('HIER_SWITCH_LEVEL') or 'n05'
if LEVEL not in LEVELS:
    raise SystemExit(f'HIER_SWITCH_LEVEL={LEVEL!r}; known levels: {sorted(LEVELS)}')
NOISE = LEVELS[LEVEL]['noise']
NG_TAG = LEVELS[LEVEL]['ng_tag']
RNN_TAG = LEVELS[LEVEL]['rnn_tag']
OUT_DIR = os.path.join(EXPORTS, *LEVELS[LEVEL]['out'])


def ng_seeds():
    """The seeds that discovered the contexts at this level, as `arms --save` recorded them.

    Hand-transcribing the list off `arms` into the source is the single most likely thing
    to go wrong once there is more than one level to do it for, so selection.json is the
    record and this reads it. The noise-0.5 tuple is kept as a fallback because tune_v15
    predates selection.json and its six seeds are quoted throughout the docs.
    """
    f = os.path.join(EXPORTS, f'tune_{NG_TAG}', 'selection.json')
    if os.path.exists(f):
        with open(f) as fh:
            return tuple(json.load(fh)['ng_seeds'])
    if NG_TAG == 'v15':
        return (0, 1, 3, 5, 6, 9)
    raise SystemExit(f'no {f}: run `hier_switch_tune.py arms {NG_TAG} --save` first')


NG_SEEDS = ng_seeds()
RNN_SEEDS = tuple(LEVELS[LEVEL]['rnn_seeds'])
# Paired forced-conflict triplets are the backbone: same trials, only the first 5 trials of
# each block differ. The sigmoid ladder is where gain is a live axis (B5's Δgain).
NG_CONDITIONS = ['softmax_rc_none', 'softmax_rc_low', 'softmax_rc_high',
                 'sigmoid_zlr30000', 'sigmoid_zlr30000_wd1e-05', 'sigmoid_zlr30000_wd3e-05']
#: The manipulation sessions (hier_switch_hooks): the paper's optogenetics, plus the
#: sigmoid triplet the gain panels need. Recorded by the same array; kept separate so a
#: re-run of the originals does not drag these along and vice versa.
NG_MANIPULATIONS = ['sigmoid_rc_none', 'sigmoid_rc_low', 'sigmoid_rc_high',
                    'softmax_lu0_rc_low', 'softmax_lu0_rc_high',
                    'softmax_lu3_rc_low', 'softmax_lu3_rc_high',
                    'softmax_lu10_rc_low', 'softmax_lu10_rc_high',
                    'softmax_blast_rc_low', 'softmax_blast_rc_high',
                    'sigmoid_blast_rc_low', 'sigmoid_blast_rc_high']
#: The RNN baseline: v17 models on their own blocks (hier_switch_test_inference.RNN300).
#: The v16 sessions on the paper's blocks (`rnn_rc_*`) stay on disk as the record of the hedge.
RNN_CONDITIONS = ['rnn300_rc_none', 'rnn300_rc_low', 'rnn300_rc_high']
RNN_BASE = RNN_CONDITIONS[0]
#: The noise-0.6 probe: the forced triplet re-run on the *already trained* models with a
#: noisier cue. It asks whether the inference mechanism survives higher sensory noise at
#: fixed weights, which is not the question tune_v18 asks, so it is kept out of
#: `all_conditions` and never reaches the group table or the figures. Set
#: HIER_SWITCH_PROBE=1 to record it; read it back with `probe`.
NG_PROBE = ['softmax_n06_rc_none', 'softmax_n06_rc_low', 'softmax_n06_rc_high']
#: Off until the manipulation sessions have been recorded, so `list` and `task` keep
#: describing what is on disk. Set HIER_SWITCH_MANIP=1 to include them.
WITH_MANIPULATIONS = bool(os.environ.get('HIER_SWITCH_MANIP'))
WITH_PROBE = bool(os.environ.get('HIER_SWITCH_PROBE'))


def models():
    """[(model_type, seed, model.pt, conditions)] — the group, in array-task order."""
    conds = NG_CONDITIONS + (NG_MANIPULATIONS if WITH_MANIPULATIONS else [])
    # The probe replaces the list rather than extending it: it is a one-off diagnostic on
    # models that already have their sessions, so re-recording those would only be skipped.
    out = [('NG', s, os.path.join(EXPORTS, f'tune_{NG_TAG}', f'NG_s{s}', 'model.pt'),
            NG_PROBE if WITH_PROBE else conds) for s in NG_SEEDS]
    if WITH_PROBE:
        return out                      # the RNN has no latent inference to probe
    out += [('RNN', s, os.path.join(EXPORTS, f'tune_{RNN_TAG}', f'RNN_s{s}', 'model.pt'),
             RNN_CONDITIONS) for s in RNN_SEEDS]
    return out


def session_dirs(model_type, seed, conditions):
    tag = f"tune_{NG_TAG if model_type == 'NG' else RNN_TAG}_{model_type}_s{seed}"
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
               # The noise level this session ran at. Recorded so `aggregate` can prove a
               # group table is built from one level; the levels are otherwise kept apart
               # by the tune tag alone, which is a naming convention and not a check.
               pulse_noise_std=meta['pulse_noise_std'],
               Z_lr=meta['Z_lr'], Z_decay=meta['Z_decay'],
               z_lr_decay=float(meta['Z_lr'] * meta['Z_decay']))
    rep['behaviour'] = behaviour(sess)
    # The paper's Fig 1f on this session's own, uncontrolled reversals: latency against
    # how ambiguous the block's first five trials happened to be.
    rep['behaviour']['by_early_conf'] = switch_by_early_conf(sess, obs)
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
        rep['hidden'] = hidden_report(sess, obs=obs)
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

def all_conditions(model_type):
    """Every condition worth *reading* for a model type.

    Deliberately not `models()`: that one is the array's work list and hides the
    manipulation conditions behind HIER_SWITCH_MANIP so `list` and `task` describe what is
    meant to be recorded. Reading is different — a results.json on disk should be found
    whether or not the flag that recorded it is set now, or the figures would silently drop
    a row depending on an environment variable.
    """
    return (NG_CONDITIONS + NG_MANIPULATIONS) if model_type == 'NG' else RNN_CONDITIONS


def collect():
    """Every results.json on disk, grouped as {(model_type, condition): {seed: report}}."""
    out = {}
    for model_type, seed, _, _ in models():
        for path in session_dirs(model_type, seed, all_conditions(model_type)):
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
    rnn_none = data.get(('RNN', RNN_BASE), {})

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
        # The RNN's steady state and switch latency are on its own 250-350-trial blocks
        # (block_scale); NG's are on the paper's 30-60.
        for name, fn in (('accuracy (steady)', lambda r: r['behaviour']['acc_steady']),
                         ('undecided rate', lambda r: r['behaviour']['undecided']),
                         ('rule decoding at t=16', lambda r: r['hidden']['decoding']['steady_aligned']['rule']['acc'][16]),
                         ('cue decoding at t=16', lambda r: r['hidden']['decoding']['steady_aligned']['cue']['acc'][16]),
                         ('context decoding at t=16', lambda r: r['hidden']['decoding']['steady_aligned']['context']['acc'][16]),
                         ('switch latency', lambda r: r['behaviour']['switch']['all']['switch']),
                         ('switch latency (decided)', lambda r: r['behaviour']['switch']['all']['dec_switch'])):
            s1, v1 = _per_seed(ng_none, None, fn)
            s2, v2 = _per_seed(rnn_none, None, fn)
            rows.append(dict(name=f'NG vs RNN: {name}', mean=float(np.nanmean(v1)),
                             rnn_mean=float(np.nanmean(v2)), n_seeds=int(np.isfinite(v1).sum()),
                             n_rnn_seeds=int(np.isfinite(v2).sum()), predicted=None,
                             per_seed=dict(zip(map(int, s1), v1.tolist())),
                             rnn_per_seed=dict(zip(map(int, s2), v2.tolist())), note=''))

    # The RNN's own Fig 1f: its forced low/high pair, on its own blocks.
    rnn_low = data.get(('RNN', 'rnn300_rc_low'), {})
    rnn_high = data.get(('RNN', 'rnn300_rc_high'), {})
    if rnn_low and rnn_high:
        seeds = sorted(set(rnn_low) & set(rnn_high))
        for key, label in (('switch', 'behaviour'), ('dec_switch', 'decided')):
            v = np.array([rnn_high[s]['behaviour']['switch']['all'][key]
                          - rnn_low[s]['behaviour']['switch']['all'][key] for s in seeds])
            rows.append(_sign_row(f'RNN: switch latency, high − low early conflict ({label})',
                                  np.array(seeds), v, +1, 'on its own 250-350-trial blocks'))

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

    rows += _encoding_rows(ng_none)
    rows += _regime_rows(ng_none)
    rows += _manipulation_rows(data)
    return rows


#: Which (source, variable) cells of the encoding table get a row, and what we expect.
#: +1 = above chance, 0 = at chance (reported as a size, not a sign). The expectations are
#: written down so the table says which ones did *not* come out that way.
_ENCODING_EXPECT = (
    ('hidden_t16', 'context', +1, 'the hidden state is gated by Z, so this is near ceiling'),
    ('hidden_pc2', 'context', 0, 'dimension-matched control: is context in the dominant PCs?'),
    ('hidden_t16', 'cue', +1, ''),
    ('hidden_t16', 'rule', +1, ''),
    ('hidden_t16', 'conflict', +1, ''),
    ('z_in', 'context', +1, 'the MDContext analogue'),
    ('z_in', 'cue', 0, 'Z should carry the context and nothing of the current trial'),
    ('z_in', 'conflict', 0, ''),
    ('grad', 'context', +1, 'the error signal is signed by which context was wrong'),
    ('grad', 'outcome', +1, ''),
    ('grad', 'cue', 0, 'along the context axis the gradient cannot carry the cue'),
    ('grad', 'rule', 0, ''),
)


def _encoding_rows(ng_none):
    """What each signal encodes, decoding accuracy minus its own shuffled-label null."""
    rows = []
    if not ng_none:
        return rows
    for src, var, expect, note in _ENCODING_EXPECT:
        seeds, v = _per_seed(ng_none, None, lambda r, s=src, x=var: (
            r['hidden']['encoding']['decoding'][s][x]
            - r['hidden']['encoding']['decoding_null'][s][x]))
        if expect == 0:
            note = (note + '; ' if note else '') + 'expected at chance: report the size'
        rows.append(_sign_row(f'Encoding: {var} from {src} (above its shuffle null)',
                              seeds, v, expect, note))
    # Mixed against demixed: how many variables a unit of each signal carries.
    seeds, v = _per_seed(ng_none, None,
                         lambda r: r['hidden']['encoding']['variance']['hidden_t16']['mean_vars_per_unit'])
    rows.append(_sign_row('Encoding: variables per hidden unit (the paper\'s Fig 2k, PFC side)',
                          seeds, v, +1, 'a count, not a difference; report the size'))
    seeds, v = _per_seed(ng_none, None,
                         lambda r: (r['hidden']['encoding']['variance']['z_in']['unique']['context']
                                    - max(r['hidden']['encoding']['variance']['z_in']['unique'][k]
                                          for k in ('cue', 'rule', 'conflict', 'outcome'))))
    rows.append(_sign_row('Encoding: Z is demixed (context variance − the best other variable)',
                          seeds, v, +1))
    return rows


def _regime_rows(ng_none):
    """The paper's Fig 3c: does the population shift regime right after a reversal?"""
    rows = []
    if not ng_none:
        return rows

    def early_minus_steady(r, key):
        ir = r['hidden'].get('integration_reversal')
        if not ir:
            return np.nan
        k = np.asarray(ir['k'], dtype=float)
        v = np.asarray(ir[key], dtype=float)
        early = np.nanmean(v[(k >= 1) & (k <= 5)])
        steady = np.nanmean(v[k >= 11]) if (k >= 11).any() else np.nanmean(v[k <= 0])
        return early - steady
    seeds, v = _per_seed(ng_none, None, lambda r: early_minus_steady(r, 'index'))
    rows.append(_sign_row('Regime: integration index, first 5 after a reversal − steady',
                          seeds, v, -1, 'the paper: integration collapses in exploration'))
    seeds, v = _per_seed(ng_none, None, lambda r: early_minus_steady(r, 'cue_velocity'))
    rows.append(_sign_row('Regime: cue velocity, first 5 after a reversal − steady',
                          seeds, v, +1, 'the paper: activity becomes input-driven'))
    return rows


#: Each manipulation against its own unperturbed control, on the matching forced session.
_MANIPULATIONS = (('softmax_lu0', 'softmax', 'latent update off, trials 1-4 (ACC→MD silencing)', +1),
                  ('softmax_lu3', 'softmax', 'latent update x3, trials 1-5 (ours, not the paper)', -1),
                  ('softmax_lu10', 'softmax', 'latent update x10, trials 1-5 (ours)', -1),
                  ('softmax_blast', 'softmax', 'Z driven to (1, 1) at the first feedback', -1),
                  ('sigmoid_blast', 'sigmoid', 'Z driven to (1, 1), sigmoid gate', -1))


def _manipulation_rows(data, criteria=(('dec_switch', 'decided'), ('z_switch', 'Z side'))):
    """Switch latency under each manipulation minus its control, within seed and split.

    The sign convention is the paper's: positive = slower to switch. Silencing the update is
    predicted to slow switching; driving the latent is predicted to speed it up.
    """
    rows = []
    for name, base, label, expect in _MANIPULATIONS:
        for split in ('low', 'high'):
            pert = data.get(('NG', f'{name}_rc_{split}'), {})
            ctrl = data.get(('NG', f'{base}_rc_{split}'), {})
            if not pert or not ctrl:
                continue
            seeds = sorted(set(pert) & set(ctrl))
            for key, crit in criteria:
                v = np.array([pert[s]['behaviour']['switch']['all'][key]
                              - ctrl[s]['behaviour']['switch']['all'][key] for s in seeds])
                rows.append(_sign_row(
                    f'Manipulation [{split} conflict]: {label} — extra trials to switch ({crit})',
                    np.array(seeds), v, expect))
    return rows


def noise_level(data):
    """The one `pulse_noise_std` every collected session ran at, or None if none says.

    Reports written before the field existed do not carry it; those are the noise-0.5
    sessions and they are simply not counted. Two *recorded* levels in one table is a
    mixed group and is refused here rather than averaged.
    """
    seen = sorted({r['pulse_noise_std'] for d in data.values() for r in d.values()
                   if r.get('pulse_noise_std') is not None})
    if len(seen) > 1:
        raise SystemExit(f'sessions from more than one noise level in one group: {seen}. '
                         'Set HIER_SWITCH_LEVEL, or check the tune tag in models().')
    return seen[0] if seen else None


def aggregate(out_dir=None):
    data = collect()
    rows = predictions(data)
    noise = noise_level(data)
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    summary = dict(sessions={f'{k[0]}/{k[1]}': sorted(v) for k, v in data.items()},
                   predictions=rows)
    # Only when some session recorded it, so a table built entirely from sessions that
    # predate the field stays byte-identical to the one already on disk.
    if noise is not None:
        summary['pulse_noise_std'] = noise
    with open(os.path.join(out_dir, 'group.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=_jsonable)
    print(f'pulse noise: {noise if noise is not None else "not recorded (pre-0.6 sessions)"}')
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


#: The probe's measures: a label, and how to pull the number out of one seed's three
#: reports (keyed 'none' / 'low' / 'high'). Conflict 0.286 and 0.5 are the paper's low and
#: high levels; the switch cost is the story figure's panel d in one number.
PROBE_MEASURES = [
    ('accuracy, conflict 0.29', lambda d: d['none']['behaviour']['psychometric']['acc'][2]),
    ('accuracy, conflict 0.50', lambda d: d['none']['behaviour']['psychometric']['acc'][3]),
    ('steady accuracy', lambda d: d['none']['behaviour']['acc_steady']),
    ('trials to switch, low early conflict', lambda d: d['low']['behaviour']['switch']['all']['switch']),
    ('trials to switch, high early conflict', lambda d: d['high']['behaviour']['switch']['all']['switch']),
    ('switch cost (high - low)', lambda d: (d['high']['behaviour']['switch']['all']['switch']
                                            - d['low']['behaviour']['switch']['all']['switch'])),
]


#: The probe is *about* the noise-0.5 networks, so it names them rather than following
#: HIER_SWITCH_LEVEL — reading it under n06 should still show the n05 models.
PROBE_TAG = LEVELS['n05']['ng_tag']
PROBE_SEEDS = (0, 1, 3, 5, 6, 9)


def _probe_reports(seed, prefix):
    """One seed's three forced sessions under a condition prefix, or None if incomplete."""
    out = {}
    for k in ('none', 'low', 'high'):
        f = os.path.join(EXPORTS, 'inference_tests', f'tune_{PROBE_TAG}_NG_s{seed}',
                         f'{prefix}_rc_{k}', 'results.json')
        if not os.path.exists(f):
            return None
        with open(f) as fh:
            out[k] = json.load(fh)
    return out


def probe():
    """The noise-0.6 probe beside its noise-0.5 partner, on the same trained networks.

    Same weights, same seeds, and a stimulus stream whose noise vector is the 0.5 one
    scaled by 1.2, so every row is a within-network difference — a much tighter comparison
    than the cross-level one the two tune tags support. What it does **not** answer is
    whether a network *grown* under the higher noise behaves this way; that is tune_v18.
    """
    pairs = [(s, lo, hi) for s in PROBE_SEEDS
             for lo, hi in [(_probe_reports(s, 'softmax'), _probe_reports(s, 'softmax_n06'))]
             if lo and hi]
    if not pairs:
        raise SystemExit('no paired probe sessions: record them with '
                         'HIER_SWITCH_PROBE=1 ./hier_switch/run_sessions.sh')
    seeds = [s for s, _, _ in pairs]
    print(f'noise-0.6 probe on the trained noise-0.5 networks: seeds {seeds}')
    print('same weights, same stream; only the cue noise differs (paired within network)\n')
    print(f'{"measure":<38} {"s=0.5":>8} {"s=0.6":>8} {"change":>8} {"sem":>7} {"seeds":>7}')
    rows = []
    for name, fn in PROBE_MEASURES:
        a = np.array([fn(lo) for _, lo, _ in pairs], dtype=float)
        b = np.array([fn(hi) for _, _, hi in pairs], dtype=float)
        d = b - a
        ok = np.isfinite(d)
        sem = float(np.nanstd(d[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else np.nan
        same = int(np.sum(np.sign(d[ok]) == np.sign(np.nanmean(d)))) if ok.any() else 0
        print(f'{name:<38} {np.nanmean(a):8.3f} {np.nanmean(b):8.3f} {np.nanmean(d):+8.3f} '
              f'{sem:7.3f} {f"{same} / {int(ok.sum())}":>7}')
        rows.append(dict(name=name, mean_n05=float(np.nanmean(a)), mean_n06=float(np.nanmean(b)),
                         change=float(np.nanmean(d)), sem=sem, n_seeds=int(ok.sum()),
                         n_same_sign=same,
                         per_seed=dict(zip(map(int, seeds), map(float, d)))))
    return rows


def check_level(path):
    """Refuse a model trained at a different cue noise than this level claims.

    The tune tag keeps the levels apart on disk, but a tag is a naming convention: one
    wrong literal and a noise-0.5 model runs a whole session while sitting in the
    noise-0.6 tree, with nothing to show for it afterwards. The model carries its own
    training config, so the check is free. Skipped for the probe, whose whole point is to
    run a noise-0.5 model at a different noise.
    """
    from hier_switch_train import load_model
    sigma = float(load_model(path)[1].pulse_noise_std)
    if not WITH_PROBE and abs(sigma - NOISE) > 1e-9:
        raise SystemExit(f'level {LEVEL} wants pulse_noise_std {NOISE}, but {path} was '
                         f'trained at {sigma}. Check HIER_SWITCH_LEVEL and the tune tag.')
    return sigma


def task(index=None):
    """One SLURM array task: record a model's conditions, then analyse them."""
    from hier_switch_test_inference import main as record
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0) if index is None else index)
    model_type, seed, path, conditions = models()[index]
    if not os.path.exists(path):
        raise SystemExit(f'no model at {path}')
    sigma = check_level(path)
    print(f'--- task {index}: level {LEVEL} (noise {NOISE}), tags NG={NG_TAG} RNN={RNN_TAG}, '
          f'NG seeds {tuple(NG_SEEDS)}')
    print(f'--- {model_type} seed {seed}, trained at noise {sigma}: {", ".join(conditions)}')
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
    elif mode == 'probe':
        probe()
    else:
        raise SystemExit(__doc__)
