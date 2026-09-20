"""
hier_switch_analyses.py — per-trial extraction and summary numbers for the hierarchical
cue→rule reversal task. Used by hier_switch_train.py to judge whether a run has found the
context structure, and by the phase-2 analyses (docs/hier_switch_analyses.md) through the
session layer at the end of this file: load_session → trial_labels → select / z_updates.

Z convention (same as flanker, docs/flanker_task.md convention 2): the logged Z already
contains the trial's own latent update, so the state a trial *used* is the previous trial's
logged Z — `z_in`.
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


def extract_trials(logger, config):
    """Unpack a logger into per-trial arrays (one row per trial, all phases)."""
    L = config.trial_len
    ii = np.concatenate(logger.inputs, axis=0).reshape(-1, config.input_size)
    oi = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1, config.output_size)
    zz = np.concatenate(logger.latent_values, axis=0)
    zz = zz.reshape(-1, zz.shape[-1])
    n = len(ii) // L
    lab = ii[:n * L].reshape(n, L, -1)[:, -1]     # last frame: every label, targets included
    out = oi[:n * L].reshape(n, L, -1)[:, :, -1]

    decision = out[:, config.response_start_timestep:].mean(axis=1)
    ctx = lab[:, config.ch('context')].astype(int)
    z = zz[:n * L].reshape(n, L, -1)[:, 0]
    z_in = np.roll(z, 1, axis=0)
    z_in[0] = np.nan

    # Trials since the last reversal (1 = first trial of a block).
    since = np.ones(n, dtype=int)
    for k in range(1, n):
        since[k] = 1 if ctx[k] != ctx[k - 1] else since[k - 1] + 1

    # Phase of each trial, from the logger's phase markers (timestep → trial).
    phase = np.empty(n, dtype=object)
    marks = [(name, ts // L) for name, ts in logger.phases] + [(None, n)]
    for (name, start), (_, end) in zip(marks[:-1], marks[1:]):
        phase[start:end] = name

    return dict(
        decision=decision,
        correct=np.sign(decision) == lab[:, config.ch('correct')],
        cue=lab[:, config.ch('cue')],
        vis=lab[:, config.ch('vis')],
        conflict=np.round(lab[:, config.ch('conflict')], 3),
        context=ctx,
        since=since,
        phase=phase,
        z=z,
        z_in=z_in,
        n=n,
    )


def z_context_axis(z, ctx):
    """Axis between the two context means of Z, the midpoint, and d' along it."""
    m0, m1 = np.nanmean(z[ctx == 0], axis=0), np.nanmean(z[ctx == 1], axis=0)
    axis = m1 - m0
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm == 0:
        return None
    axis = axis / norm
    proj = z @ axis
    mid = 0.5 * (m0 + m1) @ axis
    p0, p1 = proj[ctx == 0], proj[ctx == 1]
    sd = np.sqrt(0.5 * (np.nanvar(p0) + np.nanvar(p1)))
    return dict(axis=axis, mid=mid, proj=proj,
                dprime=float(abs(np.nanmean(p1) - np.nanmean(p0)) / max(sd, 1e-12)))


SINCE_BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 1000)]


def summarize(trials, phase=None, last_frac=1.0):
    """Summary numbers for one phase (or all), optionally only its last `last_frac`.

    acc_since   accuracy by trials since reversal (SINCE_BINS)
    acc_conf    accuracy by conflict level, trials 11+ into a block (steady state)
    z_dprime    separation of the two contexts along the Z axis between their means,
                using z_in — the state each trial actually ran under
    z_acc_since fraction of trials whose z_in sits on the correct side of the midpoint,
                by trials since reversal: how fast the latent tracks the context
    """
    sel = np.ones(trials['n'], dtype=bool) if phase is None else (trials['phase'] == phase)
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return None
    idx = idx[int(len(idx) * (1 - last_frac)):]
    sub = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == trials['n'] else v)
           for k, v in trials.items()}
    c, s = sub['correct'], sub['since']

    res: dict = dict(n=len(idx), acc=float(c.mean()))
    res['acc_since'] = {f'{a}-{b}' if a != b else f'{a}': float(c[(s >= a) & (s <= b)].mean())
                        for a, b in SINCE_BINS if np.any((s >= a) & (s <= b))}
    # First trial after a reversal whose 3-trial mean accuracy reaches chance: the switch point.
    by_k = np.array([c[s == k].mean() if np.any(s == k) else np.nan for k in range(1, 31)])
    smooth = np.convolve(np.nan_to_num(by_k, nan=0.0), np.ones(3) / 3, mode='valid')
    hit = np.flatnonzero(smooth >= 0.5)
    res['cross_trial'] = int(hit[0] + 1) if len(hit) else None
    steady = s >= 11
    res['acc_conf'] = {float(v): float(c[steady & (sub['conflict'] == v)].mean())
                       for v in np.unique(sub['conflict'])}

    ok = np.isfinite(sub['z_in']).all(axis=1)
    ax = z_context_axis(sub['z_in'][ok], sub['context'][ok])
    if ax is not None:
        res['z_dprime'] = ax['dprime']
        side = (ax['proj'] > ax['mid']).astype(int)
        # Which side of the axis is context 1: by construction the axis points 0 → 1.
        z_ok = side == sub['context'][ok]
        so = s[ok]
        res['z_acc_since'] = {f'{a}-{b}' if a != b else f'{a}': float(z_ok[(so >= a) & (so <= b)].mean())
                              for a, b in SINCE_BINS if np.any((so >= a) & (so <= b))}
        res['z_mean_ctx'] = [np.round(np.nanmean(sub['z_in'][sub['context'] == k], axis=0), 3).tolist()
                             for k in (0, 1)]
    return res


def print_summary(res, label=''):
    if res is None:
        print(f'{label}: no trials')
        return
    fmt = lambda d: '  '.join(f'{k}:{v:.2f}' for k, v in d.items())
    print(f'{label}  n={res["n"]}  acc={res["acc"]:.3f}')
    print(f'    acc by trials since reversal   {fmt(res["acc_since"])}')
    print(f'    acc by conflict (steady)       {fmt(res["acc_conf"])}')
    if 'z_dprime' in res:
        print(f'    Z context d\'={res["z_dprime"]:.2f}   Z means ctx0/ctx1 {res["z_mean_ctx"]}')
        print(f'    Z on correct side, by since    {fmt(res["z_acc_since"])}')


def block_table(trials, phase='Learning and inference', early=50, late_frac=0.2):
    """One row per context block of a phase: does the model converge *within* a block?

    acc_early   accuracy over the block's first `early` trials (relearning after a reversal)
    acc_late    accuracy over the block's last `late_frac` of trials (converged level)
    abs_dec     mean |decision| late in the block; ~0 is a model hedging, not deciding
    z_late      mean Z late in the block
    """
    idx = np.flatnonzero(trials['phase'] == phase)
    rows = []
    if len(idx) == 0:
        return rows
    starts = idx[trials['since'][idx] == 1]
    if len(starts) == 0 or starts[0] != idx[0]:
        starts = np.concatenate([[idx[0]], starts])
    ends = np.concatenate([starts[1:], [idx[-1] + 1]])
    for a, b in zip(starts, ends):
        n = b - a
        late = slice(b - max(1, int(n * late_frac)), b)
        rows.append(dict(start=int(a), n=int(n), context=int(trials['context'][a]),
                         acc_early=float(trials['correct'][a:a + min(early, n)].mean()),
                         acc_late=float(trials['correct'][late].mean()),
                         abs_dec=float(np.abs(trials['decision'][late]).mean()),
                         z_late=[round(float(v), 2) for v in np.nanmean(trials['z'][late], axis=0)]))
    return rows


def print_block_table(rows, label=''):
    print(f'{label}')
    print(f'    {"start":>6} {"n":>5} {"ctx":>3} {"early":>6} {"late":>6} {"|dec|":>6}  Z late')
    for r in rows:
        print(f'    {r["start"]:6d} {r["n"]:5d} {r["context"]:3d} {r["acc_early"]:6.2f} '
              f'{r["acc_late"]:6.2f} {r["abs_dec"]:6.2f}  {r["z_late"]}')


# ══════════════════════════════════════════════════════════════════════════════
# Sessions and the trial table (phase 2). Definitions: docs/hier_switch_analyses.md.
# ══════════════════════════════════════════════════════════════════════════════

PRIMARY = 'Inference only'     # weights frozen, Z inferred, the paper's 30-60 blocks
STEADY = 11                    # `since` from which a block counts as steady state
EARLY_N = 5                    # the paper's early-reversal window (trials 1-5)

# Latent settings a session was run with, when its files do not say (the config defaults).
_DEFAULT_META = dict(model_type='NG', Z_lr=1e4, Z_decay=3e-6, latent_activation='softmax',
                     softmax_temp=0.5, rt_threshold=0.5, lu_steps=1, wu_steps=0,
                     response_start_timestep=19, trial_len=25, n_pulses=16,
                     pulse_noise_std=0.5, reversal_conflict=None, z_restart=False,
                     conflict_counts=[[9, 0], [8, 1], [7, 2], [6, 3], [5, 4]])


def gate(z, activation, temp=0.5):
    """The gate a raw Z applies: softmax(z / T), sigmoid(z), or z itself ('none')."""
    z = np.asarray(z, dtype=float)
    if activation == 'softmax':
        # max over finite entries only: an all-NaN row (z_in of trial 0) stays NaN quietly
        e = np.exp((z - np.max(np.where(np.isfinite(z), z, -np.inf), axis=-1, keepdims=True)) / temp)
        return e / e.sum(axis=-1, keepdims=True)
    if activation == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-z))
    return z


def recover_grad(z, z_in, Z_lr, Z_decay):
    """The trial's pooled error gradient dL/dZ, from the saved Z alone.

    One plain-SGD LU step per trial, no momentum: z = z_in − Z_lr·(g + Z_decay·z_in), so
    g = −(z − z_in)/Z_lr − Z_decay·z_in. (logger.gradients_corrections is g + Z_decay·z_in:
    the decay is added to Z.grad in place before the step.)
    """
    z, z_in = np.asarray(z, float), np.asarray(z_in, float)
    return -(z - z_in) / Z_lr - Z_decay * z_in


def session_arrays(logger, cfg):
    """Per-trial arrays a recorded session keeps beyond extract_trials.

    out     (n, trial_len)  the response output at every timestep (for RT)
    pulses  (n, n_pulses, 3) the noisy HP/LP/WN frames the model saw (for the observer)
    hidden  (n, trial_len, hidden) h of the acting forward, if config.record_hidden
    """
    L = cfg.trial_len
    oi = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1, cfg.output_size)
    ii = np.concatenate(logger.inputs, axis=0).reshape(-1, cfg.input_size)
    n = len(oi) // L
    res = dict(out=oi[:n * L].reshape(n, L, -1)[:, :, cfg.ch('correct')].astype(np.float32),
               pulses=ii[:n * L].reshape(n, L, -1)[:, :cfg.n_pulses, :3].astype(np.float32))
    trace = getattr(logger, 'hidden_trace', None)
    if trace:
        res['hidden'] = np.concatenate(trace, axis=0).reshape(n, L, -1)
    return res


def session_meta(cfg, **extra):
    """What load_session needs to know about how a session was run."""
    meta = {k: getattr(cfg, k, v) for k, v in _DEFAULT_META.items()
            if k not in ('model_type', 'lu_steps', 'wu_steps', 'z_restart')}
    meta.update(lu_steps=int(cfg.no_of_steps_in_latent_space),
                wu_steps=int(cfg.no_of_steps_in_weight_space),
                seed=int(getattr(cfg, 'env_seed', 0)),
                conflict_counts=[list(c) for c in cfg.conflict_counts])
    meta.update(extra)
    return meta


def save_session(path, trials, arrays, meta):
    """session.npz: every trials.npz key, plus out / pulses / hidden (float16) / grad /
    lu_scale / clamped and the meta as a json string."""
    n = trials['n']
    d = {k: v for k, v in trials.items() if isinstance(v, np.ndarray) and k != 'phase'}
    d['phase'] = trials['phase'].astype(str)
    d.update(out=arrays['out'], pulses=arrays['pulses'])
    if 'hidden' in arrays:
        d['hidden'] = arrays['hidden'].astype(np.float16)
    d['lu_scale'] = np.asarray(arrays.get('lu_scale', np.ones(n)), dtype=np.float32)
    d['clamped'] = np.asarray(arrays.get('clamped', np.zeros(n, bool)), dtype=bool)
    d['grad'] = _grad_or_nan(trials['z'], trials['z_in'], meta, d['lu_scale'], d['clamped'])
    np.savez_compressed(path, meta=np.array(json.dumps(meta, default=float)), **d)


def _grad_or_nan(z, z_in, meta, lu_scale, clamped):
    g = recover_grad(z, z_in, meta['Z_lr'], meta['Z_decay'])
    bad = (lu_scale != 1) | clamped | (meta['lu_steps'] <= 0)
    g[bad] = np.nan
    return g.astype(np.float32)


def _meta_from_summary(folder):
    """Meta for a session saved before session.npz existed (a tune run or an inference test)."""
    meta = dict(_DEFAULT_META)
    f = os.path.join(folder, 'summary.json')
    if os.path.exists(f):
        with open(f) as fh:
            summ = json.load(fh)
        if 'overrides' in summ:                 # hier_switch_test_inference: Z restarted
            meta.update(summ['overrides'], z_restart=True, condition=summ.get('condition'))
        entry = summ.get('entry') or {}
        meta.update({k: v for k, v in entry.items() if k in meta or k in ('seed', 'name')})
        if entry.get('model') == 'RNN':
            meta.update(model_type='RNN', lu_steps=0)
        elif entry.get('model'):
            meta['model_type'] = entry['model']
    return meta


def load_session(path):
    """The one reader every phase-2 analysis uses.

    `path` is a session.npz, a trials.npz, or a folder holding one (session.npz preferred).
    Old trials.npz files have no out / pulses / hidden; those keys are then absent. `grad`
    is recovered from Z when missing. Returns a dict of per-trial arrays plus 'n' and 'meta'.
    """
    if os.path.isdir(path):
        f = os.path.join(path, 'session.npz')
        path = f if os.path.exists(f) else os.path.join(path, 'trials.npz')
    d = dict(np.load(path, allow_pickle=False))
    meta = json.loads(str(d.pop('meta'))) if 'meta' in d else _meta_from_summary(os.path.dirname(path))
    meta = {**_DEFAULT_META, **meta, 'path': path}
    n = len(d['correct'])
    d.setdefault('lu_scale', np.ones(n, np.float32))
    d.setdefault('clamped', np.zeros(n, bool))
    if 'grad' not in d:
        d['grad'] = _grad_or_nan(d['z'], d['z_in'], meta, d['lu_scale'], d['clamped'])
    d['phase'] = d['phase'].astype(str)
    d.update(n=n, meta=meta)
    return d


def _blocks(sess):
    """Block index per trial (a block starts at since == 1 or at a phase boundary)."""
    since, phase, n = sess['since'], sess['phase'], sess['n']
    new = since == 1
    new[0] = True
    new[1:] |= phase[1:] != phase[:-1]
    block = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], n]
    return block, starts, ends


def trial_labels(sess, primary=PRIMARY):
    """Add the trial table to a session: one entry per trial, all numpy.

    Every later analysis composes boolean masks from these fields (select()). Fields:

    block, block_len, pos_from_end   block index, its length, trials left after this one
    reversal     the trial's block began with a real reversal (not a session or Z start)
    transient    first block of a session whose Z restarted at Z_init (run_test): drop it
    complete     the block ended in a reversal inside the same phase (not truncated)
    err, rule, conf_level            ~correct; cue × ctx_sign (+1 attend vision); 0..4
    axis, mid, d                     context axis between the steady-state (since ≥ 11)
                                     z_in means of the two contexts, on the primary phase;
                                     midpoint projection; prototype distance ‖m1 − m0‖
    proj_in, held, aligned, stale    z_in on that axis; which side (−1 if Z is NaN or no
                                     axis); held == true context; held the other one
    contrast_in, gain_in             (z1 − z2)/2 and mean(z) of z_in; gate_* the same for the
                                     applied gate
    rt, decided                      threshold crossing of |out| from t = 19; undecided → 25
    early_conf, early_class          block: mean conflict of trials 1-5; 0 low / 1 high
                                     (forced level if reversal_conflict, else a median
                                     split over the primary phase's reversal blocks), −1 n/a
    switch_trial                     block: first correct trial with another correct within
                                     the next two (the paper's); z_switch_trial: first
                                     aligned; dec_switch_trial: as the paper's, decided only
    pre_switch                       since < switch_trial (all of a never-switching block)
    p_c, err_w, eps_cw               steady accuracy per conflict level (clipped ≤ 0.99);
                                     err / (1 − p_c); its mean over the previous 5 trials
                                     (not reset at reversals; reset at phase boundaries)
    """
    meta, n = sess['meta'], sess['n']
    since, ctx, phase = sess['since'], sess['context'].astype(int), sess['phase']
    block, starts, ends = _blocks(sess)
    blen = ends - starts
    pos = np.arange(n) - starts[block]
    rev_blk = since[starts] == 1
    rev_blk[0] = False                              # the session's first block: no reversal
    trans_blk = np.zeros(len(starts), bool)
    if meta.get('z_restart'):
        trans_blk[0] = True
    complete_blk = np.array([e < n and phase[e] == phase[a] for a, e in zip(starts, ends)])

    lab = dict(block=block, block_len=blen[block], pos_from_end=blen[block] - 1 - pos,
               reversal=rev_blk[block], transient=trans_blk[block], complete=complete_blk[block])
    lab['err'] = ~sess['correct'].astype(bool)
    ctx_sign = np.where(ctx == 0, 1.0, -1.0)
    lab['rule'] = sess['cue'] * ctx_sign
    levels = np.array([non / dom for dom, non in meta['conflict_counts']])
    lab['conf_level'] = np.abs(sess['conflict'][:, None] - levels[None]).argmin(axis=1)
    use = (phase == primary) & ~lab['transient']

    # ── Z relative to the context axis ──
    zin = sess['z_in'].astype(float)
    ok = np.isfinite(zin).all(axis=1)
    steady = use & ok & (since >= STEADY)
    lab.update(axis=None, mid=np.nan, d=np.nan, proj_in=np.full(n, np.nan),
               held=np.full(n, -1))
    if (steady & (ctx == 0)).any() and (steady & (ctx == 1)).any():
        m0, m1 = zin[steady & (ctx == 0)].mean(0), zin[steady & (ctx == 1)].mean(0)
        d = float(np.linalg.norm(m1 - m0))
        if d > 1e-9:
            axis = (m1 - m0) / d
            lab.update(axis=axis, mid=float(0.5 * (m0 + m1) @ axis), d=d, m0=m0, m1=m1)
            lab['proj_in'] = np.where(ok, np.nan_to_num(zin) @ axis, np.nan)
            lab['held'] = np.where(ok, (lab['proj_in'] > lab['mid']).astype(int), -1)
    # Signed evidence for the *true* context along the axis, in half-prototype-distance
    # units: +1 means z_in sits on the true context's prototype, −1 on the other one, 0 at
    # the midpoint. The graded version of `aligned`, for the reversal-aligned Z curves.
    lab['z_evidence'] = (lab['proj_in'] - lab['mid']) / (lab['d'] / 2) * np.where(ctx == 1, 1.0, -1.0)
    lab['aligned'] = (lab['held'] >= 0) & (lab['held'] == ctx)
    lab['stale'] = (lab['held'] >= 0) & (lab['held'] != ctx)
    lab['contrast_in'] = 0.5 * (zin[:, 0] - zin[:, 1])
    lab['gain_in'] = zin.mean(axis=1)
    g = gate(zin, meta['latent_activation'], meta['softmax_temp'])
    lab['gate_in'] = g
    lab['gate_contrast_in'] = 0.5 * (g[:, 0] - g[:, 1])
    lab['gate_gain_in'] = g.mean(axis=1)

    # ── RT ──
    if 'out' in sess:
        from flanker_analyses import _interpolated_rt
        rt, _, decided, _ = _interpolated_rt(sess['out'], meta['rt_threshold'],
                                             search_from=meta['response_start_timestep'])
        lab['rt'], lab['decided'] = rt, decided

    # ── Per block: early-reversal conflict and the three switch criteria ──
    conf = sess['conflict'].astype(float)
    early_conf = np.full(len(starts), np.nan)
    sw = np.full((3, len(starts)), np.nan)
    corr = sess['correct'].astype(bool)
    dec_corr = corr & lab['decided'] if 'decided' in lab else None
    for b, (a, e) in enumerate(zip(starts, ends)):
        if not rev_blk[b]:
            continue
        if e - a >= EARLY_N:
            early_conf[b] = conf[a:a + EARLY_N].mean()
        sw[0, b] = _first_confirmed(corr[a:e])
        sw[1, b] = _first(lab['aligned'][a:e])
        if dec_corr is not None:
            sw[2, b] = _first_confirmed(dec_corr[a:e])
    early_class = np.full(len(starts), -1)
    cand = rev_blk & np.isfinite(early_conf) & np.array([use[a] for a in starts])
    forced = meta.get('reversal_conflict')
    if forced is not None:
        early_class[rev_blk] = {'low': 0, 'high': 1}[forced]
    elif cand.any():
        med = np.median(early_conf[cand])
        early_class[cand] = (early_conf[cand] > med).astype(int)
    lab.update(early_conf=early_conf[block], early_class=early_class[block],
               switch_trial=sw[0][block], z_switch_trial=sw[1][block],
               dec_switch_trial=sw[2][block])
    lab['pre_switch'] = lab['reversal'] & ~(since >= np.nan_to_num(lab['switch_trial'], nan=np.inf))

    # ── Conflict-weighted error (the paper's ε_CW) ──
    st = use & (since >= STEADY)
    p_tab = np.array([corr[st & (lab['conf_level'] == k)].mean() if (st & (lab['conf_level'] == k)).any()
                      else np.nan for k in range(len(levels))])
    lab['p_c_table'] = np.minimum(p_tab, 0.99)
    lab['p_c'] = lab['p_c_table'][lab['conf_level']]
    lab['err_w'] = lab['err'] / (1.0 - lab['p_c'])
    eps = np.full(n, np.nan)
    ph_new = np.r_[True, phase[1:] != phase[:-1]]
    ph_starts = np.flatnonzero(ph_new)
    for a, e in zip(ph_starts, np.r_[ph_starts[1:], n]):
        x = lab['err_w'][a:e]
        if len(x) > EARLY_N:
            cs = np.r_[0.0, np.cumsum(x)]
            eps[a + EARLY_N:e] = (cs[EARLY_N:len(x)] - cs[:len(x) - EARLY_N]) / EARLY_N
    lab['eps_cw'] = eps
    sess.update(lab)
    return sess


def _first(mask):
    """1-based position of the first True, NaN if none."""
    hit = np.flatnonzero(mask)
    return float(hit[0] + 1) if len(hit) else np.nan


def _first_confirmed(c):
    """1-based position of the first True followed by another True within the next two."""
    for k in range(len(c) - 1):
        if c[k] and c[k + 1:k + 3].any():
            return float(k + 1)
    return np.nan


def select(sess, include_transient=False, **crit):
    """Boolean mask over trials from the trial table.

    Each keyword names a per-trial field. A scalar or bool matches by equality, a list or
    set by membership, a (lo, hi) tuple is an inclusive range, a callable gets the array.
    Transient trials (the start-up block of a Z-restarted session) are excluded unless
    include_transient. Example:
        select(s, phase='Inference only', err=True, stale=True, conf_level=[3, 4])
    """
    m = np.ones(sess['n'], dtype=bool)
    if not include_transient and 'transient' in sess:
        m &= ~sess['transient']
    for k, v in crit.items():
        a = sess[k]
        if callable(v):
            m &= np.asarray(v(a), dtype=bool)
        elif isinstance(v, tuple):
            m &= (a >= v[0]) & (a <= v[1])
        elif isinstance(v, (list, set, np.ndarray)):
            m &= np.isin(a, list(v))
        else:
            m &= a == v
    return m


# ══════════════════════════════════════════════════════════════════════════════
# B5: how each trial moves Z
# ══════════════════════════════════════════════════════════════════════════════

def z_updates(sess, primary=PRIMARY):
    """The trial's own latent update, split into its error and weight-decay parts, measured
    along the context axis (contrast) and the unit mean (gain). Needs trial_labels.

    dz, dz_decay, dz_err   z − z_in; −Z_lr·Z_decay·z_in; the rest (this trial's error)
    s          (dz_err · axis)/d, signed +1 toward the context Z was *not* holding: the
               fraction of the way to the other prototype this trial's error moved Z
    s_decay    the same for dz_decay (decay pulls toward the middle, so it is positive)
    tipped     the side of z differs from the side of z_in (the full update crossed mid)
    dgain      Δ mean(z) (error part; under the softmax it is identically 0)
    dgate_gain Δ mean(gate(z)), the full update in gate units
    ok         trials the numbers are defined on: primary phase, not transient, LU on and
               unscaled, not clamped, finite Z, an axis exists

    Never run it on the RNN: LU is off there and Δz is 0 by construction.
    """
    meta = sess['meta']
    if meta['lu_steps'] <= 0:
        raise ValueError('z_updates on a session with LU off (the RNN): Δz is 0 by construction')
    z, zin = sess['z'].astype(float), sess['z_in'].astype(float)
    dz = z - zin
    dz_decay = -meta['Z_lr'] * meta['Z_decay'] * zin
    dz_err = dz - dz_decay
    finite = np.isfinite(z).all(1) & np.isfinite(zin).all(1)
    ok = (select(sess, phase=primary) & finite & (sess['held'] >= 0)
          & ~sess['clamped'] & (sess['lu_scale'] == 1))
    n = sess['n']
    res = dict(dz=dz, dz_decay=dz_decay, dz_err=dz_err, ok=ok, s=np.full(n, np.nan),
               s_decay=np.full(n, np.nan), tipped=np.zeros(n, bool))
    if sess['axis'] is not None:
        sign = np.where(sess['held'] == 0, 1.0, -1.0)
        ax, d = sess['axis'], sess['d']
        res['s'] = np.where(ok, np.nan_to_num(dz_err) @ ax / d * sign, np.nan)
        res['s_decay'] = np.where(ok, np.nan_to_num(dz_decay) @ ax / d * sign, np.nan)
        held_out = (np.nan_to_num(z) @ ax > sess['mid']).astype(int)
        res['tipped'] = ok & (held_out != sess['held'])
    res['dgain'] = np.where(ok, dz_err.mean(1), np.nan)
    res['dgain_decay'] = np.where(ok, dz_decay.mean(1), np.nan)
    g_out = gate(z, meta['latent_activation'], meta['softmax_temp'])
    res['dgate_gain'] = np.where(ok, g_out.mean(1) - sess['gate_gain_in'], np.nan)
    res['dgate_contrast'] = np.where(ok, 0.5 * (g_out[:, 0] - g_out[:, 1]) - sess['gate_contrast_in'], np.nan)
    return res


def z_update_table(sess, upd, by=('state', 'err', 'conf_level'), position=None):
    """Mean s, P(tipped), mean Δgain and n per cell of `by`.

    by: any of 'state' (0 aligned / 1 stale), 'err', 'conf_level', 'context'.
    position: None (all), 'early' (since 1-5) or 'steady' (since ≥ 11).
    Returns {cell tuple: dict(n, s, s_sem, tipped, dgain, dgate_gain)}.
    """
    ok = upd['ok'].copy()
    if position == 'early':
        ok &= sess['since'] <= EARLY_N
    elif position == 'steady':
        ok &= sess['since'] >= STEADY
    keys = dict(state=sess['stale'].astype(int), err=sess['err'].astype(int),
                conf_level=sess['conf_level'], context=sess['context'].astype(int))
    cols = [keys[b] for b in by]
    out = {}
    for cell in sorted({tuple(int(c[i]) for c in cols) for i in np.flatnonzero(ok)}):
        m = ok.copy()
        for c, v in zip(cols, cell):
            m &= c == v
        s = upd['s'][m]
        out[cell] = dict(n=int(m.sum()), s=float(s.mean()),
                         s_sem=float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan,
                         tipped=float(upd['tipped'][m].mean()),
                         dgain=float(upd['dgain'][m].mean()),
                         dgate_gain=float(upd['dgate_gain'][m].mean()))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# A1-A4: behaviour
# ══════════════════════════════════════════════════════════════════════════════

REV_WINDOW = (-5, 15)      # trials around a reversal; k = 1 is the first trial of the block


def reversal_aligned(sess, values, window=REV_WINDOW, blocks=None, primary=PRIMARY):
    """`values` averaged at each offset from a reversal. k = 1 is the block's first trial,
    k ≤ 0 the trials before it (k = 0 is the last trial of the previous block).

    Returns (k, mean, sem, n). Offsets that would leave the block's phase, run past the
    session, or reach back beyond the previous block are dropped.
    """
    block, starts, ends = _blocks(sess)
    n, phase = sess['n'], sess['phase']
    keep = sess['reversal'] & ~sess['transient'] & (phase == primary)
    sel = [b for b in range(len(starts)) if keep[starts[b]] and (blocks is None or blocks[b])]
    ks = np.arange(window[0], window[1] + 1)
    vals = np.full((len(sel), len(ks)), np.nan)
    for r, b in enumerate(sel):
        a, e = starts[b], ends[b]
        for j, k in enumerate(ks):
            i = a + k - 1
            lo = starts[b - 1] if b > 0 else 0
            if lo <= i < min(e, n) and phase[i] == phase[a]:
                vals[r, j] = values[i]
    # Computed by hand rather than with nanmean/nanstd: an offset no block reaches is an
    # all-NaN column, and those warn rather than simply returning NaN.
    fin = np.isfinite(vals)
    cnt = fin.sum(axis=0)
    tot = np.where(fin, vals, 0.0).sum(axis=0)
    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    dev = np.where(fin, (vals - mean) ** 2, 0.0).sum(axis=0)
    sd = np.where(cnt > 1, np.sqrt(dev / np.maximum(cnt - 1, 1)), np.nan)
    return ks, mean, sd / np.sqrt(np.maximum(cnt, 1)), cnt


def _block_first(sess, field):
    """One value per block, read from the block's first trial."""
    _, starts, _ = _blocks(sess)
    return sess[field][starts], starts


def behaviour(sess, primary=PRIMARY, criterion_n=3):
    """A1-A4 in one dict: psychometric, reversal-aligned curves, switch latency, RT.

    A1 psychometric   accuracy / undecided rate / RT by conflict level, steady state
                      (since ≥ 11), per context — they should be mirror images
    A2 reversal       accuracy, Z on the true context's side, |decision|, undecided and RT
                      from 5 trials before to 15 after a reversal, split by early_class,
                      plus the three switch latencies per split
    A3 perseveration  errors before the switch, and trials to a 3-in-a-row criterion
    A4 RT             by conflict and by trials since the reversal; the fast-error check
                      (RT of errors on trials 1-2 against steady-state errors), with a
                      decided-only companion, since an undecided trial has no honest RT
    """
    m = select(sess, phase=primary)
    steady = m & (sess['since'] >= STEADY)
    corr = sess['correct'].astype(bool)
    has_rt = 'rt' in sess
    rt = sess['rt'] if has_rt else np.full(sess['n'], np.nan)
    dec = sess['decided'] if has_rt else np.zeros(sess['n'], bool)
    levels = np.array([non / dom for dom, non in sess['meta']['conflict_counts']])
    mean = lambda x: float(np.mean(x)) if len(x) else np.nan

    res = dict(n=int(m.sum()), acc=mean(corr[m]), acc_steady=mean(corr[steady]),
               undecided=mean(~dec[m]) if has_rt else np.nan,
               conflict=levels.round(3).tolist())
    res['psychometric'] = dict(
        acc=[mean(corr[steady & (sess['conf_level'] == c)]) for c in range(len(levels))],
        acc_ctx=[[mean(corr[steady & (sess['conf_level'] == c) & (sess['context'] == k)])
                  for c in range(len(levels))] for k in (0, 1)],
        rt=[mean(rt[steady & (sess['conf_level'] == c)]) for c in range(len(levels))],
        undecided=[mean(~dec[steady & (sess['conf_level'] == c)]) for c in range(len(levels))],
        n=[int((steady & (sess['conf_level'] == c)).sum()) for c in range(len(levels))])

    # A2/A3, all reversals and split by the conflict of the first 5 trials
    z_side = sess['aligned'].astype(float)
    z_side[sess['held'] < 0] = np.nan
    curves = dict(acc=corr.astype(float), z_side=z_side, z_evidence=sess['z_evidence'],
                  abs_dec=np.abs(sess['decision']), rt=rt, undecided=(~dec).astype(float))
    ec, starts = _block_first(sess, 'early_class')
    res['reversal'] = {}
    for label, blk in (('all', None), ('low', ec == 0), ('high', ec == 1)):
        if blk is not None and not blk.any():
            continue
        d = {}
        for key, v in curves.items():
            ks, mu, sem, cnt = reversal_aligned(sess, v, blocks=blk, primary=primary)
            d[key], d[key + '_sem'] = mu.tolist(), sem.tolist()
            d['k'], d['n_blocks'] = ks.tolist(), int(cnt.max()) if len(cnt) else 0
        res['reversal'][label] = d
        res.setdefault('switch', {})[label] = _switch_stats(sess, blk, primary, criterion_n)

    res['rt'] = dict(
        by_since=[mean(rt[m & (sess['since'] == k)]) for k in range(1, 16)],
        undecided_by_since=[mean(~dec[m & (sess['since'] == k)]) for k in range(1, 16)],
        err_early=mean(rt[m & ~corr & (sess['since'] <= 2)]),
        err_steady=mean(rt[steady & ~corr]),
        err_early_decided=mean(rt[m & ~corr & dec & (sess['since'] <= 2)]),
        err_steady_decided=mean(rt[steady & ~corr & dec]),
        corr_steady=mean(rt[steady & corr]))
    return res


def _switch_stats(sess, blk, primary, criterion_n=3):
    """Switch latency (three criteria), perseverative errors and trials to criterion, per
    reversal block, averaged over the blocks of one split."""
    from mean_prediction_analysis import _find_criterion
    block, starts, ends = _blocks(sess)
    keep = sess['reversal'] & ~sess['transient'] & (sess['phase'] == primary)
    sel = [b for b in range(len(starts)) if keep[starts[b]] and (blk is None or blk[b])]
    corr = sess['correct'].astype(bool)
    sw = {k: [] for k in ('switch', 'z_switch', 'dec_switch', 'persev', 'ttc', 'never')}
    for b in sel:
        a, e = starts[b], ends[b]
        sw['switch'].append(sess['switch_trial'][a])
        sw['z_switch'].append(sess['z_switch_trial'][a])
        sw['dec_switch'].append(sess['dec_switch_trial'][a])
        t = _find_criterion(corr[a:e], criterion_n)
        sw['ttc'].append(float(t) if t is not None else np.nan)
        sw['persev'].append(float((~corr[a:a + t]).sum()) if t is not None else float((~corr[a:e]).sum()))
        sw['never'].append(t is None)
    out = {k: (float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
           for k, v in ((k, np.asarray(v, dtype=float)) for k, v in sw.items() if k != 'never')}
    out.update(n_blocks=len(sel), never_frac=float(np.mean(sw['never'])) if sel else np.nan,
               switch_n=int(np.sum(np.isfinite(sw['switch']))))
    return out



# ══════════════════════════════════════════════════════════════════════════════
# B1-B4: the latent against the ideal observer (the MD / ACC analogues)
# ══════════════════════════════════════════════════════════════════════════════

def _corr(x, y):
    """Pearson r over the entries where both are finite."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _ols(y, X):
    """Least squares with an intercept: (coefficients, R²)."""
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if m.sum() <= X.shape[1] + 1:
        return np.full(X.shape[1] + 1, np.nan), np.nan
    A = np.c_[X[m], np.ones(m.sum())]
    beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    resid = y[m] - A @ beta
    ss = np.sum((y[m] - y[m].mean()) ** 2)
    return beta, float(1 - np.sum(resid ** 2) / ss) if ss > 0 else np.nan


def latent(sess, obs, upd=None, primary=PRIMARY):
    """B1-B4: Z against the ideal observer, and the gradient against the paper's ε_CW.

    B1 belief      z_evidence (Z on the context axis, +1 = the true context's prototype)
                   against the observer's belief in the true context: correlation, the lag
                   that maximises it, and both curves around a reversal
    B2 uncertainty 1 − |z_evidence| (0 = Z sits on a prototype, 1 = at the midpoint): its
                   peak height, position and width after a reversal, split by early_class
                   (Fig 1k,l), and its correlation with cue conflict over the first 5 trials
                   (Fig 1m; the paper finds it negative)
    B3 update rule which of err, err × conflict and err × the observer's log-odds update
                   explains the trial's move along the context axis (`s`)
    B4 ε_CW        the pooled |dL/dZ| against err/(1 − P_c) per trial and against ε_CW over
                   the paper's 5-trial window: slope and R². The contrast component
                   |g₀ − g₁|/2 is the one the softmax can act on; the gain component
                   |mean g| is reported next to it (identically inert under the softmax)
    """
    m = select(sess, phase=primary)
    unc = 1.0 - np.minimum(np.abs(sess['z_evidence']), 1.0)
    b_true = obs['b_true']
    res = dict(n=int(m.sum()))

    # B1 ── belief
    lags = np.arange(-5, 6)
    rs = [_corr(np.roll(sess['z_evidence'], -int(l))[m], b_true[m]) for l in lags]
    res['belief'] = dict(r=_corr(sess['z_evidence'][m], b_true[m]),
                         lags=lags.tolist(), r_by_lag=[float(v) for v in rs],
                         best_lag=int(lags[int(np.nanargmax(rs))]) if np.isfinite(rs).any() else None,
                         r_aligned=_corr(sess['aligned'][m].astype(float), b_true[m]))
    for key, v in (('z', sess['z_evidence']), ('obs', 2 * b_true - 1)):
        ks, mu, sem, _ = reversal_aligned(sess, v, primary=primary)
        res['belief'][f'{key}_curve'], res['belief'][f'{key}_sem'] = mu.tolist(), sem.tolist()
    res['belief']['k'] = ks.tolist()

    # B2 ── uncertainty
    res['uncertainty'] = {}
    ec, starts = _block_first(sess, 'early_class')
    for label, blk in (('all', None), ('low', ec == 0), ('high', ec == 1)):
        if blk is not None and not blk.any():
            continue
        ks, mu, sem, _ = reversal_aligned(sess, unc, blocks=blk, primary=primary)
        post = mu[ks >= 1]
        pk = int(np.nanargmax(post)) if np.isfinite(post).any() else None
        base = np.nanmean(mu[ks <= 0])
        half = np.nan
        if pk is not None:
            above = np.flatnonzero(post >= base + 0.5 * (post[pk] - base))
            half = float(above[-1] - above[0] + 1) if len(above) else np.nan
        res['uncertainty'][label] = dict(k=ks.tolist(), curve=mu.tolist(), sem=sem.tolist(),
                                         baseline=float(base),
                                         peak=float(post[pk]) if pk is not None else np.nan,
                                         peak_k=int(pk + 1) if pk is not None else None,
                                         width_half=half)
    early = m & (sess['since'] <= EARLY_N) & sess['reversal']
    prev_conf = np.r_[np.nan, sess['conflict'][:-1].astype(float)]
    res['uncertainty']['r_conflict_early'] = _corr(unc[early], sess['conflict'][early])
    res['uncertainty']['r_prev_conflict_early'] = _corr(unc[early], prev_conf[early])
    res['uncertainty']['r_conflict_steady'] = _corr(unc[m & (sess['since'] >= STEADY)],
                                                    sess['conflict'][m & (sess['since'] >= STEADY)])
    # Fig 1m at the level the model can express it: a block whose first trials were
    # ambiguous should reach a *lower* peak of Z uncertainty (the errors are discounted).
    _, starts, ends = _blocks(sess)
    keep = sess['reversal'] & ~sess['transient'] & (sess['phase'] == primary)
    pk = [(sess['early_conf'][a], np.nanmax(unc[a:min(a + 10, e)]))
          for a, e in zip(starts, ends) if keep[a] and np.isfinite(sess['early_conf'][a])]
    if len(pk) > 3:
        ecf, pku = map(np.array, zip(*pk))
        res['uncertainty']['r_peak_vs_early_conflict'] = _corr(pku, ecf)
        res['uncertainty']['peak_by_block'] = pku.tolist()
        res['uncertainty']['early_conf_by_block'] = ecf.tolist()

    # B3 ── what drives the update
    if upd is not None:
        ok = upd['ok']
        y = np.where(ok, upd['s'], np.nan)
        err = sess['err'].astype(float)
        terms = dict(err=err, err_x_conflict=err * sess['conflict'],
                     err_x_dlogit=err * np.abs(obs['dlogit']),
                     dlogit_other=np.where(sess['held'] == 0, obs['dlogit'], -obs['dlogit']))
        res['update_rule'] = {k: dict(zip(('beta', 'r2'), _ols(y, v[:, None])))
                              for k, v in terms.items()}
        res['update_rule'] = {k: dict(beta=float(v['beta'][0]), r2=v['r2'])
                              for k, v in res['update_rule'].items()}
        beta, r2 = _ols(y, np.c_[terms['err'], terms['err_x_conflict']])
        res['update_rule']['err_plus_interaction'] = dict(beta=beta.tolist(), r2=r2)

    # B4 ── the gradient as the paper's conflict-weighted error
    g = sess['grad'].astype(float)
    g_contrast = np.abs(0.5 * (g[:, 0] - g[:, 1]))
    g_gain = np.abs(g.mean(axis=1))
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(g_contrast))]
    g5 = np.full(sess['n'], np.nan)
    g5[EARLY_N:] = (cs[EARLY_N:-1] - cs[:-1 - EARLY_N]) / EARLY_N   # the previous 5 trials
    res['eps_cw'] = dict(
        r_single=_corr(g_contrast[m], sess['err_w'][m]),
        r_window=_corr(g5[m], sess['eps_cw'][m]),
        slope_window=float(_ols(g5[m], sess['eps_cw'][m][:, None])[0][0]),
        r2_window=_ols(g5[m], sess['eps_cw'][m][:, None])[1],
        mean_eps_cw=float(np.nanmean(sess['eps_cw'][m & (sess['since'] >= STEADY)])),
        g_contrast_by_conf=[float(np.nanmean(g_contrast[m & sess['err'] & (sess['conf_level'] == c)]))
                            for c in range(len(sess['p_c_table']))],
        g_contrast_by_conf_correct=[float(np.nanmean(g_contrast[m & ~sess['err'] & (sess['conf_level'] == c)]))
                                    for c in range(len(sess['p_c_table']))],
        g_gain_mean=float(np.nanmean(g_gain[m])), g_contrast_mean=float(np.nanmean(g_contrast[m])))
    return res


def normative_table(sess, upd, obs, primary=PRIMARY):
    """B5's normative comparison: the model's mean `s` against the observer's log-odds
    update, per conflict level × outcome, with the model's own P_c for reference.

    The observer's update toward the context it does not believe is ±log(P_c/(1 − P_c)) in
    the paper's notation; here it is measured per trial from the pulses rather than from a
    table, so it carries the trial's own cue evidence. It is signed the same way as the
    model's `s`: toward the context *Z* was not holding, not the one the observer doubted,
    so the two are directly comparable even while the model is stale and the observer is not.
    """
    ok = upd['ok']
    dl_other = np.where(sess['held'] == 0, obs['dlogit'], -obs['dlogit'])
    out = dict(p_c_observer=obs['p_c'], p_c_model=sess['p_c_table'].tolist(), cells=[])
    for err in (1, 0):
        for c in range(len(sess['p_c_table'])):
            mm = ok & (sess['err'] == bool(err)) & (sess['conf_level'] == c)
            if mm.sum() < 3:
                continue
            out['cells'].append(dict(err=err, conf_level=c, n=int(mm.sum()),
                                     s=float(np.nanmean(upd['s'][mm])),
                                     dlogit=float(np.nanmean(dl_other[mm])),
                                     dlogit_abs=float(np.nanmean(np.abs(obs['dlogit'][mm])))))
    xs = np.array([c['dlogit'] for c in out['cells']])
    ys = np.array([c['s'] for c in out['cells']])
    out['r'] = _corr(xs, ys)
    out['slope'] = float(_ols(ys, xs[:, None])[0][0]) if len(xs) > 2 else np.nan
    return out



# ══════════════════════════════════════════════════════════════════════════════
# Self-test: a synthetic session whose answers are known by construction.
#   .venv/bin/python hier_switch/hier_switch_analyses.py
# ══════════════════════════════════════════════════════════════════════════════

def synthetic_session(blen=20, n_blocks=4, seed=0):
    """A hand-built session: known Z trajectory, known updates, known outcomes.

    Z is simulated exactly as the model updates it — z = z_in − Z_lr·(g + Z_decay·z_in) —
    with the error part written directly as a chosen fraction `s_true` of the way toward the
    other context's prototype. Trials 4-6 of every block after the first are the switch.
    """
    rng = np.random.default_rng(seed)
    n = blen * n_blocks
    ctx = np.repeat(np.arange(n_blocks) % 2, blen)
    since = np.tile(np.arange(1, blen + 1), n_blocks)
    proto = np.array([[1.0, -1.0], [-1.0, 1.0]])          # context 0 and context 1
    d0 = float(np.linalg.norm(proto[1] - proto[0]))
    u = (proto[1] - proto[0]) / d0
    Z_lr, Z_decay = 1e4, 3e-6                              # Z_lr·Z_decay = 0.03

    s_true = np.zeros(n)
    s_true[(since == 4) & (np.arange(n) >= blen)] = 1.3     # the tipping trial of each block
    z, z_in = np.zeros((n, 2)), np.zeros((n, 2))
    cur = proto[ctx[0]].copy()
    for k in range(n):
        z_in[k] = cur
        held = 0 if cur @ u < 0 else 1
        dz_err = s_true[k] * d0 * (1.0 if held == 0 else -1.0) * u
        cur = cur - Z_lr * Z_decay * cur + dz_err
        z[k] = cur
    z_in[0] = np.nan

    conf_level = rng.integers(0, 5, n)
    levels = np.array([0, 1 / 8, 2 / 7, 3 / 6, 4 / 5])
    correct = np.ones(n, bool)
    for b in range(1, n_blocks):                            # errors until the switch
        a = b * blen
        correct[a:a + 3] = False
        correct[a + 4] = False                              # ...T F T T: first confirmed at 4
    out = np.zeros((n, 25), np.float32)                     # ramp crossing 0.5 at t = 21
    out[:, 19:] = np.array([0.0, 0.25, 0.75, 1.0, 1.0, 1.0], np.float32)
    out *= np.where(correct, 1.0, -1.0)[:, None]
    sess = dict(n=n, context=ctx, since=since, z=z, z_in=z_in, correct=correct,
                conflict=levels[conf_level], cue=np.where(rng.random(n) < 0.5, 1.0, -1.0),
                vis=np.where(rng.random(n) < 0.5, 1.0, -1.0), out=out,
                decision=np.where(correct, 0.8, -0.8),
                phase=np.array([PRIMARY] * n), lu_scale=np.ones(n),
                clamped=np.zeros(n, bool),
                meta={**_DEFAULT_META, 'Z_lr': Z_lr, 'Z_decay': Z_decay})
    sess['grad'] = recover_grad(z, z_in, Z_lr, Z_decay)
    return sess, dict(s_true=s_true, proto=proto, u=u, d=d0, conf_level=conf_level)


def _self_test():
    blen, n_blocks = 20, 4
    sess, truth = synthetic_session(blen, n_blocks)
    trial_labels(sess)
    n = sess['n']
    since = sess['since']

    assert np.array_equal(sess['conf_level'], truth['conf_level']), 'conflict level mismatch'
    assert np.all(sess['block'] == np.repeat(np.arange(n_blocks), blen))
    assert np.all(sess['block_len'] == blen)
    assert sess['pos_from_end'][blen - 1] == 0 and sess['pos_from_end'][0] == blen - 1
    assert not sess['reversal'][0] and np.all(sess['reversal'][blen:]), 'reversal flag'
    assert np.all(sess['complete'][:-blen]) and not sess['complete'][-1], 'complete flag'
    assert np.allclose(sess['rule'], sess['cue'] * np.where(sess['context'] == 0, 1, -1))

    # Axis: the two prototypes, so d ≈ ‖m1 − m0‖ and the midpoint sits at 0 (decay is
    # symmetric), and Z holds the previous context for the first three trials of a block.
    assert abs(sess['d'] - truth['d'] * 0.75) < truth['d'], f"d={sess['d']:.2f} off scale"
    p0, p1 = sess['m0'] @ sess['axis'], sess['m1'] @ sess['axis']
    assert p0 < sess['mid'] < p1, f'midpoint {sess["mid"]} outside [{p0}, {p1}]'
    exp_stale = (since <= 4) & (np.arange(n) >= blen)      # tips on trial 4, so 1-4 are stale
    assert np.array_equal(sess['stale'], exp_stale), 'stale trials'
    exp_aligned = ~exp_stale
    exp_aligned[0] = False               # trial 0 has no z_in: neither aligned nor stale
    assert np.array_equal(sess['aligned'], exp_aligned), 'aligned trials'
    assert np.all(sess['held'][1:blen] == 0) and sess['held'][0] == -1

    # Switch criteria: errors on 1-3, correct on 4, error on 5, correct from 6.
    for b in range(1, n_blocks):
        a = b * blen
        assert sess['switch_trial'][a] == 4, f'block {b} switch {sess["switch_trial"][a]}'
        assert sess['dec_switch_trial'][a] == 4, 'decided switch'
        assert sess['z_switch_trial'][a] == 5, f'Z switch {sess["z_switch_trial"][a]}'
    assert np.array_equal(sess['pre_switch'][blen:blen + 5], [True] * 3 + [False] * 2)
    assert not sess['pre_switch'][:blen].any(), 'first block is not a reversal'

    # Updates: s is what was written in, only the tipping trials cross the midpoint.
    upd = z_updates(sess)
    ok = upd['ok']
    assert ok.sum() == n - 1 and not ok[0], 'trial 0 (NaN z_in) must be excluded'
    # s is a fraction of the *measured* prototype distance, which weight decay shrinks
    # below the distance the updates were written in (truth['d']).
    scale = truth['d'] / sess['d']
    assert np.allclose(upd['s'][ok], truth['s_true'][ok] * scale, atol=1e-9), 's does not match'
    assert np.array_equal(np.flatnonzero(upd['tipped']), np.flatnonzero(truth['s_true'] > 1)), 'tipped'
    assert np.allclose(upd['dz_decay'], -0.03 * sess['z_in'], equal_nan=True), 'decay part'
    assert np.all(upd['s_decay'][ok & (truth['s_true'] == 0)] > 0), 'decay pulls toward the middle'
    assert np.allclose(upd['dgain'][ok], 0, atol=1e-12), 'gain: the update is along (1, −1)'
    # The recovered gradient is the update: z_in − Z_lr·(g + Z_decay·z_in) == z.
    g = sess['grad']
    assert np.allclose(sess['z_in'][1:] - 1e4 * (g[1:] + 3e-6 * sess['z_in'][1:]), sess['z'][1:])

    # RT: the ramp crosses 0.5 between t = 20 (0.25) and t = 21 (0.75), i.e. at 20.5.
    assert np.allclose(sess['rt'], 20.5) and sess['decided'].all(), 'RT'

    # ε_CW: the trailing 5-trial mean of err/(1 − p_c), and ≈ 1 in steady state.
    k = 30
    assert np.isnan(sess['eps_cw'][:5]).all() and np.isfinite(sess['eps_cw'][5:]).all()
    assert np.isclose(sess['eps_cw'][k], sess['err_w'][k - 5:k].mean()), 'eps_cw window'
    assert np.all((sess['p_c'] >= 0) & (sess['p_c'] <= 0.99)), 'p_c range'

    # early_conf / early_class: a median split over the reversal blocks.
    for b in range(1, n_blocks):
        a = b * blen
        assert np.isclose(sess['early_conf'][a], sess['conflict'][a:a + 5].mean())
    assert set(np.unique(sess['early_class'][sess['reversal']])) <= {0, 1}
    assert np.all(sess['early_class'][~sess['reversal']] == -1)

    # select()
    assert select(sess, err=True).sum() == (~sess['correct']).sum()
    assert np.array_equal(select(sess, since=(1, 3)), since <= 3)
    assert select(sess, conf_level=[3, 4]).sum() == np.isin(sess['conf_level'], [3, 4]).sum()
    assert select(sess, stale=True, err=True).sum() == (exp_stale & ~sess['correct']).sum()
    print(f'OK  trial_labels + z_updates on a synthetic session: {n} trials, {n_blocks} blocks, '
          f"axis {np.round(sess['axis'], 3)}, d={sess['d']:.2f}")


if __name__ == '__main__':
    _self_test()
