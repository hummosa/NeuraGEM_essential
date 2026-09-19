"""
hier_switch_analyses.py — per-trial extraction and summary numbers for the hierarchical
cue→rule reversal task. Used by hier_switch_train.py to judge whether a run has found the
context structure.

Z convention (same as flanker, docs/flanker_task.md convention 2): the logged Z already
contains the trial's own latent update, so the state a trial *used* is the previous trial's
logged Z — `z_in`.
"""

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
