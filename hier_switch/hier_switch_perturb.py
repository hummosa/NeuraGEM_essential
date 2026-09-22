"""
hier_switch_perturb.py — hold Z still and ask what it was doing.

The latent update is switched off and Z is clamped at a chosen raw value for a whole
session, so the same frozen weights run the same test trials under a grid of gates. Two
directions are separated:

* **contrast** — (z₀ − z₁)/2, which context the gate selects. Under the softmax it is the
  only live direction.
* **gain** — the mean of the two units. The softmax removes it (the gate sums to 1); under
  a sigmoid gate it is a real axis, and "MD activation = set Z to (1, 1)" is a move along it.

The questions (P7, and the user's "does Z's gain modulate cue integration"):
does a gate near the middle slow the rule code down and leave the cue code alone, and does
raising the gain change the integration index? Both are read out with the same functions
the unclamped sessions use (hier_switch_hidden.py), so the numbers are comparable.

No new plumbing is needed for this: `run_test` takes `Z_init` per unit and
`test_no_of_steps_in_latent_space=0` freezes Z where it starts.

    .venv/bin/python hier_switch/hier_switch_perturb.py <model.pt> softmax|sigmoid [n_trials]

Each cell writes a session (exports/hier_switch/clamp/<tag>/<activation>_m<..>_d<..>/) that
load_session reads like any other, plus one clamp_grid.json holding the metrics table.
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
import matplotlib
matplotlib.use('Agg')

from hier_switch_analyses import (PRIMARY, STEADY, load_session, save_session, select,
                                  session_arrays, session_meta, trial_labels)
from hier_switch_hidden import decode, fit_axes, integration
from hier_switch_train import load_model, run_test
from hier_switch_analyses import extract_trials

# Raw Z = (m + d, m − d): m is the gain (unit mean), d the contrast. Under the softmax only
# d matters; the sigmoid grid moves both.
# Mostly d ≥ 0, since ±d differ only in which context the gate selects and every metric is
# reported split by whether it matches the block; one negative cell per activation checks
# that symmetry rather than assuming it.
GRIDS = {
    'softmax': dict(m=[0.0], d=[0.0, 0.25, 0.5, 1.0, -0.5]),
    'sigmoid': dict(m=[-1.0, 0.0, 1.0, 2.0], d=[0.0, 0.5, 1.0, 2.0, -1.0]),
}


def clamp_cells(activation, grid=None):
    g = grid or GRIDS[activation]
    return [(m, d) for m in g['m'] for d in g['d']]


def cell_name(activation, m, d):
    return f'{activation}_m{m:+.2f}_d{d:+.2f}'.replace('.', 'p')


def run_cell(model, cfg, tag, activation, m, d, n_trials=1000, root=None):
    """One clamp cell: weights frozen, LU off, Z held at (m + d, m − d) for the session."""
    name = cell_name(activation, m, d)
    run_name = os.path.join('clamp', tag, name)
    logger, _, tcfg = run_test(model, cfg, run_name=run_name, n_trials=n_trials,
                               latent_activation=activation, Z_init=[m + d, m - d],
                               test_no_of_steps_in_latent_space=0, record_hidden=True)
    trials = extract_trials(logger, tcfg)
    meta = session_meta(tcfg, condition=name, model_type='NG', z_restart=True, tag=tag,
                        clamp=dict(activation=activation, m=m, d=d),
                        overrides=dict(latent_activation=activation, Z_init=[m + d, m - d]))
    arrays = session_arrays(logger, tcfg)
    arrays['clamped'] = np.ones(trials['n'], bool)
    save_session(tcfg.export_path + 'session.npz', trials, arrays, meta)
    return tcfg.export_path


def cell_metrics(path, ref_axis=None):
    """Behaviour and integration for one clamped session, split by whether the clamped gate
    matches the block's context.

    **Which context a clamped gate selects is read off behaviour**, as the context the model
    is more accurate in: `acc_match` is that accuracy and `acc_mismatch` the other one. Both
    are reported raw as `acc_ctx`, so the choice is always visible. Geometry was tried first
    and does not work — the gain direction takes a sigmoid cell off the context axis, and
    the raw middle gate (0.5, 0.5) makes the model behave as context 0 rather than sitting
    between the two. The cost of picking the larger of two numbers is a small upward bias
    on a cell that is at chance in both contexts (~0.02 with 500 steady trials each), which
    is why `acc_ctx` is there and why the chance cells read ~0.5 rather than 0.5 exactly.
    `side_geom` keeps the nearest-prototype answer for comparison.
    """
    sess = trial_labels(load_session(path))
    clamp = sess['meta']['clamp']
    z = sess['z'][0].astype(float)
    side_geom = None
    if ref_axis is not None and ref_axis.get('m0') is not None:
        side_geom = int(np.linalg.norm(z - np.asarray(ref_axis['m1']))
                        < np.linalg.norm(z - np.asarray(ref_axis['m0'])))
    m = select(sess, phase=PRIMARY)
    steady = m & (sess['since'] >= STEADY)
    acc_ctx = [float(sess['correct'][steady & (sess['context'] == c)].mean()) for c in (0, 1)]
    side = int(np.argmax(acc_ctx))
    match = sess['context'] == side
    res = dict(**{k: float(v) for k, v in clamp.items() if k != 'activation'},
               activation=clamp['activation'], side=side, side_geom=side_geom,
               n=int(m.sum()),
               acc=float(sess['correct'][m].mean()), acc_ctx=acc_ctx,
               acc_match=float(sess['correct'][steady & match].mean()),
               acc_mismatch=float(sess['correct'][steady & ~match].mean()),
               abs_decision=float(np.abs(sess['decision'][m]).mean()),
               undecided=float((~sess['decided'][m]).mean()),
               rt=float(sess['rt'][m].mean()),
               rt_decided=float(sess['rt'][m & sess['decided']].mean()))
    axes = fit_axes(sess, steady)
    res['integration'] = integration(sess, axes, steady)
    res['integration_match'] = integration(sess, axes, steady & match)
    dec = decode(sess, mask=steady)
    res['decoding'] = {k: dict(buildup=v['buildup'], acc16=v['acc'][16], acc24=v['acc'][-1])
                       for k, v in dec.items()}
    res['gate'] = np.round(sess['gate_in'][-1], 4).tolist()
    return res


def clamp_grid(model_path, activation='softmax', n_trials=1000, grid=None, reference=None):
    """Run the whole grid for one activation and write clamp_grid.json next to the cells."""
    model, cfg = load_model(model_path)
    cfg._model_path = os.path.abspath(model_path)
    parts = os.path.normpath(model_path).split(os.sep)
    tag = '_'.join(parts[-3:-1])
    ref_axis = _reference_axis(reference)
    rows = []
    for m, d in clamp_cells(activation, grid):
        path = run_cell(model, cfg, tag, activation, m, d, n_trials=n_trials)
        rows.append(cell_metrics(path, ref_axis))
        r = rows[-1]
        print(f"  {cell_name(activation, m, d):24s} acc {r['acc']:.3f} "
              f"(match {r['acc_match']:.3f} / mismatch {r['acc_mismatch']:.3f})  "
              f"|dec| {r['abs_decision']:.2f}  undecided {r['undecided']:.2f}  "
              f"RT {r['rt']:.2f}  index {r['integration']['index']:.2f}  "
              f"vel {r['integration']['cue_velocity']:.3f}")
    out = os.path.join(_ROOT, 'exports', 'hier_switch', 'clamp', tag,
                       f'clamp_grid_{activation}.json')
    with open(out, 'w') as f:
        json.dump(dict(model=model_path, activation=activation, n_trials=n_trials,
                       reference=reference, cells=rows), f, indent=1, default=float)
    print(f'Exported: {out}')
    return rows


def _reference_axis(reference):
    """The unclamped session's context axis and its two prototypes, or None."""
    if not reference or not os.path.exists(reference):
        return None
    ref = trial_labels(load_session(reference))
    if ref['axis'] is None:
        return None
    return dict(axis=ref['axis'], mid=ref['mid'],
                m0=ref.get('m0').tolist(), m1=ref.get('m1').tolist())


def level_tags(root):
    """The clamp tags belonging to this level's selected seeds, in seed order.

    `clamp/` accumulates: a re-selection that drops a seed leaves its cells on disk, and a
    second noise level adds a whole set. Listing the directory and taking everything would
    then pool dropped seeds, and eventually two noise levels, into one panel.
    """
    from hier_switch_group import NG_SEEDS, NG_TAG
    want = [f'tune_{NG_TAG}_NG_s{s}' for s in NG_SEEDS]
    return [t for t in want if os.path.isdir(os.path.join(root, t))]


def recompute(tag=None, activation=None):
    """Rebuild clamp_grid_*.json from the saved sessions, without re-running any model."""
    root = os.path.join(_ROOT, 'exports', 'hier_switch', 'clamp')
    for t in ([tag] if tag else level_tags(root)):
        for act in ([activation] if activation else ('softmax', 'sigmoid')):
            cells = [c for c in clamp_cells(act)]
            paths = [os.path.join(root, t, cell_name(act, m, d)) for m, d in cells]
            paths = [p for p in paths if os.path.exists(os.path.join(p, 'session.npz'))]
            if not paths:
                continue
            ref = _reference_axis(reference_for(t, act))
            rows = [cell_metrics(p, ref) for p in paths]
            out = os.path.join(root, t, f'clamp_grid_{act}.json')
            with open(out, 'w') as f:
                json.dump(dict(model=t, activation=act, n_trials=rows[0]['n'],
                               reference=reference_for(t, act), cells=rows), f,
                          indent=1, default=float)
            print(f'Rebuilt {out} ({len(rows)} cells)')


def tasks():
    """[(seed, activation)] — one SLURM array task per cell block."""
    from hier_switch_group import NG_SEEDS
    return [(s, a) for s in NG_SEEDS for a in ('softmax', 'sigmoid')]


def reference_for(tag, activation):
    name = {'softmax': 'softmax_rc_none', 'sigmoid': 'sigmoid_zlr30000'}[activation]
    return os.path.join(_ROOT, 'exports', 'hier_switch', 'inference_tests', tag, name)


def run_task(index=None, n_trials=1000):
    from hier_switch_group import LEVEL, NG_TAG, check_level
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0) if index is None else index)
    seed, act = tasks()[index]
    path = os.path.join(_ROOT, 'exports', 'hier_switch', f'tune_{NG_TAG}', f'NG_s{seed}',
                        'model.pt')
    tag = f'tune_{NG_TAG}_NG_s{seed}'
    sigma = check_level(path)
    print(f'--- clamp task {index}: level {LEVEL}, {tag}, {act}, trained at noise {sigma}')
    clamp_grid(path, act, n_trials=n_trials, reference=reference_for(tag, act))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'list'
    if mode == 'list':
        for i, (s, a) in enumerate(tasks()):
            print(i, f'seed {s}', a, f'{len(clamp_cells(a))} cells')
    elif mode == 'recompute':
        recompute(*sys.argv[2:])
    elif mode == 'task':
        run_task(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:                                   # <model.pt> softmax|sigmoid [n_trials]
        path, act = sys.argv[1], sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        tag = '_'.join(os.path.normpath(path).split(os.sep)[-3:-1])
        clamp_grid(path, act, n_trials=n, reference=reference_for(tag, act))
