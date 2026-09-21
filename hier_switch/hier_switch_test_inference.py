"""
hier_switch_test_inference.py — load a trained model and test its context inference under
different latent conditions.

Every condition runs on its own copy of the model, weights frozen, Z restarted from the
condition's starting gate, on the same test trials: the paper's 30-60-trial blocks, drawn
from the model's own seed on the test RNG stream.

    .venv/bin/python hier_switch/hier_switch_test_inference.py <model.pt> [condition ...]

With no conditions named, all of CONDITIONS run. Models come from run(save_model=True),
e.g. the v15 grid: exports/hier_switch/tune_v15/NG_s0/model.pt.

Results: exports/hier_switch/inference_tests/<model>/<condition>/ (summary.json, trials.npz,
session.npz, panels). A table is printed at the end. session.npz is what the phase-2
analyses read (hier_switch_analyses.load_session): the per-trial arrays plus the per-timestep
outputs, the pulse frames the model saw, the hidden states and the recovered dL/dZ.

Conditions
    softmax            as trained: softmax over the 2 Z units at the trained temperature,
                       the trained Z_lr and weight decay, Z from Z_init (the uniform gate).
    nosoftmax_zlr*     softmax off at test: raw Z *is* the gate.
                       - Z starts at [0.5, 0.5], the softmax of the uniform start, so the
                         first trial sees exactly the gate the weights were trained on.
                         Starting at Z = 0 would be the silent gate.
                       - Weight decay is off: L2 on raw Z pulls toward 0, which without the
                         softmax is the silent gate, not the middle.
                       - Z_lr is swept, because the gradient no longer passes through the
                         softmax.
                       Z can now move both units together — a gain direction the softmax
                       removed — so the table reports the gate sum alongside the usual
                       numbers.
    sigmoid_zlr30000_wd*  the sigmoid at its best Z_lr over a weight-decay ladder
                       (Z_lr x Z_decay = 0.03 / 0.09 / 0.3 / 0.9). Gain is a live axis only
                       without the softmax, so the gain analyses (B5) run here.
    *_rc_low / *_rc_high  the paper's controlled reversals: the first 5 trials of every
                       block forced to 7:2 or 6:3 conflict (config.reversal_conflict). The
                       unforced partner is *_rc_none; the three are paired trial by trial.
    rnn300_rc_*        the backprop baseline as the group uses it (v17 models, trained on
                       300-trial blocks): LU off, weights plastic at WU_lr 3e-3, on
                       250-350-trial blocks — the shortest on which a plastic RNN commits
                       and re-learns instead of hedging (docs/hier_switch_task.md, v17).
                       4500 trials, ~15 reversals. Never analysed with z_updates.
    rnn / rnn_rc_*     the v16 baseline on the paper's 30-60 blocks at its trained WU_lr:
                       the record of the hedge (undecided on 99.9 % of trials).
    sigmoid_zlr*       the softmax replaced by a per-unit sigmoid at test. Z starts at 0,
                       and sigmoid(0) = 0.5 is the same [0.5, 0.5] gate as the uniform
                       softmax start. Weight decay as trained: under the sigmoid it pulls
                       toward that middle gate, not toward silence. Z_lr is swept, because
                       at the same raw Z the sigmoid gate is ~4x less sensitive than the
                       softmax at temperature 0.5.

A one-page comparison, inference_summary.pdf, lands in the model's folder:
- accuracy by trials since a reversal;
- the gate's total gain over the session;
one line per condition.
"""

import copy
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

import matplotlib.pyplot as plt

import plot_style
from plot_style import FigSize
from hier_switch_analyses import save_session, session_arrays, session_meta
from hier_switch_train import load_model, run_test, report, plot_panels, block_window

NO_SOFTMAX = dict(latent_activation='none', Z_init=[0.5, 0.5], Z_decay=0.0)
SIGMOID = dict(latent_activation='sigmoid', Z_init=0.0)       # weight decay as trained
# The reversal-aligned analyses want reversals, not trials: 2000 trials is ~45 blocks.
LONG = dict(n_trials=2000)
# The RNN baseline at test: no latent update, weights plastic at 3e-3, 250-350-trial blocks.
RNN300 = dict(test_no_of_steps_in_latent_space=0, WU_lr=3e-3, block_len_range=(250, 350))


def _rc(name, base):
    """A condition and its two forced-early-conflict partners, on the same 2000 trials."""
    return {f'{name}_rc_{k}': dict(base, **LONG, reversal_conflict=v)
            for k, v in (('none', None), ('low', 'low'), ('high', 'high'))}


def _manip(name, base, perturb):
    """A manipulation on the forced low/high pair — the paper's design, which applies its
    optogenetics to reversals of a known early-conflict level and compares within animal."""
    return {f'{name}_rc_{k}': dict(base, **LONG, reversal_conflict=k, perturb=perturb)
            for k in ('low', 'high')}


CONDITIONS = {
    'softmax': dict(),
    **{f'nosoftmax_zlr{z:g}': dict(NO_SOFTMAX, Z_lr=float(z)) for z in (300, 1000, 3000, 10000)},
    **{f'sigmoid_zlr{z:g}': dict(SIGMOID, Z_lr=float(z)) for z in (1000, 3000, 10000, 30000, 1e5)},
    # Z_decay held at the trained 3e-6 above, so Z_lr * Z_decay also grows with Z_lr. The
    # ladder below holds Z_lr at its best value and moves the decay strength on its own:
    # Z_lr * Z_decay = 0.03 (wd1e-06), 0.09 (= sigmoid_zlr30000), 0.3, 0.9.
    'sigmoid_zlr30000_wd1e-06': dict(SIGMOID, Z_lr=3e4, Z_decay=1e-6),
    'sigmoid_zlr30000_wd1e-05': dict(SIGMOID, Z_lr=3e4, Z_decay=1e-5),
    'sigmoid_zlr30000_wd3e-05': dict(SIGMOID, Z_lr=3e4, Z_decay=3e-5),
    **_rc('softmax', dict()),
    # The sigmoid at its best setting, as a forced-conflict triplet: the only gate where the
    # gain is a live axis, so it is where "drive both units" is a gain move and not a reset.
    **_rc('sigmoid', dict(SIGMOID, Z_lr=3e4)),
    # ── The manipulations (hier_switch_hooks), each on the forced low/high pair ──
    # Every one acts on the trials just after a reversal and then stops, as the paper's
    # optogenetics does. `lu0` is ACC→MD silencing (Fig 4h): the gradient is still computed
    # and logged, only the step is zeroed. `lu3` / `lu10` are an "ACC stimulation" the paper
    # never performed — ours, and labelled as ours. `blast` drives both latent units and
    # lets the gradient take over, the MD-activation analogue (Fig 5d).
    **_manip('softmax_lu0', dict(), dict(kind='lu_scale', k=0.0, trials=[1, 4])),
    **_manip('softmax_lu3', dict(), dict(kind='lu_scale', k=3.0, trials=[1, 5])),
    **_manip('softmax_lu10', dict(), dict(kind='lu_scale', k=10.0, trials=[1, 5])),
    **_manip('softmax_blast', dict(), dict(kind='z_set', z=[1.0, 1.0], trials=[1, 1])),
    **_manip('sigmoid_blast', dict(SIGMOID, Z_lr=3e4),
             dict(kind='z_set', z=[1.0, 1.0], trials=[1, 1])),
    # Momentum: not a control for anything, a question. Z is persistent where the paper's MD
    # is transient, and the paper's ACC builds up over consecutive errors. Momentum is the
    # one change that would make the latent update build up the same way, so the panels show
    # what it does to Z, to the update and to the gradient around a reversal.
    **{f'softmax_mom{mu:g}_rc_none': dict(LONG, reversal_conflict=None,
                                          perturb=dict(kind='momentum', mu=mu, trials=[1, 5]))
       for mu in (0.5, 0.9)},
    # The RNN baseline (v16 models): no latent update, weights plastic, as it trained. On
    # the paper's 30-60 blocks it hedges at every weight learning rate (v17 tuning log).
    **_rc('rnn', dict(test_no_of_steps_in_latent_space=0)),
    'rnn': dict(test_no_of_steps_in_latent_space=0),
    # The RNN baseline the group uses (v17 models): weights plastic at WU_lr 3e-3 on
    # 250-350-trial blocks, where it perseverates, hedges and re-learns each block through
    # its weights (~80 trials to switch). 4500 trials for ~15 reversals; the forced
    # low/high pair as for NeuraGEM. n_trials is in run_test's own units (trials).
    **{f'rnn300_rc_{k}': dict(RNN300, n_trials=4500, reversal_conflict=v)
       for k, v in (('none', None), ('low', 'low'), ('high', 'high'))},
}
# Which conditions a model type runs when none are named.
DEFAULTS = dict(NG=[k for k in CONDITIONS if not k.startswith('rnn')],
                RNN=[k for k in CONDITIONS if k.startswith('rnn300')])


def gate_values(z, cfg):
    """The gate the model applied: softmax(z / T) under the softmax, raw z without it."""
    if cfg.latent_activation == 'softmax':
        e = np.exp((z - np.nanmax(z, axis=1, keepdims=True)) / cfg.softmax_temp)
        return e / e.sum(axis=1, keepdims=True)
    if cfg.latent_activation == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-z))
    return z


def test_condition(model, cfg, tag, name, overrides):
    logger, _, tcfg = run_test(model, cfg, run_name=f'inference_tests/{tag}/{name}',
                               record_hidden=True, **overrides)
    trials, _, te = report(logger, tcfg, label=f'{tag} {name}')
    plot_panels(logger, tcfg, filename='panels_full.pdf')
    win = block_window(trials, tcfg, 'Inference only', n_blocks=8, from_end=False)
    if win:
        plot_panels(logger, tcfg, *win, filename='panels_test.pdf')

    half = slice(trials['n'] // 2, None)          # second half: past the start-up transient
    gate = gate_values(trials['z_in'][half], tcfg)
    extra = dict(abs_decision=float(np.abs(trials['decision'][half]).mean()),
                 gate_mean=[round(float(v), 3) for v in np.nanmean(gate, axis=0)],
                 gate_sum=float(np.nanmean(gate.sum(axis=1))))
    with open(tcfg.export_path + 'summary.json', 'w') as f:
        json.dump(dict(test=te, extra=extra, condition=name, overrides=overrides), f,
                  indent=1, default=float)
    np.savez_compressed(tcfg.export_path + 'trials.npz',
                        **{k: v for k, v in trials.items() if isinstance(v, np.ndarray) and k != 'phase'},
                        phase=trials['phase'].astype(str))
    # session.npz: the superset every phase-2 analysis reads.
    meta = session_meta(tcfg, condition=name, overrides={k: v for k, v in overrides.items()},
                        model_path=getattr(cfg, '_model_path', ''), model_type=_model_type(cfg),
                        z_restart=True, tag=tag)
    save_session(tcfg.export_path + 'session.npz', trials, session_arrays(logger, tcfg), meta)
    return te, extra, trials


def _model_type(cfg):
    return str(getattr(cfg, 'experiment_to_run', '')).replace('hier_switch_', '') or 'NG'


FAMILIES = {                          # label, colour ramp; darker = higher Z_lr
    'softmax': ('softmax, as trained', None),
    'nosoftmax': ('no activation', plt.cm.Purples),
    'sigmoid': ('sigmoid', plt.cm.Oranges),
}


def _family(name):
    return name.split('_zlr')[0].split('_rc_')[0]


def _zlr_of(name):
    """The Z_lr a condition's name encodes, or NaN when it does not name one.

    Not every member of a family carries a `_zlr` tag: `sigmoid_rc_none` runs the family's
    default rate. Returning NaN rather than raising is what keeps those conditions from
    taking the whole summary figure down with them.
    """
    if '_zlr' not in name:
        return float('nan')
    try:
        return float(name.split('_zlr')[1].split('_')[0])
    except (IndexError, ValueError):
        return float('nan')


def _style(name, names):
    """Softmax in black; each other family on its own one-hue ramp ordered by Z_lr. Conditions
    that also change weight decay (a '_wd' suffix) are dashed. A condition of the family that
    names no Z_lr gets the ramp's full tone."""
    ls = '--' if '_wd' in name else '-'
    fam = _family(name)
    if fam == 'softmax':
        return 'k', ls
    cmap = FAMILIES[fam][1]
    z = _zlr_of(name)
    levels = sorted({v for v in (_zlr_of(n) for n in names if _family(n) == fam) if v == v})
    if not levels or z != z:
        return cmap(0.8), ls
    frac = levels.index(z) / max(1, len(levels) - 1)
    return cmap(0.4 + 0.55 * frac), ls


def summary_figure(rows, path):
    """One line per condition: accuracy by trials since a reversal, and the gate's total gain.

    rows: (name, trials, gate_sum_trace). The legend names families, not conditions: within a
    family, darker is a higher Z_lr, and dashed also changes weight decay.
    """
    from matplotlib.lines import Line2D
    names = [r[0] for r in rows]
    fig, (ax_a, ax_g) = plt.subplots(1, 2, figsize=FigSize.row(2))
    ks = np.arange(1, 21)
    for name, trials, gsum in rows:
        col, ls = _style(name, names)
        acc = [trials['correct'][trials['since'] == k].mean() for k in ks]
        ax_a.plot(ks, acc, color=col, linestyle=ls, linewidth=1.1)
        g = np.convolve(gsum, np.ones(20) / 20, mode='valid')
        ax_g.plot(np.arange(len(g)) + 10, g, color=col, linestyle=ls, linewidth=0.9)
    ax_a.axhline(0.5, color='k', linewidth=0.5, alpha=0.3)
    ax_a.set_xlabel('Trials since reversal')
    ax_a.set_ylabel('Accuracy')
    ax_a.set_ylim(0, 1.02)
    ax_g.axhline(1.0, color='k', linewidth=0.5, alpha=0.3)
    ax_g.set_xlabel('Test trial')
    ax_g.set_ylabel('Gate sum, MA(20)')
    ax_g.set_ylim(-0.5, 2.5)
    present = [f for f in FAMILIES if any(_family(n) == f for n in names)]
    handles = [Line2D([], [], color='k' if FAMILIES[f][1] is None else FAMILIES[f][1](0.8),
                      linewidth=1.2, label=FAMILIES[f][0]) for f in present]
    if any('_wd' in n for n in names):
        handles.append(Line2D([], [], color='0.4', linestyle='--', linewidth=1.2,
                              label='decay held at Z_lr·Z_decay = 0.03'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               frameon=False, handlelength=1.4, columnspacing=1.0)
    ax_g.text(1.0, 1.02, 'darker = higher Z_lr', transform=ax_g.transAxes, ha='right',
              va='bottom', fontsize=5.5, color='0.35')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    print(f'Exported: {path}')
    return fig


def rows_from_disk(tag, cfg):
    """Every condition of this model with results on disk, in CONDITIONS order."""
    root = os.path.join(_ROOT, 'exports', 'hier_switch', 'inference_tests', tag)
    rows = []
    for name, over in CONDITIONS.items():
        f = os.path.join(root, name, 'trials.npz')
        if not os.path.exists(f) or _family(name) not in FAMILIES:
            continue          # the rc_* and rnn conditions have their own figures
        d = dict(np.load(f, allow_pickle=True))
        c = copy.copy(cfg)
        c.latent_activation = over.get('latent_activation', cfg.latent_activation)
        rows.append((name, d, gate_values(d['z_in'], c).sum(axis=1)))
    return rows


def main(path, names):
    model, cfg = load_model(path)
    cfg._model_path = os.path.abspath(path)
    parts = os.path.normpath(path).split(os.sep)
    tag = '_'.join(parts[-3:-1])                  # e.g. tune_v15_NG_s0
    names = names or DEFAULTS.get(_model_type(cfg), list(CONDITIONS))
    plot_style.set_plot_style()
    rows = []
    root = os.path.join(_ROOT, 'exports', 'hier_switch', 'inference_tests', tag)
    for name in names:
        # Recording is the expensive part and a session is deterministic, so a condition
        # already on disk is left alone. Delete its folder (or pass force=1) to redo it.
        done = os.path.join(root, name, 'session.npz')
        if os.path.exists(done) and not os.environ.get('HIER_SWITCH_FORCE'):
            print(f'  = {name}: session.npz exists, skipping')
            continue
        te, extra, trials = test_condition(model, cfg, tag, name, CONDITIONS[name])
        rows.append((name, te, extra, trials))

    print(f'\n{tag}  (test sessions; see each condition for its blocks and what is plastic)')
    hdr = (f'{"condition":<20} {"acc":>5} {"steady":>6} {"s1":>5} {"s2":>5} {"s3":>5} '
           f'{"cross":>5} {"Z dp":>5} {"|dec|":>5}  gate (mean, sum)')
    print(hdr)
    print('-' * len(hdr))
    for name, te, ex, _ in rows:
        g = lambda k: te['acc_since'].get(k, float('nan'))
        print(f'{name:<20} {te["acc"]:5.3f} {g("11-1000"):6.3f} {g("1"):5.2f} {g("2"):5.2f} '
              f'{g("3"):5.2f} {str(te.get("cross_trial")):>5} {te.get("z_dprime", float("nan")):5.2f} '
              f'{ex["abs_decision"]:5.2f}  {ex["gate_mean"]}, {ex["gate_sum"]:.2f}')
    out = os.path.join(_ROOT, 'exports', 'hier_switch', 'inference_tests', tag)
    # The comparison figure covers the softmax / sigmoid / no-activation families only; a
    # model whose conditions are all outside them (the RNN baseline) has nothing to draw.
    disk_rows = rows_from_disk(tag, cfg)
    if disk_rows:
        summary_figure(disk_rows, os.path.join(out, 'inference_summary.pdf'))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
