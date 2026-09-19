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
panels). A table is printed at the end.

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
from hier_switch_train import load_model, run_test, report, plot_panels, block_window

NO_SOFTMAX = dict(latent_activation='none', Z_init=[0.5, 0.5], Z_decay=0.0)
SIGMOID = dict(latent_activation='sigmoid', Z_init=0.0)       # weight decay as trained

CONDITIONS = {
    'softmax': dict(),
    **{f'nosoftmax_zlr{z:g}': dict(NO_SOFTMAX, Z_lr=float(z)) for z in (300, 1000, 3000, 10000)},
    **{f'sigmoid_zlr{z:g}': dict(SIGMOID, Z_lr=float(z)) for z in (1000, 3000, 10000, 30000, 1e5)},
    # Z_decay held at the trained 3e-6 above, so Z_lr * Z_decay also grows with Z_lr. This
    # one holds it at the trained 0.03 instead, to tell Z_lr apart from decay strength.
    'sigmoid_zlr30000_wd1e-06': dict(SIGMOID, Z_lr=3e4, Z_decay=1e-6),
}


def gate_values(z, cfg):
    """The gate the model applied: softmax(z / T) under the softmax, raw z without it."""
    if cfg.latent_activation == 'softmax':
        e = np.exp((z - np.nanmax(z, axis=1, keepdims=True)) / cfg.softmax_temp)
        return e / e.sum(axis=1, keepdims=True)
    if cfg.latent_activation == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-z))
    return z


def test_condition(model, cfg, tag, name, overrides):
    logger, _, tcfg = run_test(model, cfg, run_name=f'inference_tests/{tag}/{name}', **overrides)
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
    return te, extra, trials


FAMILIES = {                          # label, colour ramp; darker = higher Z_lr
    'softmax': ('softmax, as trained', None),
    'nosoftmax': ('no activation', plt.cm.Purples),
    'sigmoid': ('sigmoid', plt.cm.Oranges),
}


def _family(name):
    return name.split('_zlr')[0]


def _style(name, names):
    """Softmax in black; each other family on its own one-hue ramp ordered by Z_lr. Conditions
    that also change weight decay (a '_wd' suffix) are dashed."""
    ls = '--' if '_wd' in name else '-'
    fam = _family(name)
    if fam == 'softmax':
        return 'k', ls
    zlr = lambda n: float(n.split('_zlr')[1].split('_')[0])
    levels = sorted({zlr(n) for n in names if _family(n) == fam})
    frac = levels.index(zlr(name)) / max(1, len(levels) - 1)
    return FAMILIES[fam][1](0.4 + 0.55 * frac), ls


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
        if not os.path.exists(f):
            continue
        d = dict(np.load(f, allow_pickle=True))
        c = copy.copy(cfg)
        c.latent_activation = over.get('latent_activation', cfg.latent_activation)
        rows.append((name, d, gate_values(d['z_in'], c).sum(axis=1)))
    return rows


def main(path, names):
    model, cfg = load_model(path)
    parts = os.path.normpath(path).split(os.sep)
    tag = '_'.join(parts[-3:-1])                  # e.g. tune_v15_NG_s0
    plot_style.set_plot_style()
    rows = []
    for name in names:
        te, extra, trials = test_condition(model, cfg, tag, name, CONDITIONS[name])
        rows.append((name, te, extra, trials))

    print(f'\n{tag}  (test: paper blocks, weights frozen)')
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
    summary_figure(rows_from_disk(tag, cfg), os.path.join(out, 'inference_summary.pdf'))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:] or list(CONDITIONS))
