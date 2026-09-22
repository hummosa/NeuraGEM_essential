"""Is the noise-0.6 RNN baseline's hedging its test weight learning rate, or the task?

`RNN300` fixes WU_lr at 3e-3 and 250-350-trial blocks (hier_switch_test_inference), tuned
at pulse_noise_std 0.5 as the shortest blocks on which a plastic RNN commits instead of
hedging. At 0.6 the baseline is undecided on 92 % of trials and only 2 of 20 seeds clear
the learner bar. That is either a fact about the task at this noise, or an artefact of a
knob that was never retuned.

This runs one trained seed's test session at several WU_lr and reports what each does. It
is a **diagnostic** — it says whether the rate is the cause. Adopting a different rate is
a second change alongside the noise and needs a decision, not a script.

    .venv/bin/python hier_switch/rnn_wu_probe.py [tag] [seed] [lr ...]
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from hier_switch_analyses import block_scale
from hier_switch_test_inference import RNN300
from hier_switch_train import load_model, report, run_test


def probe(tag='v19', seed=6, lrs=(1e-3, 3e-3, 1e-2), n_trials=4500):
    path = os.path.join(_ROOT, 'exports', 'hier_switch', f'tune_{tag}', f'RNN_s{seed}',
                        'model.pt')
    model, cfg = load_model(path)
    print(f'{tag} RNN_s{seed}, trained at pulse_noise_std {cfg.pulse_noise_std}\n')
    print(f'{"WU_lr":>8} {"acc":>7} {"steady":>7} {"undecided":>10} {"|dec|":>7}')
    rows = []
    for lr in lrs:
        over = dict(RNN300, n_trials=n_trials)
        over['WU_lr'] = float(lr)
        logger, _, tcfg = run_test(model, cfg, run_name=f'scratch/wu_probe/{tag}_s{seed}_lr{lr:g}',
                                   record_hidden=False, **over)
        trials, _, te = report(logger, tcfg, label=f'WU_lr {lr:g}')
        lo, _, _ = block_scale(dict(block_len_range=list(tcfg.block_len_range)))
        steady = te['acc_since'].get(f'{lo + 1}-1000', te['acc_since'].get('11-1000'))
        dec = np.abs(trials['decision'])
        und = float((dec < tcfg.rt_threshold).mean())
        print(f'{lr:8.0e} {te["acc"]:7.3f} {float(steady):7.3f} {und:10.3f} '
              f'{float(dec.mean()):7.3f}')
        rows.append((lr, te['acc'], steady, und))
    return rows


if __name__ == '__main__':
    a = sys.argv[1:]
    tag = a[0] if a else 'v19'
    seed = int(a[1]) if len(a) > 1 else 6
    lrs = [float(x) for x in a[2:]] or (1e-3, 3e-3, 1e-2)
    probe(tag, seed, lrs)
