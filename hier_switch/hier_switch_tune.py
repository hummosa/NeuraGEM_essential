"""
hier_switch_tune.py — tuning grids for NG on the hierarchical switching task.

One grid entry per SLURM array task (or run them all in sequence locally). Each run writes
its summary numbers, per-trial arrays and the three panel figures to
exports/hier_switch/tune_<TAG>/<name>/.

    .venv/bin/python hier_switch/hier_switch_tune.py list  <TAG>     # print the grid
    SLURM_ARRAY_TASK_ID=3 .venv/bin/python hier_switch/hier_switch_tune.py run <TAG>
    .venv/bin/python hier_switch/hier_switch_tune.py collect <TAG>   # table of results
    .venv/bin/python hier_switch/hier_switch_tune.py arms <TAG>      # seeds grouped by setting
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use('Agg')

from hier_switch_train import run, run_test, report, plot_panels, block_window


def _ng(zlr, **kw):
    return dict(model='NG', Z_lr=float(zlr), **kw)


POST = dict(pre_gating=False, post_gating=True)
PRE = dict(pre_gating=True, post_gating=False)


def _oracle_tests(decay_fracs, steeps=(0, 2), zlrs=(300, 1000, 3000, 10000)):
    """Test settings for run_test: steepness x Z_lr at each per-trial decay fraction."""
    return [dict(label=f's{st}_zlr{z:g}_df{df:g}', Z_lr=float(z),
                 exponential_increase_steepness=[st], Z_decay=df / z)
            for df in decay_fracs for st in steeps for z in zlrs]

_FIRST3000 = [(3000, (3000, 3000)), (10**9, (200, 200))]

GRIDS = {
    # v1: Z_lr scale and gating site. 8000 training trials, 1500 test.
    'v1': (
        [dict(name='Oracle', model='Oracle'), dict(name='RNN', model='RNN')]
        + [dict(name=f'NG_pre_zlr{z:g}', **_ng(z)) for z in (1e3, 3e3, 1e4, 3e4, 1e5)]
        + [dict(name=f'NG_post_zlr{z:g}', **_ng(z, **POST)) for z in (1e3, 3e3, 1e4, 3e4, 1e5)]
    ),
    # v2: v1 learned nothing (everything at chance, Z never separated the contexts). Add the
    # block-length curriculum: one 3000-trial block, blocks of 200-300 until trial 6000, then
    # the paper's 30-60 until 12000.
    'v2': (
        [dict(name='Oracle', model='Oracle', Z_lr=1e4), dict(name='RNN', model='RNN')]
        + [dict(name=f'NG_pre_zlr{z:g}', **_ng(z)) for z in (1e3, 3e3, 1e4, 3e4)]
        + [dict(name=f'NG_post_zlr{z:g}', **_ng(z, **POST)) for z in (1e3, 3e3, 1e4, 3e4)]
    ),
    # v3: learnability only — can NG learn the task inside a long block at all? v2 showed
    # the RNN learning its 3000-trial first block (0.89) while every NG run stayed at chance:
    # the latent update drives raw Z (the gate) to ~0 before the weights have learned
    # anything, which silences the network. Long blocks throughout (3000, then 1000s).
    #   RNN                     reference: learns within a block by weights alone
    #   NG_none_zlr10 / 100     slow Z: does learning survive LU if Z barely moves?
    #   NG_none_passive_zlr*    3000 weights-only trials first (Z held), then LU on
    #   NG_sig_zlr*             sigmoid gate, Z_init 0 (= gate 0.5): no zero, no sign flip
    'v3': (
        [dict(name='RNN', model='RNN')]
        + [dict(name=f'NG_none_zlr{z:g}', **_ng(z)) for z in (10, 100)]
        + [dict(name=f'NG_none_passive_zlr{z:g}', **_ng(z), n_passive_trials=3000,
                n_train_trials=9000, train_block_schedule=[(10**9, (1000, 1000))])
           for z in (100, 1000)]
        + [dict(name=f'NG_sig_zlr{z:g}', **_ng(z), latent_activation='sigmoid', Z_init=0.0)
           for z in (100, 1000, 10000)]
    ),
    # v4: softmax Z, blocks cut to 20% (600, then 200s), 8000 training trials — now the
    # config defaults. Oracle trains once with the true context as Z, then the oracle is
    # removed and Z is optimised at each test Z_lr on a copy of the same weights: that
    # brackets the latent learning rate that works when the weights are known to use Z.
    # NG_*_first3000 keeps v3's long first block, in case 600 trials is too short to learn in.
    'v4': (
        [dict(name='Oracle', model='Oracle', test_Z_lrs=(100, 300, 1000, 3000, 1e4, 3e4)),
         dict(name='RNN', model='RNN')]
        + [dict(name=f'NG_sm_zlr{z:g}', **_ng(z)) for z in (300, 1000, 3000, 1e4)]
        + [dict(name='NG_sm_zlr1000_first3000', **_ng(1000),
                train_block_schedule=[(3000, (3000, 3000)), (10**9, (200, 200))])]
    ),
    # v5: v4 learned nothing with a 600-trial first block — not even the Oracle, which v1
    # showed learning in ~2000 trials with no activation. The difference is the gate: a
    # softmax at temperature 1 turns the oracle one-hot into [0.73, 0.27] vs [0.27, 0.73],
    # and the weights cannot use that (Oracle curve = RNN curve). Sharpen the softmax:
    #   Oracle_T*            temp 0.5 / 0.25 / 0.1, each then tested at a Z_lr sweep; lower
    #                        temp scales the gradient through the softmax by 1/temp, so the
    #                        sweep starts lower than v4's
    #   RNN_first2000        reference with a first block long enough to learn in (v3: ~1800)
    #   NG_T0.25_*           temp 0.25 x Z_lr x first block 600 (as asked) vs 2000
    'v5': (
        [dict(name=f'Oracle_T{t:g}', model='Oracle', softmax_temp=t,
              test_Z_lrs=(10, 30, 100, 300, 1000, 3000, 1e4)) for t in (0.5, 0.25, 0.1)]
        + [dict(name='RNN_first2000', model='RNN',
                train_block_schedule=[(2000, (2000, 2000)), (10**9, (200, 200))])]
        + [dict(name=f'NG_T0.25_zlr{z:g}_first{f}', **_ng(z), softmax_temp=0.25,
                train_block_schedule=[(f, (f, f)), (10**9, (200, 200))])
           for f in (600, 2000) for z in (100, 300, 1000, 3000)]
    ),
    # v6: discovery with the Oracle-bracketed latent (temp 0.5, Z_lr ~1000). v5's NG settled
    # Z in one softmax corner and kept it there in BOTH contexts: the other corner's gate was
    # never trained, so a reversal gives Z no reason to go there, and the weights did all the
    # re-learning until they hedged. Here: a 3000-trial first block (the length after which
    # the RNN could re-learn 200-300-trial blocks in v2; 600 and 2000 left every non-oracle
    # model at chance), then 200s. Z_dim 4 gives spare gate patterns for a new context to
    # claim without un-training the old one.
    'v6': (
        [dict(name='RNN_first3000', model='RNN')]
        + [dict(name=f'NG_T0.5_Zd{d}_zlr{z:g}', **_ng(z), latent_dims=[d])
           for d in (2, 4) for z in (300, 1000, 3000)]
    ),
    # v7: Oracle only. Training never updates the Oracle's Z, so steepness and decay act only
    # at test: one training per gating site, then the whole latent grid on copies of it.
    # Steepness 0 (uniform weight over the trial's timesteps) vs 2 (the old default).
    # Decay is given as the fraction of Z removed per trial, df = Z_lr * Z_decay: the SGD
    # step is Z <- (1 - Z_lr*Z_decay) Z - Z_lr*grad, so a literal Z_decay of 0.3 at Z_lr 1000
    # would multiply Z by -299 each trial. df 0.3 is a leak with a ~3-trial memory.
    # Pre vs post gating both trained (the config edit left pre_gating=True, the default every
    # run since v3 used). Each (gating, df) job retrains the same seed, i.e. the same weights.
    'v7': [dict(name=f'Oracle_{g}_df{df:g}', model='Oracle', **G, tests=_oracle_tests([df]))
           for g, G in (('pre', PRE), ('post', POST)) for df in (0, 0.03, 0.1, 0.3)],
    # v8: the v7 Oracle settings imported to NG (discovery from scratch): pre-gating,
    # steepness 0, temp 0.5, weight decay on Z (df = Z_lr * Z_decay in the run names). Config-default schedule (1200 trials
    # of 600-blocks, then 200s). RNN on the same schedule for reference; one NG arm keeps a
    # 3000-trial first block, the only start v6 learned from.
    'v8': (
        [dict(name='RNN', model='RNN')]
        + [dict(name=f'NG_zlr{z:g}_df{df:g}', **_ng(z), Z_decay=df / z)
           for z in (1000, 3000) for df in (0.03, 0.1, 0.3)]
        + [dict(name='NG_zlr3000_df0.1_first3000', **_ng(3000), Z_decay=0.1 / 3000,
                train_block_schedule=[(3000, (3000, 3000)), (10**9, (200, 200))])]
    ),
    # v9: discovery with plain weight decay on Z, one value for the whole run.
    # v6 (no decay) moved Z but kept it in one corner for both contexts: with Z uniform in
    # the first block, both gate populations learn mapping A, so after a reversal moving Z
    # changes nothing. An off-centre Z_init makes block 1 mostly one population's. Crossed
    # with Z_decay 0 / 1e-5 / 3.3e-5 (the last is the Oracle's best at Z_lr 3000). 3000-trial
    # first block, then 200s. One arm on the short default schedule, and an RNN reference.
    'v9': (
        [dict(name='RNN_first3000', model='RNN', train_block_schedule=_FIRST3000)]
        + [dict(name=f'NG_init{a:g}_decay{d:.1e}', **_ng(3000), Z_init=[a, -a], Z_decay=d,
                train_block_schedule=_FIRST3000)
           for a in (0, 0.5) for d in (0.0, 1e-5, 0.1 / 3000)]
        + [dict(name='NG_init0.5_decay1.0e-05_short', **_ng(3000), Z_init=[0.5, -0.5], Z_decay=1e-5)]
    ),
    # v10: is v9's discovery (NG_init0.5_decay1e-05, seed 0) robust? Seeds 0-4 at that
    # setting, the same seeds with Z_init at the middle (does the offset matter, or was seed
    # 0 lucky?), and weight decay 5e-6 / 2e-5 at the offset, seeds 0-2.
    'v10': (
        [dict(name=f'NG_init{a:g}_decay1e-05_s{sd}', **_ng(3000), Z_init=[a, -a], Z_decay=1e-5, seed=sd)
         for a in (0.5, 0) for sd in range(5)]
        + [dict(name=f'NG_init0.5_decay{d:g}_s{sd}', **_ng(3000), Z_init=[0.5, -0.5], Z_decay=d, seed=sd)
           for d in (5e-6, 2e-5) for sd in range(3)]
    ),
    # v11: v10 discovered in 2/5 seeds (init 0.5, decay 1e-5). Failing seeds either never
    # learned the 3000-trial first block (s2), or learned it and then hedged before Z split
    # (s1, s3): once |decision| is small the Z gradient vanishes and Z stays at the middle.
    # The winners split Z within the first 2-3 reversals. Three levers on that race, 5 seeds
    # each, all init 0.5 and Z_lr * Z_decay = 0.03:
    #   slowW    WU_lr 5e-4 (default 1e-3): weights hedge more slowly
    #   fastZ    Z_lr 1e4, Z_decay 3e-6: Z splits faster
    #   first4k  4000-trial first block, 10000 trials: time for slow learners
    'v11': (
        [dict(name=f'slowW_s{sd}', **_ng(3000), Z_init=[0.5, -0.5], Z_decay=1e-5, WU_lr=5e-4, seed=sd)
         for sd in range(5)]
        + [dict(name=f'fastZ_s{sd}', **_ng(1e4), Z_init=[0.5, -0.5], Z_decay=3e-6, seed=sd)
           for sd in range(5)]
        + [dict(name=f'first4k_s{sd}', **_ng(3000), Z_init=[0.5, -0.5], Z_decay=1e-5, seed=sd,
                n_train_trials=10000, train_block_schedule=[(4000, (4000, 4000)), (10**9, (200, 200))])
           for sd in range(5)]
    ),
    # v12: passive phase instead of the long first block (config defaults now): 3000 trials,
    # weights only, Z held at Z_init, one 1500-trial block of each context; then 5000 active
    # trials on 200-trial blocks. Early errors carry no context information, so letting Z
    # follow them drags it to uniform before the weights have anything for it to gate.
    # Z_lr {3000, 1e4} x Z_lr*Z_decay {0.01, 0.03, 0.1}, off-centre Z_init, seeds 0-5; the
    # centred Z_init at Z_lr 1e4 / 0.03 (does passive training make the offset unnecessary?);
    # one RNN for reference.
    'v12': (
        [dict(name=f'zlr{z:g}_wd{p:g}_s{sd}', **_ng(z), Z_decay=p / z, Z_init=[0.5, -0.5], seed=sd)
         for z in (3000, 1e4) for p in (0.01, 0.03, 0.1) for sd in range(6)]
        + [dict(name=f'zlr1e4_wd0.03_init0_s{sd}', **_ng(1e4), Z_decay=0.03 / 1e4, Z_init=[0, 0], seed=sd)
           for sd in range(6)]
        + [dict(name='RNN', model='RNN')]
    ),
    # v13: the config defaults — passive 2 x 2000 (v12's 2 x 1500 was too short for most
    # seeds to learn the task), Z_lr 1e4, Z_decay 3e-6, uniform Z_init, active 200-trial
    # blocks. 10 seeds: how many converge?
    'v13': [dict(name=f'NG_s{sd}', model='NG', seed=sd) for sd in range(10)],
    # v14: v13 with WU_lr 3e-3 (default 1e-3) in both phases. v13's three non-converging
    # seeds never learned the task in the passive phase; does faster weight learning get
    # them there, and does it cost discovery once the weights can chase the reversals?
    'v14': [dict(name=f'NG_wu3e-3_s{sd}', model='NG', seed=sd, WU_lr=3e-3) for sd in range(10)],
    # v15: the v13 seeds that discovered (0, 1, 3, 5, 6, 9), retrained with v13's phase lengths
    # (passive 2 x 2000, active 5000) and saved as model.pt, for inference tests on trained
    # models (hier_switch_test_inference.py). Each seed is deterministic, so a retrained seed
    # should reproduce its v13 numbers.
    'v15': [dict(name=f'NG_s{sd}', model='NG', seed=sd, n_passive_trials=4000,
                 n_train_trials=5000, save_model=True) for sd in (0, 1, 3, 5, 6, 9)],
    # v16: the backprop baseline for the phase-2 analyses. RNN (LU off, Z at the uniform gate)
    # at v13's phase lengths, 10 seeds, saved. The RNN's test phase has plastic weights, so
    # the model is saved *before* it (a `tests` entry turns train_model's own test off) and
    # the test session runs on a copy. LU stays off at test: run_test would otherwise turn
    # it on for a model trained without it.
    'v16': [dict(name=f'RNN_s{sd}', model='RNN', seed=sd, n_passive_trials=4000,
                 n_train_trials=5000, save_model=True,
                 tests=[dict(label='test', test_no_of_steps_in_latent_space=0)])
            for sd in range(10)],
    # v17: the RNN baseline that behaves. v16 hedges: on 200-trial active blocks its output
    # collapses (|decision| 0.05) and no test-time weight learning rate rescues it on the
    # paper's 30-60 blocks (1e-3..3e-2 all undecided on 98-100 % of trials; 1e-1 thrashes).
    # Trained on 300-trial active blocks it re-learns each block through its weights and
    # enters the test committed; tested on 250-350-trial blocks with WU_lr 3e-3 it
    # perseverates, hedges and re-learns (~80 trials to switch, 0.89 late-block accuracy on
    # seed 0). Same passive phase and trial count as v16; only the block length differs.
    'v17': [dict(name=f'RNN_s{sd}', model='RNN', seed=sd, n_passive_trials=4000,
                 n_train_trials=5000, train_block_schedule=[(10**9, (300, 300))],
                 save_model=True,
                 tests=[dict(label='test', test_no_of_steps_in_latent_space=0, WU_lr=3e-3,
                             block_len_range=(250, 350))])
            for sd in range(10)],
}
_CURRICULUM_1 = [(3000, (3000, 3000)), (6000, (200, 300))]
_LONG_BLOCKS = [(3000, (3000, 3000)), (10**9, (1000, 1000))]
COMMON = {'v1': dict(n_test_trials=1500, n_train_trials=8000),
          'v2': dict(n_test_trials=1500, n_train_trials=12000,
                     train_block_schedule=_CURRICULUM_1),
          'v3': dict(n_test_trials=1000, n_train_trials=12000,
                     train_block_schedule=_LONG_BLOCKS),
          'v4': dict(n_test_trials=1000, n_train_trials=8000),
          'v5': dict(n_test_trials=1000, n_train_trials=8000),
          'v6': dict(n_test_trials=1000, n_train_trials=8000, softmax_temp=0.5,
                     train_block_schedule=[(3000, (3000, 3000)), (10**9, (200, 200))]),
          'v7': dict(n_test_trials=1000, n_train_trials=8000),
          'v8': dict(n_test_trials=1000, n_train_trials=8000),
          'v9': dict(n_test_trials=1000, n_train_trials=8000),
          'v10': dict(n_test_trials=1000, n_train_trials=8000, train_block_schedule=_FIRST3000),
          'v11': dict(n_test_trials=1000, n_train_trials=8000, train_block_schedule=_FIRST3000),
          'v12': dict(n_test_trials=1000),
          'v13': dict(n_test_trials=1000),
          'v14': dict(n_test_trials=1000),
          'v15': dict(n_test_trials=1000),
          'v16': dict(n_test_trials=1000),
          'v17': dict(n_test_trials=3000)}       # ~10 of its 250-350-trial blocks


def grid(tag):
    out = []
    for entry in GRIDS[tag]:
        e = dict(COMMON.get(tag, {}))
        e.update(entry)
        e.setdefault('seed', 0)
        out.append(e)
    return out


def _save(logger, cfg, name, info, res_tr=None):
    """Summary numbers, per-trial arrays and panel figures for one logger."""
    trials, tr, te = report(logger, cfg, label=name)
    res_tr = tr if tr is not None else res_tr
    plot_panels(logger, cfg, filename='panels_full.pdf')
    for phase, from_end, fn in (('Learning and inference', True, 'panels_train.pdf'),
                                ('Inference only', False, 'panels_test.pdf')):
        win = block_window(trials, cfg, phase, n_blocks=8, from_end=from_end)
        if win:
            plot_panels(logger, cfg, *win, filename=fn)
    with open(cfg.export_path + 'summary.json', 'w') as f:
        json.dump(dict(train=res_tr, test=te, entry=info), f, indent=1, default=float)
    np.savez_compressed(cfg.export_path + 'trials.npz',
                        **{k: v for k, v in trials.items() if isinstance(v, np.ndarray) and k != 'phase'},
                        phase=trials['phase'].astype(str))
    return res_tr


def run_entry(tag, entry):
    entry = dict(entry)
    name, model, seed = entry.pop('name'), entry.pop('model'), entry.pop('seed')
    tests = _tests(entry)
    entry.pop('test_Z_lrs', None)
    entry.pop('tests', None)
    save_model = entry.pop('save_model', False)
    run_name = f'tune_{tag}/{name}'
    info = dict(name=name, model=model, seed=seed, **entry)
    logger, trained, cfg = run(model, seed, run_test_phase=not tests, run_name=run_name,
                               save_model=save_model, **entry)
    res_tr = _save(logger, cfg, name, info)
    # One training, several test sessions: each on a copy of the trained weights, Z
    # optimised from the uninformed gate with its own latent settings.
    for t in tests:
        t = dict(t)
        label = t.pop('label')
        tlog, _, tcfg = run_test(trained, cfg, run_name=f'{run_name}/{label}', **t)
        _save(tlog, tcfg, f'{name} {label}', dict(info, test=label, **t), res_tr=res_tr)


def _tests(entry):
    """An entry's test sessions as [dict(label=..., **run_test kwargs)]."""
    if entry.get('tests'):
        return list(entry['tests'])
    return [dict(label=f'test_zlr{z:g}', Z_lr=z) for z in (entry.get('test_Z_lrs') or [])]


def collect(tag):
    root = os.path.join(os.path.dirname(_HERE), 'exports', 'hier_switch', f'tune_{tag}')
    rows = []
    for e in grid(tag):
        p = os.path.join(root, e['name'], 'summary.json')
        if not os.path.exists(p):
            rows.append((e['name'], None))
            continue
        with open(p) as f:
            rows.append((e['name'], json.load(f)))
        for t in _tests(e):
            q = os.path.join(root, e['name'], t['label'], 'summary.json')
            label = '  ' + t['label']
            if os.path.exists(q):
                with open(q) as f:
                    rows.append((label, json.load(f)))
            else:
                rows.append((label, None))
    hdr = (f'{"name":<18} | {"tr acc":>6} {"tr 11+":>6} | {"te acc":>6} {"te 11+":>6} '
           f'{"s1":>5} {"s2":>5} {"s3":>5} {"s4-5":>5} {"s6-10":>5} | {"Z dp":>5} '
           f'{"Zs1":>5} {"Zs2":>5} {"Zs3":>5} {"Zs4-5":>5} | {"cross":>5}')
    print(hdr)
    print('-' * len(hdr))
    for name, s in rows:
        if s is None:
            print(f'{name:<18} | (missing)')
            continue
        tr, te = s['train'], s['test']
        g = lambda d, k: d.get(k, float('nan')) if d else float('nan')
        if te is None:
            print(f'{name:<18} | {tr["acc"]:6.3f} {g(tr["acc_since"], "11-1000"):6.3f} | (no test)')
            continue
        zs = te.get('z_acc_since', {})
        print(f'{name:<18} | {tr["acc"]:6.3f} {g(tr["acc_since"], "11-1000"):6.3f} | '
              f'{te["acc"]:6.3f} {g(te["acc_since"], "11-1000"):6.3f} '
              + ' '.join(f'{g(te["acc_since"], k):5.2f}' for k in ('1', '2', '3', '4-5', '6-10'))
              + f' | {te.get("z_dprime", float("nan")):5.2f} '
              + ' '.join(f'{g(zs, k):5.2f}' for k in ('1', '2', '3', '4-5'))
              + f' | {str(te.get("cross_trial", "")):>5}')


def arms(tag):
    """Seeds grouped by setting (run name minus its _s<seed> suffix).

    discovered   test steady-state (trials 11+) >= 0.8 and Z separates the contexts (d' > 1.5)
    passive_ok   the passive phase reached >= 0.75 in its last block, i.e. the task itself
                 was learned; a seed that fails here never had a chance to discover anything
    """
    import re
    from collections import defaultdict
    from hier_switch_analyses import block_table
    root = os.path.join(os.path.dirname(_HERE), 'exports', 'hier_switch', f'tune_{tag}')
    groups = defaultdict(list)
    for e in grid(tag):
        p = os.path.join(root, e['name'], 'summary.json')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            te = json.load(f)['test']
        d = dict(np.load(os.path.join(root, e['name'], 'trials.npz'), allow_pickle=True))
        d['n'] = len(d['correct'])
        pb = block_table(d, 'no inference learning')
        passive_ok = bool(pb) and pb[-1]['acc_late'] >= 0.75
        found = te['acc_since'].get('11-1000', 0) >= 0.8 and te.get('z_dprime', 0) > 1.5
        groups[re.sub(r'_s\d+$', '', e['name'])].append(
            dict(found=found, passive_ok=passive_ok, acc=te['acc'],
                 steady=te['acc_since'].get('11-1000'), cross=te.get('cross_trial')))
    print(f'{"arm":<26} {"discovered":>10} {"passive ok":>10} | among discovered: '
          f'{"test":>5} {"steady":>6} {"cross":>5}')
    for arm, rs in groups.items():
        hit = [r for r in rs if r['found']]
        m = lambda k: np.mean([r[k] for r in hit]) if hit else float('nan')
        print(f'{arm:<26} {len(hit):>4} / {len(rs):<3} {sum(r["passive_ok"] for r in rs):>4} / {len(rs):<3} | '
              f'{"":17}{m("acc"):5.2f} {m("steady"):6.2f} {m("cross"):5.1f}')


if __name__ == '__main__':
    mode, tag = sys.argv[1], sys.argv[2]
    if mode == 'list':
        for i, e in enumerate(grid(tag)):
            print(i, e)
    elif mode == 'run':
        task = os.environ.get('SLURM_ARRAY_TASK_ID')
        entries = grid(tag)
        for i in ([int(task)] if task is not None else range(len(entries))):
            run_entry(tag, entries[i])
    elif mode == 'collect':
        collect(tag)
    elif mode == 'arms':
        arms(tag)
    else:
        raise SystemExit(f'unknown mode {mode!r}: list | run | collect')
