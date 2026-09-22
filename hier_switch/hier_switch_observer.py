"""
hier_switch_observer.py — the ideal observer for the hierarchical cue→rule reversal task.

It is the normative reference for three things the model is judged against:
  * the accuracy ceiling at each conflict level (P1), i.e. how much of the error is the
    task's own sensory noise rather than the model;
  * the context belief a reversal should produce (B1), and how fast it should switch (P2);
  * the size of the latent update each trial deserves (B5's normative comparison): the
    feedback's log-odds update of the context, which is exactly the paper's "an error on an
    ambiguous cue is weak evidence" statement in its normative form.

It runs on a recorded session (hier_switch_analyses.load_session), which carries the noisy
pulse frames the model actually saw, so observer and model see the same trials.

Two stages per trial.

1. **Cue posterior from the pulses.** Every pulse frame is a one-hot [HP, LP, WN] plus
   Gaussian noise of s.d. `pulse_noise_std` on all three channels, so a frame's likelihood
   under type k is N(x_t; e_k, σ²I). The frame types are exchangeable within a trial, and
   the observer does not know the conflict level, so it marginalises over the five levels:

       P(x | cue) = Σ_ℓ (1/5) Π_t Σ_k π_k(cue, ℓ) N(x_t; e_k),
       π(cue=HP, ℓ) = (dom, non, 7) / 16 on (HP, LP, WN)

   q = P(HP | x) with a flat prior over the cue. This treats the frames as independent
   draws from the level's mixture; the generator instead shuffles a fixed multiset (dom of
   one type, non of the other, 7 WN), so the likelihood is slightly misspecified. It is the
   version the tuning log used: 0.923 accuracy at σ = 0.5, against 0.890 for a plain HP−LP
   count, and it is the ceiling quoted in docs/hier_switch_task.md. The misspecification is
   visible at zero noise: with 5 HP and 4 LP frames seen, the exact model knows the cue with
   certainty while this one returns 5/4 odds (q = 0.56). It always reads the cue *correctly*
   there, so the ceiling is unaffected; only the confidence at high conflict is understated.

2. **Context filter.** Contexts reverse with hazard h ≈ 1/45 (blocks are 30-60 trials), so
   before the trial the predictive belief is

       b_t = (1 − h)·p_{t−1} + h·(1 − p_{t−1}).

   The only feedback is the correct side, and correct = vis × cue_sign × ctx_sign, so under
   context c the feedback says the cue was `correct × vis × ctx_sign(c)`:

       P(correct side | c, x) = q if that product is +1 (HP) else 1 − q,

   which gives the posterior p_t. The two context likelihoods sum to 1, so the feedback's
   log-odds update is ±log(q / (1 − q)) — large when the cue was unambiguous, small when it
   was not. That is the normative form of the conflict weighting B5 measures in the model.

The observer's own choice uses the predictive belief b_t, before it sees the feedback:
pick the side with the larger Σ_c b_t(c)·P(cue = side × vis × ctx_sign(c) | x).

    .venv/bin/python hier_switch/hier_switch_observer.py            # self-check
    .venv/bin/python hier_switch/hier_switch_observer.py <session>  # run it on a session
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from hier_switch_analyses import PRIMARY, load_session, trial_labels, _blocks, _first_confirmed

HAZARD = 1.0 / 45.0          # mean block length of the paper's 30-60 range

#: The accuracy ceiling per `pulse_noise_std`: how often the ideal observer reads the cue
#: correctly. Selection criteria are expressed as a fraction of this (hier_switch_tune.arms,
#: hier_switch_figures.LEARNED), so that a seed is judged against what the noise level
#: allows rather than against a constant that silently gets stricter as the noise rises.
#:
#: These are **fixed, documented numbers, not recomputed per run** — the estimate moves by
#: ~0.007 between sample sizes (σ=0.5 gives 0.930 / 0.927 / 0.923 at 2000 / 4000 / 20000
#: trials), and a criterion that drifts between runs is worse than one that is slightly off.
#: Reproduce with `ceiling(sigma, n_trials=20000)` below; env_seed 0, data_stream 1.
CEILING = {0.5: 0.9231, 0.6: 0.8944, 0.7: 0.8696, 0.8: 0.8461, 0.9: 0.8235}


def ceiling(sigma, n_trials=20000):
    """Measure the ceiling at one noise level: a fresh test stream, no model, no training.

    This is how the CEILING table above was produced. It is deliberately not called at
    selection time — see the note there.
    """
    from hier_switch_config import HierSwitchConfig
    from hier_switch_dataset import HierSwitchDataset

    cfg = HierSwitchConfig()
    cfg.no_of_blocks, cfg.data_stream, cfg.env_seed = int(n_trials), 1, 0
    cfg.pulse_noise_std = float(sigma)
    counts = [tuple(c) for c in cfg.conflict_counts]
    x = np.asarray(HierSwitchDataset(cfg).data_sequence).reshape(-1, cfg.trial_len, cfg.input_size)
    pulses, cue = x[:, :cfg.n_pulses, :3], x[:, 0, cfg.ch('cue')]
    conf = np.round(x[:, 0, cfg.ch('conflict')], 3)
    ll = cue_log_likelihood(pulses, counts, float(sigma), cfg.n_pulses)
    q = 1.0 / (1.0 + np.exp(ll[:, 1] - ll[:, 0]))
    ok = np.where(q >= 0.5, 1.0, -1.0) == cue
    per = {float(v): float(ok[conf == v].mean()) for v in np.unique(conf)}
    return float(ok.mean()), per


def cue_log_likelihood(pulses, counts, sigma, n_pulses=None):
    """log P(pulse frames | cue = HP) and | cue = LP), marginalised over conflict level.

    pulses: (n, n_pulses, 3). counts: [(dom, non), ...]. Returns (n, 2) = (HP, LP).
    """
    pulses = np.asarray(pulses, dtype=float)
    n, n_frames, _ = pulses.shape
    n_pulses = n_pulses or n_frames
    # log N(x_t; e_k, σ²I) up to a constant shared by all k: −|x − e_k|² / 2σ²
    d2 = ((pulses[:, :, None, :] - np.eye(3)[None, None, :, :]) ** 2).sum(-1)
    logN = -d2 / (2.0 * sigma ** 2)                              # (n, frames, 3)
    n_wn = n_pulses - sum(counts[0])
    out = np.zeros((n, 2))
    for cue in (0, 1):                                           # 0 = HP dominant, 1 = LP
        per_level = []
        for dom, non in counts:
            w = np.zeros(3)
            w[cue], w[1 - cue], w[2] = dom, non, n_wn
            w = w / w.sum()
            with np.errstate(divide='ignore'):
                lw = np.log(w)
            frame = _logsumexp(logN + lw[None, None, :], axis=2)  # (n, frames)
            per_level.append(frame.sum(axis=1))
        out[:, cue] = _logsumexp(np.stack(per_level, axis=1), axis=1) - np.log(len(counts))
    return out


def _logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.exp(a - m).sum(axis=axis))


def ideal_observer(sess, hazard=HAZARD, primary=PRIMARY):
    """Run the observer over a recorded session. Adds nothing to `sess`; returns a dict.

    q        P(cue = HP | pulses), from the frames alone
    p_cue    P(the observer read the cue correctly) = q if the cue was HP else 1 − q
    b, p     context belief before / after this trial's feedback, as P(context = 1)
    dlogit   the feedback's log-odds update of the context, signed toward context 1
    dlogit_other  the same update signed toward the context the belief did *not* favour:
             the normative counterpart of the model's `s` (B5)
    choice, correct   the observer's own response and whether it was right
    read_cue how the observer read the cue (±1), and `read_ok` whether that was right
    p_c      the fraction of trials the cue was read correctly, per conflict level: the
             accuracy ceiling, and the paper's P_corrcue in ε_CW. `p_cue_mean` is the mean
             posterior instead, which is lower — the observer is not certain when it is right
    b_true, p_true   the belief in the *true* context, for the reversal-aligned comparison
    switch_trial      per block, the paper's criterion applied to the observer's choices
    acc, acc_given_context   overall, and with the true context handed to it
    """
    if 'pulses' not in sess:
        raise ValueError('the observer needs the pulse frames: record the session first '
                         '(hier_switch_test_inference.py writes session.npz)')
    meta = sess['meta']
    counts = [tuple(c) for c in meta['conflict_counts']]
    sigma = float(meta['pulse_noise_std'])
    ll = cue_log_likelihood(sess['pulses'], counts, sigma, meta['n_pulses'])
    q = 1.0 / (1.0 + np.exp(ll[:, 1] - ll[:, 0]))                # P(cue = HP | x)
    cue_hp = sess['cue'] > 0
    p_cue = np.where(cue_hp, q, 1 - q)

    n = sess['n']
    # The correct side is the task's three-way parity, so it is in the saved labels.
    vis = sess['vis']
    corr_side = vis * sess['cue'] * np.where(sess['context'] == 0, 1.0, -1.0)
    # P(feedback | context c): the feedback says the cue was correct × vis × ctx_sign(c).
    need_hp = {c: (corr_side * vis * (1 if c == 0 else -1)) > 0 for c in (0, 1)}
    lik = {c: np.where(need_hp[c], q, 1 - q) for c in (0, 1)}

    b = np.zeros(n)                              # P(context = 1) before the feedback
    p = np.zeros(n)                              # ... after it
    prev = 0.5
    phase_new = np.r_[True, sess['phase'][1:] != sess['phase'][:-1]]
    for k in range(n):
        if phase_new[k]:
            prev = 0.5                           # a phase starts a fresh stream of trials
        b[k] = (1 - hazard) * prev + hazard * (1 - prev)
        post = b[k] * lik[1][k]
        p[k] = post / (post + (1 - b[k]) * lik[0][k])
        prev = p[k]

    with np.errstate(divide='ignore'):
        dlogit = np.log(lik[1]) - np.log(lik[0])
    dlogit_other = np.where(b > 0.5, -dlogit, dlogit)      # toward the less-believed context

    # The observer's choice, from the predictive belief only.
    score = {}
    for side in (1.0, -1.0):
        s = np.zeros(n)
        for c, pc in ((0, 1 - b), (1, b)):
            hp = (side * vis * (1 if c == 0 else -1)) > 0
            s += pc * np.where(hp, q, 1 - q)
        score[side] = s
    choice = np.where(score[1.0] >= score[-1.0], 1.0, -1.0)
    correct = choice == corr_side
    # With the context handed over, the observer just reads the cue.
    read_cue = np.where(q >= 0.5, 1.0, -1.0)
    read_ok = read_cue == sess['cue']
    given = (read_cue * vis * np.where(sess['context'] == 0, 1.0, -1.0)) == corr_side
    true1 = sess['context'] == 1

    m = sess['phase'] == primary
    cell = lambda v, c: v[m & (sess['conf_level'] == c)]
    res = dict(q=q, p_cue=p_cue, b=b, p=p, dlogit=dlogit, dlogit_other=dlogit_other,
               choice=choice, correct=correct, read_cue=read_cue, read_ok=read_ok,
               b_true=np.where(true1, b, 1 - b), p_true=np.where(true1, p, 1 - p),
               acc=float(correct[m].mean()), acc_given_context=float(given[m].mean()),
               p_c=[float(cell(read_ok, c).mean()) if len(cell(read_ok, c)) else np.nan
                    for c in range(len(counts))],
               p_cue_mean=[float(cell(p_cue, c).mean()) if len(cell(p_cue, c)) else np.nan
                           for c in range(len(counts))])
    res['switch_trial'] = _observer_switches(sess, correct, primary)
    return res


def _observer_switches(sess, correct, primary):
    """The paper's switch criterion on the observer's own choices, per reversal block."""
    _, starts, ends = _blocks(sess)
    keep = sess['reversal'] & ~sess['transient'] & (sess['phase'] == primary)
    out = np.full(sess['n'], np.nan)
    for a, e in zip(starts, ends):
        if keep[a]:
            out[a:e] = _first_confirmed(correct[a:e])
    return out


def _self_check():
    """σ = 0 makes the cue certain; every σ in CEILING reproduces its documented ceiling.

    The ceilings are checked at 2000 trials against the 20000-trial table, so the 0.01
    tolerance is doing real work: it is the sampling spread, not slack.
    """
    from hier_switch_config import HierSwitchConfig
    from hier_switch_dataset import HierSwitchDataset

    cfg = HierSwitchConfig()
    cfg.no_of_blocks, cfg.data_stream = 2000, 1
    cfg.env_seed = 0
    cfg.pulse_noise_std = 1e-3
    counts = [tuple(c) for c in cfg.conflict_counts]
    x = np.asarray(HierSwitchDataset(cfg).data_sequence).reshape(-1, cfg.trial_len, cfg.input_size)
    cue = x[:, 0, cfg.ch('cue')]
    conf = np.round(x[:, 0, cfg.ch('conflict')], 3)
    ll = cue_log_likelihood(x[:, :cfg.n_pulses, :3], counts, cfg.pulse_noise_std, cfg.n_pulses)
    q = 1.0 / (1.0 + np.exp(ll[:, 1] - ll[:, 0]))
    # Noiseless: the cue is always read correctly. The frame-wise likelihood is not certain
    # about it at high conflict (see the module docstring), but it is right.
    assert float((np.where(q >= 0.5, 1.0, -1.0) == cue).mean()) == 1.0, 'noiseless cue misread'
    assert (q[conf == 0] > 0.99).all() or (q[conf == 0] < 0.01).any(), 'conflict 0 unsure'
    print('OK  sigma=0.001: cue read correctly 1.000')

    for sigma, expect in sorted(CEILING.items()):
        acc, per = ceiling(sigma, n_trials=2000)
        assert abs(acc - expect) < 0.01, f'ceiling at sigma={sigma}: {acc:.4f}, table says {expect}'
        print(f'OK  sigma={sigma}: cue read correctly {acc:.3f} (table {expect})  by conflict '
              f'{ {k: round(v, 3) for k, v in per.items()} }')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        s = trial_labels(load_session(sys.argv[1]))
        obs = ideal_observer(s)
        print(f"observer: acc {obs['acc']:.3f}  given the context {obs['acc_given_context']:.3f}  "
              f"P(cue correct) by conflict {np.round(obs['p_c'], 3)}")
        m = s['phase'] == PRIMARY
        print(f"model:    acc {s['correct'][m].mean():.3f}  "
              f"switch: observer {np.nanmean(obs['switch_trial']):.2f} vs "
              f"model {np.nanmean(s['switch_trial'][m]):.2f}")
    else:
        _self_check()
