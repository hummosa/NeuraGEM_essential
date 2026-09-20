"""
hier_switch_dataset.py — trial generator for the hierarchical cue→rule reversal task.

See hier_switch_config.py for the task, the trial layout and the channel map. Importing
this module registers the dataset as DATASET_REGISTRY['hier_switch'].

Run it directly for a self-check of the generator (no training):

    .venv/bin/python hier_switch/hier_switch_dataset.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from datasets import BaseTaskDataset, DATASET_REGISTRY

HP, LP, WN = 0, 1, 2

# Forced early-reversal conflict levels, the paper's low and high (7:2 and 6:3). The config
# carries them too; the getattr fallback keeps configs pickled before the knob existed usable.
REVERSAL_CONFLICT_LEVELS = {'low': (7, 2), 'high': (6, 3)}


class HierSwitchDataset(BaseTaskDataset):
    """
    One base-class block is one trial (config.block_size = trial_len), so the session has
    config.no_of_blocks trials. The context schedule is generated here: blocks of
    block_len_range trials, alternating between context 0 and context 1.

    Context 0: HP → attend vision, LP → attend audition. Context 1: the reverse.
    context_ids and hlcids both carry the context (0/1) per timestep — context_ids for the
    oracle-Z condition, hlcids for the block shading in plot_logger_panels.
    """

    def generate_sequences(self):
        cfg = self.config
        # train_model's passive phase (Phase 1, weights only, Z held at Z_init) builds its
        # dataset with _allow_latent_updates False. That phase is passive_n_blocks equal blocks
        # of alternating context, drawn from its own stream (data_stream + 2) so it neither
        # replays the Phase-2 trials nor needs a schedule of its own.
        passive = not getattr(cfg, '_allow_latent_updates', True)
        stream = int(getattr(cfg, 'data_stream', 0)) + (2 if passive else 0)
        # Keyed on (env_seed, stream) so the test phase gets its own trials.
        rng = np.random.default_rng([int(cfg.env_seed), stream])
        self.rng = rng

        n_trials = sum(self.block_sizes) // cfg.trial_len
        if passive:
            first = cfg.first_context
            ctx0 = int(rng.integers(2)) if first is None else int(first)
            n_blk = max(1, int(getattr(cfg, 'passive_n_blocks', 1)))
            edges = np.linspace(0, n_trials, n_blk + 1).astype(int)
            contexts = []
            for b in range(n_blk):
                contexts.extend([(ctx0 + b) % 2] * int(edges[b + 1] - edges[b]))
        else:
            contexts = self._context_schedule(n_trials, rng)

        counts = [tuple(c) for c in cfg.conflict_counts]
        p = cfg.p_conflict
        n_wn = cfg.n_pulses - cfg.n_informative
        L, onset = cfg.trial_len, cfg.target_onset
        n_ch = cfg.input_size

        # Forced early-reversal conflict (test streams only): trials 1..n of every block.
        forced = getattr(cfg, 'reversal_conflict', None)
        if forced is not None and stream != 0 and not passive:
            levels = getattr(cfg, 'reversal_conflict_levels', REVERSAL_CONFLICT_LEVELS)
            forced = tuple(levels[forced])
            n_forced = int(getattr(cfg, 'reversal_conflict_n', 5))
        else:
            forced = None
        since = 0

        data = np.zeros((n_trials * L, n_ch), dtype=np.float32)
        ctx_seq = np.repeat(np.asarray(contexts, dtype=np.float32), L)

        for k in range(n_trials):
            ctx      = contexts[k]
            ctx_sign = 1.0 if ctx == 0 else -1.0
            cue_sign = 1.0 if rng.random() < 0.5 else -1.0         # +1 = HP dominant
            dom, non = counts[rng.choice(len(counts), p=p)]
            since = 1 if k == 0 or contexts[k] != contexts[k - 1] else since + 1
            if forced is not None and since <= n_forced:
                dom, non = forced          # drawn above anyway, so the stream is unchanged
            conflict = non / dom

            dominant, other = (HP, LP) if cue_sign > 0 else (LP, HP)
            pulses = np.array([dominant] * dom + [other] * non + [WN] * n_wn)
            rng.shuffle(pulses)

            vis = 1.0 if rng.random() < 0.5 else -1.0
            aud = -vis
            correct = vis * cue_sign * ctx_sign

            trial = data[k * L:(k + 1) * L]
            trial[np.arange(cfg.n_pulses), pulses] = 1.0
            trial[:cfg.n_pulses, :3] += rng.normal(0.0, cfg.pulse_noise_std, (cfg.n_pulses, 3))
            trial[onset:, cfg.ch('vis')] = vis
            trial[onset:, cfg.ch('aud')] = aud
            trial[:, cfg.ch('cue')]      = cue_sign
            trial[:, cfg.ch('conflict')] = conflict
            trial[:, cfg.ch('context')]  = ctx
            trial[:, cfg.ch('correct')]  = correct

        return list(data), list(ctx_seq), list(ctx_seq)

    def _context_schedule(self, n_trials, rng):
        """Per-trial context: alternating blocks with lengths uniform in a (lo, hi) range.

        The range is block_len_range, except in the training stream (data_stream 0) while a
        train_block_schedule segment is active: [(until_trial, (lo, hi)), ...] gives the
        range for blocks that *start* before until_trial. After the last segment, and in the
        test stream always, block_len_range applies.
        """
        cfg = self.config
        schedule = getattr(cfg, 'train_block_schedule', None)
        if int(getattr(cfg, 'data_stream', 0)) != 0:
            schedule = None
        first = cfg.first_context
        ctx = int(rng.integers(2)) if first is None else int(first)
        out = []
        while len(out) < n_trials:
            lo, hi = cfg.block_len_range
            for until, rng_range in (schedule or []):
                if len(out) < until:
                    lo, hi = rng_range
                    break
            out.extend([ctx] * int(rng.integers(lo, hi + 1)))
            ctx = 1 - ctx
        return out[:n_trials]


DATASET_REGISTRY['hier_switch'] = HierSwitchDataset


if __name__ == '__main__':
    from hier_switch_config import HierSwitchConfig

    cfg = HierSwitchConfig()
    cfg.pulse_noise_std = 0.0          # noiseless, so the pulse counts can be read back
    cfg.train_block_schedule = None    # paper blocks from trial 1; the schedule is checked below
    cfg.no_of_blocks = 2000
    ds = HierSwitchDataset(cfg)
    x = np.asarray(ds.data_sequence).reshape(-1, cfg.trial_len, cfg.input_size)
    lab = x[:, 0]
    cue, conf, ctx, corr = (lab[:, cfg.ch(c)] for c in ('cue', 'conflict', 'context', 'correct'))
    vis = x[:, -1, cfg.ch('vis')]

    n_hp = x[:, :cfg.n_pulses, HP].sum(1)
    n_lp = x[:, :cfg.n_pulses, LP].sum(1)
    n_wn = x[:, :cfg.n_pulses, WN].sum(1)
    assert np.all(np.sign(n_hp - n_lp) == cue), 'dominant pulse type disagrees with cue'
    assert np.allclose(np.minimum(n_hp, n_lp) / np.maximum(n_hp, n_lp), conf), 'conflict mismatch'
    assert np.all(n_wn == cfg.n_pulses - cfg.n_informative)
    assert np.all(corr == vis * cue * np.where(ctx == 0, 1, -1)), 'correct side != vis*cue*ctx'
    assert np.all(x[:, :cfg.target_onset, cfg.ch('vis')] == 0), 'target visible before onset'

    runs = np.split(ctx, np.flatnonzero(np.diff(ctx)) + 1)
    lens = np.array([len(r) for r in runs[:-1]])          # the last block is truncated
    lo, hi = cfg.block_len_range
    assert np.all((lens >= lo) & (lens <= hi)), f'block length outside {cfg.block_len_range}'
    assert all(r[0] != s[0] for r, s in zip(runs[:-1], runs[1:])), 'contexts do not alternate'

    print(f'OK  {len(x)} trials, {len(runs)} blocks, block length {lens.min()}-{lens.max()}, '
          f'P(cue=HP)={np.mean(cue > 0):.2f}, conflict levels {np.unique(np.round(conf, 3))}, '
          f'trial_len={cfg.trial_len}, target_onset={cfg.target_onset}')

    # The default training schedule, and the test stream ignoring it.
    cfg = HierSwitchConfig()
    cfg.no_of_blocks = 6000
    def lens():
        c = np.asarray(HierSwitchDataset(cfg).llcid_sequence)[::cfg.trial_len]
        return [len(r) for r in np.split(c, np.flatnonzero(np.diff(c)) + 1)]

    train_lens = lens()
    start = 0
    for n in train_lens[:-1]:                      # each block within its segment's range
        lo, hi = cfg.block_len_range
        for until, r in cfg.train_block_schedule:
            if start < until:
                lo, hi = r
                break
        assert lo <= n <= hi, f'block at trial {start} has length {n}, expected {lo}-{hi}'
        start += n
    cfg.data_stream = 1
    test_lens = lens()
    assert max(test_lens[:-1]) <= cfg.block_len_range[1], 'test stream used the training schedule'
    print(f'OK  schedule {cfg.train_block_schedule}: training blocks {train_lens[:4]}..., '
          f'test blocks {test_lens[:5]}...')

    # Forced early-reversal conflict: trials 1-5 of every test block at the forced level,
    # every other trial identical to the unforced session; the training stream untouched.
    def session(stream, forced):
        cfg = HierSwitchConfig()
        cfg.no_of_blocks, cfg.data_stream, cfg.reversal_conflict = 1000, stream, forced
        return np.asarray(HierSwitchDataset(cfg).data_sequence).reshape(-1, cfg.trial_len, cfg.input_size)
    for stream in (0, 1):
        base = session(stream, None)
        ctx = base[:, 0, cfg.ch('context')]
        since = np.ones(len(ctx), dtype=int)
        for k in range(1, len(ctx)):
            since[k] = 1 if ctx[k] != ctx[k - 1] else since[k - 1] + 1
        early = since <= cfg.reversal_conflict_n
        for forced, level in (('low', 2 / 7), ('high', 3 / 6)):
            x = session(stream, forced)
            if stream == 0:
                assert np.array_equal(x, base), 'reversal_conflict changed the training stream'
                continue
            conf = x[:, 0, cfg.ch('conflict')]
            assert np.allclose(conf[early], level), f'{forced}: early trials not at {level:.3f}'
            assert np.array_equal(x[~early], base[~early]), f'{forced}: later trials changed'
            for c in ('cue', 'context', 'correct', 'vis'):
                assert np.array_equal(x[:, -1, cfg.ch(c)], base[:, -1, cfg.ch(c)]), c
    print(f'OK  reversal_conflict: trials 1-{cfg.reversal_conflict_n} forced to 7:2 / 6:3 in the '
          f'test stream, everything else unchanged; training stream untouched')
