"""
rotation_curriculum_rdm.py — How a context representation *develops* across S1 training.

The companion analysis to the fMRI study: one pattern per trial, a representational
dissimilarity matrix (RDM) over trials sampled across the whole of Stage 1, and a summary
index read off that matrix as a function of training. Every step after "one pattern per
trial" is metric-only, so it transfers to voxels unchanged — which is the point of running
it on the model first.

Why a checkerboard is the prediction
------------------------------------
`train_rotations = [0, 60]` with `rotation_block_order='random_no_repeat'` and two values
means blocks **strictly alternate** (the config says so at
`configs.py`: "with only two rotations the two are the same thing"). Order the sampled
trials chronologically and a model that holds context has low dissimilarity within a block,
high across adjacent blocks — a checkerboard. A model that does not is flat.

Cue frames, not outcome frames
------------------------------
`frames='cue'` by default. At the outcome frame the attack (x, y) is in the input, so
rotation is readable from the *current stimulus* and a checkerboard there is partly
stimulus-driven. At the cue frame only the colour one-hot is present, so rotation has to
come from held context — Z, or the recurrence. `RDM_FRAMES_CONTRAST` runs both, and the gap
between them is a result in its own right: it says the human design must keep the cue and
outcome periods separable.

Windows, and why they are mini-block aligned
--------------------------------------------
A mini-block is `n_colors = 5` trials and contains every colour exactly once. Hidden
activity depends on the current colour, so a window that is not a whole mini-block carries a
colour imbalance into the matrix. Two windows per block:

    early  the 2nd mini-block — past the post-switch transient, context still fresh
    late   the last complete mini-block — settled

Trials are ordered **by colour within each window**, so any residual colour structure shows
up as a fine 5-periodic pattern, visually distinct from the block-scale checkerboard.

The index, and the drift problem
--------------------------------
`context_modulation` returns two numbers per sliding window of `WINDOW_BLOCKS` consecutive
blocks. Both exclude same-block pairs, because trials in one block are temporally adjacent
and would be close for reasons that have nothing to do with context.

    cmi     (mean between-context − mean within-context) / mean overall, over cross-block
            pairs. This is what an fMRI paper reports, and what transfers directly.

    cmi_dc  a drift-orthogonal version. Under strict alternation *same context is exactly
            even block lag*, so any context contrast is also a lag-parity contrast, and a
            slow drift in activity across training is a smooth function of lag that lands in
            the same place. `cmi_dc = ((d1 + d3)/2 − d2) / overall` is the parity contrast at
            block lags 1, 2, 3; because (1+3)/2 = 2 it cancels any *linear* trend in lag by
            construction, and suppresses smooth ones. `self_test`'s drift-only case is what
            pins this down: `cmi` may be non-zero there, `cmi_dc` must not be.

Read `cmi` as the headline and `cmi_dc` as the control. Where they disagree, the drift is
real and `cmi_dc` is the one to believe.

Guardrails
----------
`build_rdm` z-scores features across the **whole sampled trial set, pooled**. Keep it that
way: z-scoring within block or within window would subtract out the context offset itself.
Nothing here mean-centres per block, and nothing regresses time out before building the RDM
— drift is handled by the local window and by `cmi_dc`, not by pre-whitening.

Entry points
------------
    analyze_arm(z_lr, rep='Z')          one arm, all seeds → RDM + curves
    analyze_grid(reps=('Z',))           every Z_lr arm
    summarize_rdm(grid)                 printed table
    plot_rdm_grid(grid)                 R1 — the checkerboard figure
    plot_development(grid)              R2 — CMI against training block
    plot_frame_contrast(grid_by_frame)  R4 — cue vs outcome

Run `.venv/bin/python rotation_curriculum_rdm.py` for the synthetic self-test, then main().
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

import plot_style
from plot_style import FigSize

from rotation_curriculum_config import (
    EXPORT_ROOT,
    HEADLINE_NOISE,
    RUN_NAME,
    Z_LR,
    ZLR_INFO,
    result_path,
    train_key,
)
from rotation_curriculum_analysis import _phase_range
from rotation_decoding_analysis import extract_decode_samples, flatten_logger
from rotation_geometry_analysis import build_rdm

plot_style.set_plot_style()


# ── Knobs ─────────────────────────────────────────────────────────────────────

#: The S1 phase to sample. S1 carries ['no inference learning', 'Learning and inference'];
#: the passive warm-up is a different learning regime and is excluded.
S1_PHASE = 'Learning and inference'

#: Which mini-blocks a window reads, as {name: selector}. A selector is a mini-block index
#: counted from the switch (0 = the mini-block containing it), a negative index counted from the
#: end of the block, or 'last' / 'last-N'. Dict order is the order of the matrix rows.
#:
#: Any mini-block is addressable, which is what makes the separation-front sweep free — the
#: colour-balancing logic is untouched because every mini-block, including 0, is a full colour
#: permutation. `sweep_miniblocks` walks this axis.
DEFAULT_WINDOWS: Dict[str, Any] = {'early': 1, 'late': 'last'}

#: Names only, for the call sites that just need the row layout.
WINDOWS: Tuple[str, ...] = tuple(DEFAULT_WINDOWS)

#: Arms the separation-front figure draws. Not `RDM_ARMS`: the two floors (0.01, 0.9) have no
#: structure to show on this axis, while 0.05 and 0.4 are where it is most visible.
SWEEP_ARMS: Tuple[Any, ...] = (0.05, 0.1, 0.2, 0.4)

#: Mini-blocks the separation-front sweep walks. Every block has at least 9 complete
#: mini-blocks (the geometric draw's floor is 93 timesteps), so 0-8 are always available.
SWEEP_MINIBLOCKS: Tuple[Any, ...] = (0, 1, 2, 3, 4, 6, 8, 'last')

#: Consecutive blocks per sliding window of the development curve. 4 gives two blocks of each
#: context, and block lags 1, 2, 3 — which `cmi_dc` needs.
WINDOW_BLOCKS = 4

#: Z is sampled at t + z_lag. -1 pairs each hidden state with the Z that actually drove it
#: (`rotation_decoding_analysis.extract_decode_samples`).
Z_LAG = -1

#: Arms drawn as matrices in R1. The full grid still goes into the curves and the table.
RDM_ARMS: Tuple[Any, ...] = ('RNN', 0.01, 0.2, 0.9)

#: Representations. 'H' needs a run with log_hidden_states=True (see the module docs).
REPS: Tuple[str, ...] = ('Z', 'H')

#: R4 runs the whole thing at both frame types.
RDM_FRAMES_CONTRAST: Tuple[str, ...] = ('cue', 'outcome')

METRIC = 'euclidean'
N_PERMUTATIONS = 200
FIG_DIR = EXPORT_ROOT / 'figures'


# ── Sampling ──────────────────────────────────────────────────────────────────

def blocked_range(logger) -> Tuple[int, int]:
    """[start, end) of S1's blocked-learning phase, excluding the passive warm-up.

    Uses `logger.phases`, which is written before compaction, rather than
    `len(logger.inputs)` — compaction collapses that to 1.
    """
    return _phase_range(logger, S1_PHASE)


def _miniblock_phase(colour: np.ndarray, n_colors: int) -> int | None:
    """Offset at which `colour` breaks into whole mini-blocks, or None if it never does.

    A mini-block is a random permutation of the `n_colors` colours (`datasets.py`), so the
    grouping is recoverable from the colour sequence itself — which is what this does, rather
    than trusting arithmetic off the block start.

    That matters because the block boundaries are not all equally trustworthy.  Every real
    boundary comes from an `llcid` change in the *logged* stream and is exact, but the first
    one is the phase start read from `logger.phases`, which is counted differently and lands
    three trials into a mini-block.  Measured on S1 seed 0: blocks 1+ align at offset 0, block
    0 at offset 3.  Detecting the offset makes every window colour-balanced regardless, and
    fails loudly instead of silently returning a window with a colour twice and one missing.
    """
    for phase in range(n_colors):
        n_chunks = (len(colour) - phase) // n_colors
        if n_chunks < 1:
            continue
        chunks = (colour[phase:phase + n_chunks * n_colors]
                  .reshape(n_chunks, n_colors))
        if all(sorted(ch) == list(range(n_colors)) for ch in chunks):
            return phase
    return None


def _resolve_selector(sel: Any, n_mb: int) -> int:
    """Mini-block selector -> concrete index within a block of `n_mb` complete mini-blocks.

    int >= 0   counted from the switch (0 = the mini-block the switch falls in)
    int < 0    counted from the end (-1 = last)
    'last'     the last complete mini-block
    'last-N'   N before that
    """
    if isinstance(sel, str):
        if sel == 'last':
            return n_mb - 1
        if sel.startswith('last-'):
            return n_mb - 1 - int(sel[len('last-'):])
        raise ValueError(f"Unknown window selector '{sel}'; expected an int, 'last' or 'last-N'.")
    sel = int(sel)
    return sel if sel >= 0 else n_mb + sel


def sample_trials(logger, config, rep: str = 'Z', frames: str = 'cue',
                  windows: Dict[str, Any] | Sequence[str] | None = None,
                  z_lag: int = Z_LAG,
                  drop_first_block: bool = True) -> Dict[str, Any] | None:
    """One pattern per sampled trial, with the metadata the RDM is read against.

    Returns dict(X (n, dim), context (n,) in {0, 1}, block (n,), window (n,), colour (n,),
    t_index (n,), n_blocks, n_dropped) or None if the logger carries no usable samples.

    Mini-blocks are recovered from the colour sequence (see `_miniblock_phase`), so each
    window holds every colour exactly once.  The first block is dropped by default: its start
    is the phase boundary rather than a context switch, so "early in the block" does not mean
    the same thing there, and it is a partial block besides.

    Blocks that cannot supply *every* requested window are dropped whole, so each retained
    block contributes exactly `len(windows) * n_colors` rows.  That regularity is what makes
    averaging RDMs across seeds meaningful — the row layout is then identical across seeds
    even though the block lengths and the leading rotation are not.
    """
    if windows is None:
        windows = dict(DEFAULT_WINDOWS)
    elif not isinstance(windows, dict):          # a bare sequence of names keeps the defaults
        windows = {k: DEFAULT_WINDOWS[k] for k in windows}

    t0, t1 = blocked_range(logger)
    s = extract_decode_samples(logger, config, frames=frames, z_lag=z_lag,
                               t_range=(t0, t1), require_hidden=(rep != 'Z'))
    if s is None:
        return None

    nc = int(config.n_colors)
    ii, _, _, _ = flatten_logger(logger, config)
    # The colour one-hot rides the CUE frame only — the outcome frame carries zeros there and
    # the attack (x, y) instead (`datasets.RotatingTargetsDataset`). So an outcome frame's
    # colour is read off the frame before it, which `frame_type` (0=cue, 1=outcome) locates.
    ft = s['frame_type']
    colour = ii[np.maximum(s['t_index'] - ft, 0), :nc].argmax(axis=1)

    rotations = np.asarray(sorted(np.unique(s['angle_deg'])), dtype=float)
    all_blocks = np.unique(s['block_idx'])
    if drop_first_block:
        all_blocks = all_blocks[1:]

    rows: List[int] = []
    win_name: List[str] = []
    n_dropped = 0

    for b in all_blocks:
        in_b = np.where(s['block_idx'] == b)[0]
        # One row per trial at a single frame type; at 'all' the cue rows index the trials and
        # their outcome partners are re-attached below.
        trial_rows = in_b if frames != 'all' else in_b[ft[in_b] == 0]
        phase = _miniblock_phase(colour[trial_rows], nc)
        if phase is None:
            n_dropped += 1
            continue
        n_mb = (len(trial_rows) - phase) // nc
        picks = {name: _resolve_selector(sel, n_mb) for name, sel in windows.items()}
        if (not picks or max(picks.values()) >= n_mb or min(picks.values()) < 0
                or len(set(picks.values())) != len(windows)):
            n_dropped += 1          # block too short to supply distinct windows
            continue
        for name in windows:
            lo = phase + picks[name] * nc
            sel = trial_rows[lo:lo + nc]
            if frames == 'all':     # re-attach each cue frame's outcome partner, by timestep
                want = np.concatenate([s['t_index'][sel], s['t_index'][sel] + 1])
                sel = in_b[np.isin(s['t_index'][in_b], want)]
            sel = sel[np.argsort(colour[sel], kind='stable')]
            rows.extend(sel.tolist())
            win_name.extend([name] * len(sel))

    if not rows:
        return None

    idx = np.asarray(rows, dtype=int)
    ctx = np.searchsorted(rotations, s['angle_deg'][idx]).astype(int)
    # Relabel so the first retained block is always context 0.  Which rotation a seed happens
    # to start with is a coin flip, and dissimilarity does not care about the label — but the
    # figure's context strip and any elementwise average across seeds do, so pin it here.
    # `rotation_deg` keeps the untouched ground truth.
    ctx = (ctx != ctx[0]).astype(int)
    return dict(
        X=s[rep][idx],
        context=ctx,
        rotation_deg=s['angle_deg'][idx],
        block=s['block_idx'][idx].astype(int),
        window=np.asarray(win_name),
        colour=colour[idx].astype(int),
        t_index=s['t_index'][idx].astype(int),
        n_blocks=int(len(np.unique(s['block_idx'][idx]))),
        n_dropped=n_dropped,
        rep=rep, frames=frames, windows=dict(windows),
    )


def trial_rdm(samples: Dict[str, Any], metric: str = METRIC) -> np.ndarray:
    """Pairwise dissimilarity between trial patterns. Features z-scored pooled — see the docs."""
    return build_rdm(samples['X'], metric=metric)


# ── The index ─────────────────────────────────────────────────────────────────

def context_modulation(rdm: np.ndarray, context: np.ndarray, block: np.ndarray,
                       k: int = WINDOW_BLOCKS) -> Dict[str, np.ndarray]:
    """Local context contrast per sliding window of `k` consecutive blocks.

    Returns dict(centre, cmi, cmi_dc, n_windows). Same-block pairs are excluded throughout;
    see the module docstring for why, and for what `cmi_dc` is orthogonal to.
    """
    blocks = np.unique(block)
    centre, cmi, cmi_dc = [], [], []

    for i in range(len(blocks) - k + 1):
        win = blocks[i:i + k]
        sel = np.isin(block, win)
        d = rdm[np.ix_(sel, sel)]
        b, c = block[sel], context[sel]
        iu = np.triu_indices(len(b), k=1)
        vals = d[iu]
        lag = np.abs(b[iu[0]] - b[iu[1]])
        same_ctx = c[iu[0]] == c[iu[1]]
        cross = lag > 0                       # drop same-block pairs

        if not cross.any():
            continue
        overall = float(vals[cross].mean())
        if overall <= 0:                      # a constant representation, e.g. the RNN arm's Z
            centre.append(float(win.mean())); cmi.append(0.0); cmi_dc.append(0.0)
            continue

        w_in = cross & same_ctx
        w_bt = cross & ~same_ctx
        cmi.append(float((vals[w_bt].mean() - vals[w_in].mean()) / overall)
                   if w_in.any() and w_bt.any() else np.nan)

        # Parity contrast at block lags 1, 2, 3: (d1 + d3)/2 - d2 kills any linear trend in lag.
        dl = {L: float(vals[cross & (lag == L)].mean())
              for L in (1, 2, 3) if (cross & (lag == L)).any()}
        cmi_dc.append(float(((dl[1] + dl[3]) / 2 - dl[2]) / overall)
                      if {1, 2, 3} <= set(dl) else np.nan)
        centre.append(float(win.mean()))

    return dict(centre=np.asarray(centre), cmi=np.asarray(cmi),
                cmi_dc=np.asarray(cmi_dc), n_windows=len(centre))


def permutation_null(rdm: np.ndarray, context: np.ndarray, block: np.ndarray,
                     k: int = WINDOW_BLOCKS, n: int = N_PERMUTATIONS,
                     seed: int = 0) -> Dict[str, float]:
    """Null for `cmi` from shuffling the block→context map, keeping block structure intact.

    Permuting *blocks* rather than trials is the point: trials within a block are not
    exchangeable, so a trial-level shuffle would give a null far tighter than the data's own
    noise and every arm would clear it.
    """
    rng = np.random.default_rng(seed)
    blocks = np.unique(block)
    labels = np.asarray([context[block == b][0] for b in blocks])
    draws = []
    for _ in range(n):
        perm = rng.permutation(labels)
        ctx = np.zeros_like(context)
        for b, lab in zip(blocks, perm):
            ctx[block == b] = lab
        draws.append(np.nanmean(context_modulation(rdm, ctx, block, k=k)['cmi']))
    draws = np.asarray(draws, dtype=float)
    return dict(mean=float(np.nanmean(draws)),
                lo=float(np.nanpercentile(draws, 2.5)),
                hi=float(np.nanpercentile(draws, 97.5)))


def curve_reliability(curves: np.ndarray, n_splits: int = 50, seed: int = 0) -> float:
    """Split-half reliability of the development curve's SHAPE across seeds, Spearman-Brown.

    Rank-correlates the mean curve of one random half of the seeds against the other half.
    This measures whether the *shape* of the development curve replicates, not whether the
    arm has an effect: an arm with a large but genuinely flat index has no shape to recover
    and lands near 0 or below. Read it beside `cmi`, never instead of it. Needs >= 4 seeds;
    returns nan below that, or if every split is constant.
    """
    from scipy.stats import spearmanr

    curves = np.asarray(curves, dtype=float)
    n = len(curves)
    if n < 4:
        return float('nan')
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a = np.nanmean(curves[idx[: n // 2]], axis=0)
        b = np.nanmean(curves[idx[n // 2:]], axis=0)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        if np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
            continue                      # a constant curve (e.g. the RNN arm) has no shape
        r, _ = spearmanr(a[ok], b[ok])
        if np.isfinite(r):
            rs.append(2 * r / (1 + r) if r > -1 else np.nan)
    return float(np.nanmean(rs)) if rs else float('nan')


# ── Loading one arm ───────────────────────────────────────────────────────────

def load_arm_runs(z_lr: Any, seeds: Sequence[int] | None = None,
                  noise_std: float = HEADLINE_NOISE, stage: str = 'S1',
                  cue_mode: str = 'oracle_z', run_name: str | None = None,
                  verbose: bool = False) -> List[Tuple[Any, Any]]:
    """(logger, config) per seed for one arm and one curriculum stage.

    Stage 'S1' is unkeyed; 'S2' / 'S3' / 'S3_pinned' are keyed by cue_mode.
    """
    import pickle

    if seeds is None:
        # Every seed present on disk, NOT `active_seeds()`. The sibling behavioural analysis
        # takes its count from `AnalysisParams(n_seeds=active_seeds())`, which follows the
        # PILOT flag — so with PILOT left True it would read 1 of the 10 seeds sitting in the
        # export directory and say so only in a column heading. Globbing cannot under-read.
        root = result_path(z_lr, noise_std, 0).parent
        if run_name is not None:
            root = EXPORT_ROOT.parent / run_name / train_key(z_lr) / f'noise_std-{noise_std}'
        seeds = sorted(int(f.stem.split('-')[-1]) for f in root.glob('results_seed-*.pkl'))

    runs: List[Tuple[Any, Any]] = []
    for seed in seeds:
        # `run_name` reads a specific run regardless of what the config currently resolves to —
        # the way to read the existing behavioural results while RECORD_HIDDEN points RUN_NAME at
        # the hidden-logging directory, and vice versa. Same convention as
        # inspect_curriculum_run.load_tree.
        path = result_path(z_lr, noise_std, seed)
        if run_name is not None:
            path = (EXPORT_ROOT.parent / run_name / train_key(z_lr)
                    / f'noise_std-{noise_std}' / f'results_seed-{seed}.pkl')
        if not path.exists():
            if verbose:
                print(f'  missing: {path}')
            continue
        with path.open('rb') as f:
            tree = pickle.load(f)
        if stage == 'S1':
            runs.append((tree['S1'], tree['configs']['S1']))
        else:
            runs.append((tree[stage][cue_mode], tree['configs'][stage][cue_mode]))
    return runs


def analyze_arm(z_lr: Any, rep: str = 'Z', frames: str = 'cue', stage: str = 'S1',
                metric: str = METRIC, k: int = WINDOW_BLOCKS,
                windows: Dict[str, Any] | Sequence[str] | None = None,
                permute: bool = True, runs: List[Tuple[Any, Any]] | None = None,
                **load_kw) -> Dict[str, Any] | None:
    """One arm, every seed: the seed-averaged RDM and the development curves.

    RDMs are built per seed and averaged elementwise — patterns are never pooled across
    seeds, because different networks have different hidden bases. The elementwise average is
    legitimate only because `sample_trials` gives every seed the same row layout and pins the
    first block to context 0; seeds are truncated to their common block count first.
    """
    # `runs` lets a caller that re-analyses the same arm many times (sweep_miniblocks) pay the
    # pickle load once instead of once per setting.
    if runs is None:
        runs = load_arm_runs(z_lr, stage=stage, **load_kw)
    if not runs:
        return None

    per_seed = []
    for logger, config in runs:
        try:
            s = sample_trials(logger, config, rep=rep, frames=frames, windows=windows)
        except ValueError as exc:                    # no hidden states logged in this run
            return dict(error=str(exc), z_lr=z_lr, rep=rep, frames=frames)
        if s is None:
            continue
        per_seed.append((s, trial_rdm(s, metric=metric)))
    if not per_seed:
        return None

    n_blocks = min(s['n_blocks'] for s, _ in per_seed)
    ref = per_seed[0][0]
    # Read rows-per-block off the data rather than computing it: at frames='all' a window holds
    # two rows per colour, not one, and hard-coding len(windows) * n_colors silently truncated
    # half of every 'all' run.
    rows_per_block = len(ref['window']) // ref['n_blocks']
    n_rows = n_blocks * rows_per_block
    window_names = list(ref['windows'])

    window = ref['window'][:n_rows]
    context = ref['context'][:n_rows]
    block = ref['block'][:n_rows]

    rdm = np.mean([D[:n_rows, :n_rows] for _, D in per_seed], axis=0)

    curves: Dict[str, np.ndarray] = {}
    null: Dict[str, Dict[str, float]] = {}
    centre: Dict[str, np.ndarray] = {}
    # 'both' is only meaningful with more than one window; with a single window it would be
    # the same numbers under a second name.
    passes = (['both'] if len(window_names) > 1 else []) + window_names
    for w in passes:
        mask = np.ones(n_rows, bool) if w == 'both' else (window == w)
        rows = []
        for s, D in per_seed:
            cm = context_modulation(D[:n_rows, :n_rows][np.ix_(mask, mask)],
                                    context[mask], block[mask], k=k)
            rows.append(cm)
        n_win = min(c['n_windows'] for c in rows)
        if n_win == 0:
            continue
        curves[w] = np.stack([c['cmi'][:n_win] for c in rows])
        curves[w + '_dc'] = np.stack([c['cmi_dc'][:n_win] for c in rows])
        centre[w] = rows[0]['centre'][:n_win]
        null[w] = (permutation_null(rdm[np.ix_(mask, mask)], context[mask], block[mask], k=k)
                   if permute else dict(mean=0.0, lo=np.nan, hi=np.nan))

    return dict(z_lr=z_lr, rep=rep, frames=frames, stage=stage, metric=metric,
                rdm=rdm, context=context, block=block, window=window,
                curves=curves, centre=centre, null=null,
                windows=dict(ref['windows']), window_names=window_names,
                n_seeds=len(per_seed), n_blocks=n_blocks,
                n_dropped=int(sum(s['n_dropped'] for s, _ in per_seed)),
                reliability={w: curve_reliability(curves[w])
                             for w in curves if not w.endswith('_dc')})


def analyze_grid(arms: Sequence[Any] = Z_LR, reps: Sequence[str] = ('Z',),
                 frames: str = 'cue', **kw) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    """Every (rep, arm) cell. Cells that cannot be read are reported and skipped."""
    grid: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for rep in reps:
        for z_lr in arms:
            res = analyze_arm(z_lr, rep=rep, frames=frames, **kw)
            if res is None:
                print(f'  {rep} {z_lr}: no runs on disk')
                continue
            if 'error' in res:
                print(f'  {rep} {z_lr}: {res["error"]}')
                continue
            grid[(rep, z_lr)] = res
    return grid


# ── Printed summary ───────────────────────────────────────────────────────────

def summarize_rdm(grid: Dict[Tuple[str, Any], Dict[str, Any]],
                  windows: Sequence[str] = ('both',) + WINDOWS) -> None:
    """One row per (rep, arm, window): the index late in training, its null, its reliability.

    `cmi` is the headline; `cmi_dc` is the drift-orthogonal control. `late-third` averages the
    curve over the final third of training, which is where the developed representation lives.
    """
    print(f'\n{"rep":>3} {"Z_lr":>6} {"win":>5} {"blocks":>6} {"seeds":>5} '
          f'{"cmi(all)":>9} {"cmi(late3)":>11} {"dc(late3)":>10} '
          f'{"null 95%":>16} {"rel":>6}')
    print('-' * 92)
    for (rep, z_lr), res in grid.items():
        for w in windows:
            if w not in res['curves']:
                continue
            c = res['curves'][w]
            dc = res['curves'][w + '_dc']
            third = max(1, c.shape[1] // 3)
            nl = res['null'][w]
            null_txt = '[{:.3f},{:.3f}]'.format(nl['lo'], nl['hi'])
            print(f'{rep:>3} {str(z_lr):>6} {w:>5} {res["n_blocks"]:>6} {res["n_seeds"]:>5} '
                  f'{np.nanmean(c):>9.3f} {np.nanmean(c[:, -third:]):>11.3f} '
                  f'{np.nanmean(dc[:, -third:]):>10.3f} '
                  f'{null_txt:>16} '
                  f'{res["reliability"].get(w, float("nan")):>6.2f}')
        if res['n_dropped']:
            print(f'      ({res["n_dropped"]} block(s) dropped across seeds — too short to '
                  f'supply every window)')


# ── Self-test ─────────────────────────────────────────────────────────────────

def _synthetic(n_blocks: int = 14, n_colors: int = 5, dim: int = 32,
               context_gain=0.0, drift_gain: float = 0.0, colour_gain: float = 0.0,
               noise: float = 1.0, seed: int = 0) -> Dict[str, np.ndarray]:
    """Synthetic trial patterns with a known code, in `sample_trials`'s layout.

    `context_gain` may be a scalar or a callable of the block fraction (0 -> 1), which is how
    the ramp case builds a representation that *develops*. Blocks strictly alternate context,
    as they do in the task.
    """
    rng = np.random.default_rng(seed)
    v_ctx, v_drift = rng.standard_normal(dim), rng.standard_normal(dim)
    v_col = rng.standard_normal((n_colors, dim))
    gain = context_gain if callable(context_gain) else (lambda f: context_gain)

    X, ctx, blk, win, col = [], [], [], [], []
    for b in range(n_blocks):
        f = b / max(1, n_blocks - 1)
        for w in WINDOWS:
            for c in range(n_colors):
                x = noise * rng.standard_normal(dim)
                x = x + gain(f) * (1.0 if b % 2 == 0 else -1.0) * v_ctx
                x = x + drift_gain * f * v_drift
                x = x + colour_gain * v_col[c]
                X.append(x); ctx.append(b % 2); blk.append(b); win.append(w); col.append(c)
    return dict(X=np.asarray(X), context=np.asarray(ctx), block=np.asarray(blk),
                window=np.asarray(win), colour=np.asarray(col))


_SYN_T_WINDOWS: Tuple[str, ...] = ('mb0', 'mb1', 'late')
_SYN_T_BLOCKS = 8
_SYN_T_COLORS = 5


def _syn_mask(window: str) -> np.ndarray:
    """Row mask for one window of `_synthetic_templates`'s fixed layout."""
    w = np.asarray([w for _ in range(_SYN_T_BLOCKS)
                    for w in _SYN_T_WINDOWS for _ in range(_SYN_T_COLORS)])
    return w == window


def _synthetic_templates(mode: str, dim: int = 32, noise: float = 0.25,
                         seed: int = 0) -> Dict[str, np.ndarray]:
    """Trial patterns with a KNOWN held context, in `sample_trials`'s layout.

    The `late` window always holds the block's own context (every arm does settle correctly —
    see `template_index`), so the templates are well posed. `mode` sets what the early windows
    hold: the previous block's state ('persev'), this block's ('correct'), or the midpoint
    ('ambiguous'). Blocks alternate context, as they do in the task.
    """
    rng = np.random.default_rng(seed)
    settled = {0: rng.standard_normal(dim) * 2.0, 1: rng.standard_normal(dim) * 2.0}
    X, ctx, blk, win = [], [], [], []
    for b in range(_SYN_T_BLOCKS):
        c_now = b % 2
        c_prev = 1 - c_now
        for w in _SYN_T_WINDOWS:
            if w == 'late' or mode == 'correct':
                held = settled[c_now]
            elif mode == 'persev':
                held = settled[c_prev]
            elif mode == 'ambiguous':
                held = 0.5 * (settled[0] + settled[1])
            else:
                raise ValueError(f'Unknown mode {mode!r}')
            for _ in range(_SYN_T_COLORS):
                X.append(held + noise * rng.standard_normal(dim))
                ctx.append(c_now); blk.append(b); win.append(w)
    return dict(X=np.asarray(X), context=np.asarray(ctx),
                block=np.asarray(blk), window=np.asarray(win))


def self_test(verbose: bool = True) -> bool:
    """Four synthetic cases, no trained model. Pins the estimator before any data is read."""
    ok = True

    def report(name, got, want):
        nonlocal ok
        good = want(got)
        ok = ok and good
        if verbose:
            print(f'  {"PASS" if good else "FAIL"}  {name}: {got}')

    # (1) A context effect that RAMPS over training must produce a rising curve that ends high.
    s = _synthetic(context_gain=lambda f: 1.6 * f)
    cm = context_modulation(trial_rdm(s), s['context'], s['block'])
    ramp = dict(first=round(float(cm['cmi'][0]), 3), last=round(float(cm['cmi'][-1]), 3),
                dc_last=round(float(cm['cmi_dc'][-1]), 3))
    report('ramping context -> rising curve', ramp,
           lambda g: g['last'] > g['first'] + 0.10 and g['last'] > 0.15 and g['dc_last'] > 0.10)

    # (2) No context effect at all: the index sits at zero and inside its own null.
    s = _synthetic(context_gain=0.0)
    D = trial_rdm(s)
    cm = context_modulation(D, s['context'], s['block'])
    nl = permutation_null(D, s['context'], s['block'], n=100)
    flat = dict(cmi=round(float(np.nanmean(cm['cmi'])), 3),
                null=(round(nl['lo'], 3), round(nl['hi'], 3)))
    report('no context -> zero, inside null', flat,
           lambda g: abs(g['cmi']) < 0.05 and g['null'][0] < g['cmi'] < g['null'][1])

    # (3) Pure drift, no context. This is the case cmi_dc exists for: drift biases `cmi`
    #     (within-context pairs sit at larger block lags, so it goes NEGATIVE), while the
    #     lag-parity contrast must stay near zero.
    s = _synthetic(context_gain=0.0, drift_gain=6.0, noise=0.6)
    cm = context_modulation(trial_rdm(s), s['context'], s['block'])
    drift = dict(cmi=round(float(np.nanmean(cm['cmi'])), 3),
                 cmi_dc=round(float(np.nanmean(cm['cmi_dc'])), 3))
    report('drift only -> cmi_dc immune', drift,
           lambda g: abs(g['cmi_dc']) < 0.10 and g['cmi'] < 0.05)

    # (4) A colour effect with no context effect must not read as context. This is what the
    #     mini-block-aligned, colour-balanced window buys.
    s = _synthetic(context_gain=0.0, colour_gain=3.0)
    cm = context_modulation(trial_rdm(s), s['context'], s['block'])
    colour = dict(cmi=round(float(np.nanmean(cm['cmi'])), 3),
                  cmi_dc=round(float(np.nanmean(cm['cmi_dc'])), 3))
    report('colour only -> zero', colour,
           lambda g: abs(g['cmi']) < 0.05 and abs(g['cmi_dc']) < 0.05)

    # ── the directional index ────────────────────────────────────────────────
    # (5) A perfectly perseverative model must score ~1 and a correct one ~0 on data where cmi
    #     cannot tell them apart at all. This pair is the reason the index exists.
    persev = template_index(_synthetic_templates('persev'))
    correct = template_index(_synthetic_templates('correct'))
    ambig = template_index(_synthetic_templates('ambiguous'))
    # cmi has to be compared on the mb0 rows ALONE, which is how it is actually read on this
    # axis (`sweep_miniblocks` analyses one window at a time). Mixing the late rows in would
    # break the block clustering and compare something else.
    def _cmi_at(mode, window='mb0'):
        s_ = _synthetic_templates(mode)
        mk = _syn_mask(window)
        sub = dict(X=s_['X'][mk])
        return float(np.nanmean(context_modulation(
            trial_rdm(sub), s_['context'][mk], s_['block'][mk])['cmi']))

    cmi_p, cmi_c = _cmi_at('persev'), _cmi_at('correct')

    report('perseverative model -> index ~1',
           dict(mb0=round(float(np.nanmean(persev[_syn_mask('mb0')])), 3)),
           lambda g: g['mb0'] > 0.85)
    report('correct model -> index ~0',
           dict(mb0=round(float(np.nanmean(correct[_syn_mask('mb0')])), 3)),
           lambda g: g['mb0'] < 0.15)
    report('ambiguous model -> index ~0.5',
           dict(mb0=round(float(np.nanmean(ambig[_syn_mask('mb0')])), 3)),
           lambda g: abs(g['mb0'] - 0.5) < 0.12)
    report('...while cmi cannot tell those two apart',
           dict(persev=round(float(cmi_p), 3), correct=round(float(cmi_c), 3)),
           lambda g: abs(g['persev'] - g['correct']) < 0.10 and g['persev'] > 0.3)
    # (6) The reference window itself must score ~0: a block's own settled state is at the
    #     new-context template, estimated from OTHER blocks, so this is not circular.
    report('reference window -> index ~0',
           dict(late=round(float(np.nanmean(correct[_syn_mask('late')])), 3)),
           lambda g: g['late'] < 0.15)

    if verbose:
        print(f'\nself-test: {"PASS" if ok else "FAIL"}')
    return ok


# ── Figures ───────────────────────────────────────────────────────────────────

def _context_strip(ax, block: np.ndarray, context: np.ndarray, cs) -> None:
    """Context of each block as a rule just outside the matrix, on both edges.

    Offset in POINTS below / left of the axis rather than in axes fractions, the reasoning
    `flanker_figure_utils.super_labels` records: a fractional offset has to guess how tall the
    tick labels are and drifts the moment the panel height or the font size changes.
    """
    from matplotlib.transforms import offset_copy

    colors = (cs.contextA, cs.contextB)
    x_tr = offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=-2.5, units='points')
    y_tr = offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=-2.5, units='points')
    for b in np.unique(block):
        i = np.where(block == b)[0]
        lo, hi = i[0] - 0.5, i[-1] + 0.5
        c = colors[int(context[i[0]])]
        ax.plot([lo, hi], [0, 0], transform=x_tr, color=c, lw=1.4, clip_on=False, zorder=6,
                solid_capstyle='butt')
        ax.plot([0, 0], [lo, hi], transform=y_tr, color=c, lw=1.4, clip_on=False, zorder=6,
                solid_capstyle='butt')


def plot_rdm_grid(grid: Dict[Tuple[str, Any], Dict[str, Any]],
                  arms: Sequence[Any] = RDM_ARMS, reps: Sequence[str] = REPS,
                  window: str = 'both', figsize=None, panel: float = 0.78):
    """R1 — the checkerboard figure. One matrix per (representation, Z_lr).

    Trials run chronologically, so the diagonal is training time and the block-scale
    checkerboard is the context effect. Within each block the rows are
    `early` mini-block then `late`, each colour-ordered, so a colour effect would appear as a
    fine 5-row pattern rather than at the block scale.

    A panel per arm is exactly what `docs/figure_style.md` warns against for curves, but a
    matrix cannot be overlaid; `plot_candidate_rdms` sets the same precedent.
    """
    cells = [(rep, z) for rep in reps for z in arms if (rep, z) in grid]
    reps = [r for r in reps if any(c[0] == r for c in cells)]
    arms = [z for z in arms if any(c[1] == z for c in cells)]
    if not cells:
        raise ValueError('Nothing to draw — no (rep, arm) cell in the grid.')

    cs = plot_style.Color_scheme()
    metric = grid[cells[0]].get('metric', METRIC)
    nr, ncl = len(reps), len(arms)
    fig, axes = plt.subplots(nr, ncl, squeeze=False,
                             figsize=figsize or FigSize.custom(panel * ncl + 0.55,
                                                               panel * nr + 0.42))
    # One colour scale PER ROW. Absolute dissimilarity is not comparable across
    # representations of different dimensionality — a 2-dim Z and a 64-dim hidden state live on
    # different distance scales, which is the same reason `cmi` is not comparable across them
    # (see `unit_discriminability`). A shared scale flattens every row but the widest.
    def _span(rep):
        # 2nd-98th percentile of the OFF-DIAGONAL entries, pooled over the row's arms. The
        # diagonal is 0 by construction and would drag vmin down to zero on its own; hidden-state
        # distances then occupy the top fifth of the colormap and the checkerboard is invisible.
        vals = []
        for z in arms:
            if (rep, z) not in grid:
                continue
            D = grid[(rep, z)]['rdm']
            vals.append(D[~np.eye(len(D), dtype=bool)])
        v = np.concatenate(vals)
        return float(np.nanpercentile(v, 2)), float(np.nanpercentile(v, 98))

    row_span = {rep: _span(rep) for rep in reps}
    row_im = {}
    for i, rep in enumerate(reps):
        vmin, vmax = row_span[rep]
        for j, z in enumerate(arms):
            ax = axes[i][j]
            if (rep, z) not in grid:
                ax.axis('off')
                continue
            res = grid[(rep, z)]
            mask = (np.ones(len(res['window']), bool) if window == 'both'
                    else res['window'] == window)
            D = res['rdm'][np.ix_(mask, mask)]
            blk, ctx = res['block'][mask], res['context'][mask]
            row_im[rep] = ax.imshow(D, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax,
                                    interpolation='none', rasterized=True)
            # Block boundaries: without them the checkerboard's period is guesswork.
            for b in np.unique(blk)[1:]:
                e = np.where(blk == b)[0][0] - 0.5
                ax.axhline(e, color='w', lw=0.3, alpha=0.6)
                ax.axvline(e, color='w', lw=0.3, alpha=0.6)
            _context_strip(ax, blk, ctx, cs)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel('Training trials')
            if i == 0:
                ax.set_title(ZLR_INFO[z].label)          # otherwise unidentifiable panels
            if j == 0:
                ax.set_ylabel(rep)
    for i, rep in enumerate(reps):
        if rep not in row_im:
            continue
        cb = fig.colorbar(row_im[rep], ax=list(axes[i]), fraction=0.028, pad=0.02)
        cb.ax.tick_params(labelsize=4.5)
        # Every row labelled, and named after the distance actually computed rather than
        # describing the scaling — the per-row scaling is documented, the metric is the thing a
        # reader cannot infer from the picture.
        cb.set_label(f'Dissimilarity ({metric})', fontsize=5)
    return fig


def plot_development(grid: Dict[Tuple[str, Any], Dict[str, Any]],
                     rep: str = 'Z', windows: Sequence[str] = WINDOWS,
                     figsize=None):
    """R2 — the development curve: context modulation against training block.

    One line per individual's `Z_lr`, band = SEM across seeds. No null band is drawn: the
    permutation null's width scales with how structured an arm's RDM is, so the widest one
    would come from the strongest arm and reading it as a common threshold would be wrong.
    Per-arm nulls are in `summarize_rdm`; the `'RNN'` arm on the plot is the empirical floor.
    """
    fig, axes = plt.subplots(1, len(windows), squeeze=False,
                             figsize=figsize or FigSize.row(len(windows)))
    arms = [z for z in Z_LR if (rep, z) in grid]
    for j, w in enumerate(windows):
        ax = axes[0][j]
        n_seeds = 0
        for z in arms:
            res = grid[(rep, z)]
            if w not in res['curves']:
                continue
            c = res['curves'][w]
            n_seeds = max(n_seeds, c.shape[0])
            mu = np.nanmean(c, axis=0)
            se = np.nanstd(c, axis=0) / np.sqrt(c.shape[0])
            x = res['centre'][w]
            info = ZLR_INFO[z]
            ax.plot(x, mu, color=info.color, label=info.label)
            ax.fill_between(x, mu - se, mu + se, color=info.color, alpha=0.25, lw=0)
        ax.axhline(0.0, color='0.6', lw=0.6, zorder=0)
        ax.set_xlabel('training block')
        ax.set_ylabel('context modulation' if j == 0 else None)
        ax.set_title(w)                  # the two panels are otherwise indistinguishable
    from flanker_figure_utils import share_ylim
    share_ylim(*axes[0])
    # A row above the axes: eight arms will not fit inside a panel without covering the data.
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    fig.supxlabel(f'{rep}, cue frames · mean ± SEM over {n_seeds} seeds · '
                  f'{WINDOW_BLOCKS}-block sliding window')
    fig.tight_layout()
    return fig


# ── R3: does the representational measure track behaviour? ────────────────────

def behavioural_recovery(z_lr: Any, scalar: str = 'mb2', stage: str = 'S1',
                         **load_kw) -> np.ndarray:
    """Per-seed behavioural adaptation scalar for one arm, from the existing analysis.

    Reuses `rotation_curriculum_analysis.switch_aligned_error` unchanged and reads the same
    scalars off it that `curve_summary` does, at the same `block_frac` the curriculum analysis
    uses for this stage (1/3 for S1 — flexibility only means something once the task is
    learned). Lower is better: 0 is predicting the correct target, 0.5 is chance.
    """
    from rotation_curriculum_analysis import (AnalysisParams, _frac_for, switch_aligned_error,
                                              trial_axis)

    params = AnalysisParams()
    frac = _frac_for(stage, params)
    x = trial_axis(params)
    out = []
    for logger, config in load_arm_runs(z_lr, stage=stage, **load_kw):
        arr = switch_aligned_error(logger, config, params, frac)
        if arr.shape[0] == 0 or np.all(np.isnan(arr)):
            out.append(np.nan)
            continue
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(arr, axis=0)
        nc = int(config.n_colors)
        spans = {'mb1': (1, nc), 'mb2': (nc + 1, 2 * nc),
                 'asym': (max(1, int(x.max()) - nc + 1), int(x.max())),
                 'pre': (int(x.min()), 0)}
        lo, hi = spans[scalar]
        sel = (x >= lo) & (x <= hi)
        out.append(float(np.nanmean(mean[sel])) if sel.any() else np.nan)
    return np.asarray(out, dtype=float)


def plot_behaviour_validation(grid: Dict[Tuple[str, Any], Dict[str, Any]],
                              rep: str = 'Z', windows: Sequence[str] = WINDOWS,
                              scalar: str = 'mb2', figsize=None, **load_kw):
    """R3 — context modulation against behavioural recovery, one point per (Z_lr, seed).

    The panel that asks whether the index measures something behaviourally real. x is the
    behavioural scalar (lower = recovers faster from a switch, 0.5 = chance), y is the index
    over the last third of training, so both describe the developed state.

    Both windows are drawn because they do not agree, which is the finding. The behavioural
    scalar is read a few trials *after* a switch, so it is the `early` window that is the
    matched comparison; a high `late` index with poor behaviour means the representation
    separates the contexts by the end of a block but not in time to be used. Measured on Z:
    `Z_lr = 0.05` is exactly that case, the highest late index in the grid (1.16) alongside
    near-chance recovery. Do not read this figure as validation in one direction.
    """
    fig, axes = plt.subplots(1, len(windows), squeeze=False,
                             figsize=figsize or FigSize.row(len(windows), FigSize.large))
    for j, w in enumerate(windows):
        ax = axes[0][j]
        for z in [z for z in Z_LR if (rep, z) in grid]:
            c = grid[(rep, z)]['curves'].get(w)
            if c is None:
                continue
            third = max(1, c.shape[1] // 3)
            y = np.nanmean(c[:, -third:], axis=1)
            b = behavioural_recovery(z, scalar=scalar, **load_kw)
            n = min(len(y), len(b))
            ax.scatter(b[:n], y[:n], s=plt.rcParams['lines.markersize'] ** 2 * 0.5,
                       color=ZLR_INFO[z].color, label=ZLR_INFO[z].label,
                       edgecolors='none', alpha=0.85)
        ax.axvline(0.5, color='0.6', lw=0.6, zorder=0)      # behavioural chance
        ax.axhline(0.0, color='0.6', lw=0.6, zorder=0)
        ax.set_xlabel(f'S1 normalized state error ({scalar})')
        ax.set_ylabel(f'context modulation ({rep})' if j == 0 else None)
        ax.set_title(w)
    from flanker_figure_utils import share_ylim
    share_ylim(*axes[0])
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    return fig


# ── R4: how much of the effect is the stimulus? ───────────────────────────────

def plot_frame_contrast(by_frame: Dict[str, Dict[Tuple[str, Any], Dict[str, Any]]],
                        rep: str = 'Z', window: str = 'late', figsize=None):
    """R4 — the index at cue vs outcome frames, per arm.

    At the outcome frame the attack (x, y) sits in the input, so rotation is readable from the
    current stimulus; at the cue frame it is not. The gap is how much of an outcome-frame
    effect is stimulus rather than held context — and the reason the human design has to keep
    the two periods separable.

    This is an `H` panel. Z is a free parameter fitted across the window, not a function of
    the current frame's contents, so its two frame types nearly coincide (measured: within
    ~0.02 at every arm) — which is a useful check that the frame split is doing what it says,
    and nothing more.
    """
    frames = [f for f in RDM_FRAMES_CONTRAST if f in by_frame]
    arms = [z for z in Z_LR if any((rep, z) in by_frame[f] for f in frames)]
    fig, ax = plt.subplots(figsize=figsize or FigSize.wide)
    x = np.arange(len(arms), dtype=float)
    for off, f in zip((-0.12, 0.12), frames):
        mu, se = [], []
        for z in arms:
            res = by_frame[f].get((rep, z))
            c = None if res is None else res['curves'].get(window)
            if c is None:
                mu.append(np.nan); se.append(np.nan); continue
            third = max(1, c.shape[1] // 3)
            v = np.nanmean(c[:, -third:], axis=1)
            mu.append(np.nanmean(v)); se.append(np.nanstd(v) / np.sqrt(len(v)))
        # Frame type is the second factor, so it rides marker fill rather than a second hue —
        # the arm's own colour still identifies it (docs/figure_style.md).
        filled = (f == 'cue')
        for i, z in enumerate(arms):
            ax.errorbar(x[i] + off, mu[i], yerr=se[i], marker='o', ls='none',
                        color=ZLR_INFO[z].color,
                        markerfacecolor=ZLR_INFO[z].color if filled else 'none',
                        capsize=1.2, lw=0.7)
    ax.axhline(0.0, color='0.6', lw=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([ZLR_INFO[z].label for z in arms], rotation=45, ha='right')
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(f'context modulation ({rep}, {window})')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='none', color='0.35', label='cue'),
                       Line2D([], [], marker='o', ls='none', color='0.35',
                              markerfacecolor='none', label='outcome')],
              loc='best', frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.25)
    fig.tight_layout()
    return fig


# ── Driver ────────────────────────────────────────────────────────────────────

def main(reps: Sequence[str] = REPS, seeds: Sequence[int] | None = None,
         run_name: str | None = None, save_plots: bool = True,
         extras: bool = False) -> Dict[str, Any]:
    """The RDM matrices, and a table, from whatever is on disk.

    Two sets of three, the same plot under three trial selections:

        rdm_matrices{,_early,_late}.pdf          rows `Z` and `H` — the latent, and the hidden
                                                state after the Z gate multiplies it
        rdm_matrices_pregate{,_early,_late}.pdf  row `H_pre` — the same timestep before the gate

        (no suffix)  both windows per block — 10 trials, early and late interleaved, so the
                     checkerboard has a 10-row period
        _early       the 2nd mini-block only, 5 trials just after the switch
        _late        the last complete mini-block only, 5 trials once settled

    All of them share one colour scale per representation, so the three windows can be compared
    in level as well as in pattern. Columns are the `Z_lr` arms.

    `extras=True` adds the derived-index panels (context modulation against training block, the
    directional "which context" index, the mini-block x training-block map, the behaviour
    scatter, the cue/outcome contrast). They are diagnostics for questions an RDM cannot answer
    on its own, not the deliverable, and they are not computed unless asked for.

    'H' / 'H_pre' are skipped with a message on a run that did not log hidden activity — set
    `RECORD_HIDDEN = True` in rotation_curriculum_config and re-run the sweep.
    """
    from flanker_figure_utils import save

    load_kw = dict(seeds=seeds, run_name=run_name)
    # Figures belong with the run they were computed FROM, not with whatever RUN_NAME currently
    # resolves to — otherwise reading an older run with run_name= silently files its panels under
    # the new run's directory.
    fig_dir = FIG_DIR if run_name is None else (EXPORT_ROOT.parent / run_name / 'figures')
    print(f'Run: {run_name or RUN_NAME}\nFigures: {fig_dir}')

    frames = RDM_FRAMES_CONTRAST if extras else ('cue',)
    by_frame = {f: analyze_grid(reps=reps, frames=f, **load_kw) for f in frames}
    grid = by_frame['cue']
    if not grid:
        print('No cue-frame cells could be read — nothing to draw.')
        return dict(by_frame=by_frame)
    summarize_rdm(grid)

    present = [r for r in reps if any(c[0] == r for c in grid)]
    # Derived indices are diagnostics, not the deliverable — computed only when asked for.
    templates: Dict[str, Any] = {}
    sweeps: Dict[str, Any] = {}
    if extras:
        templates = {rep: template_grid(rep=rep, **load_kw) for rep in present}
        for rep in present:
            summarize_templates(templates[rep])
    if not save_plots:
        return dict(by_frame=by_frame, grid=grid, templates=templates, sweeps=sweeps)

    fig_dir.mkdir(parents=True, exist_ok=True)
    # Inspection size, not the paper preset: a 130x130 matrix inside a 0.78in panel is not
    # legible, and these are the figures the analysis exists to produce. Shrink them when the
    # paper layout is fixed, per docs/figure_style.md.
    #
    # The three window views share one colour scale, because `_span` reads the FULL matrix while
    # each panel draws a slice of it. So `early` and `late` are directly comparable, including in
    # overall distance level, rather than each being renormalised to its own range.
    # The pre-gate hidden state goes in its own file. It is the null case — Z never enters the
    # recurrence under post_gating — so keeping it as a third row only shrank the two rows that
    # carry signal.
    gated = [r for r in present if r != 'H_pre']
    pregate = [r for r in present if r == 'H_pre']
    for tag, window in (('', 'both'), ('_early', 'early'), ('_late', 'late')):
        if gated:
            save(plot_rdm_grid(grid, reps=gated, window=window, panel=1.35),
                 fig_dir / f'rdm_matrices{tag}.pdf', note=f'{window} window(s) per block')
        if pregate:
            save(plot_rdm_grid(grid, reps=pregate, window=window, panel=1.35),
                 fig_dir / f'rdm_matrices_pregate{tag}.pdf',
                 note=f'pre-gate hidden, {window} window(s) per block')

    if extras:
        for rep in present:
            save(plot_development(grid, rep=rep),
                 fig_dir / f'extra_context_modulation_{rep}.pdf')
            if templates.get(rep):
                save(plot_template_index(templates[rep], rep=rep, **load_kw),
                     fig_dir / f'extra_which_context_{rep}.pdf')
            sweeps[rep] = sweep_miniblocks(rep=rep, **load_kw)
            if sweeps[rep]:
                save(plot_separation_front(sweeps[rep], rep=rep),
                     fig_dir / f'extra_separation_front_{rep}.pdf')
            save(plot_behaviour_validation(grid, rep=rep, **load_kw),
                 fig_dir / f'extra_vs_behaviour_{rep}.pdf')
            if 'outcome' in by_frame and by_frame['outcome']:
                save(plot_frame_contrast(by_frame, rep=rep),
                     fig_dir / f'extra_frame_contrast_{rep}.pdf')

    return dict(by_frame=by_frame, grid=grid, templates=templates, sweeps=sweeps)




# ── The separation front: mini-block since switch x training block ────────────

def sweep_miniblocks(mbs: Sequence[Any] = SWEEP_MINIBLOCKS, arms: Sequence[Any] = Z_LR,
                     rep: str = 'Z', frames: str = 'cue', stage: str = 'S1',
                     metric: str = METRIC, k: int = WINDOW_BLOCKS,
                     **load_kw) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    """`cmi` over the 2-D grid of (mini-block since switch) x (training block).

    The axis with room left. Late-block separation saturates early for the well-tuned arms, so
    what keeps developing is how *fast* separation is reached after a switch — i.e. whether the
    front moves toward the switch as training proceeds. Every mini-block is already in the
    sampled data and every one is a full colour permutation, so this costs no new compute.

    > **`cmi` cannot be read as context identity at small mini-blocks.** An RDM is invariant to
    > a global A<->B relabeling, and "still holding the previous block's context" *is* a global
    > relabeling under strict alternation — so reliable perseveration and instant correct
    > inference give the *same* `cmi`. Measured on synthetic data: 0.684 for correct inference
    > vs 0.677 for perfect perseveration. A third reading is open too, because the switch
    > surprise is context-specific (A->B and B->A move the targets in opposite directions, and
    > the transition is determined by the new context), so a checkerboard at mini-block 0 can
    > also be the error signal alone. Three incompatible explanations, one number.
    >
    > What `cmi` *does* say here is whether the representation is clustered by block at all.
    > High = clustered (by something), ~0 = not clustered. Which context is held needs the
    > directional index, not this.

    Loads each arm's pickles once and re-analyses them per mini-block. `permute=False`
    throughout — the permutation null is the expensive part and the null for a single window is
    already in `analyze_grid`'s table.
    """
    out: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for z_lr in arms:
        runs = load_arm_runs(z_lr, stage=stage, **load_kw)
        if not runs:
            print(f'  {rep} {z_lr}: no runs on disk')
            continue
        rows, centres, kept = [], None, []
        for mb in mbs:
            res = analyze_arm(z_lr, rep=rep, frames=frames, stage=stage, metric=metric, k=k,
                              windows={'w': mb}, permute=False, runs=runs)
            if res is None or 'error' in res:
                print(f'  {rep} {z_lr} mb={mb}: '
                      f'{res["error"] if res and "error" in res else "no samples"}')
                continue
            rows.append(res['curves']['w'])
            kept.append(mb)
            centres = res['centre']['w']
        if not rows:
            continue
        n = min(r.shape[1] for r in rows)
        out[(rep, z_lr)] = dict(
            z_lr=z_lr, rep=rep, mbs=kept,
            cmi=np.stack([r[:, :n] for r in rows]),     # (n_mb, n_seeds, n_block_windows)
            centre=centres[:n],
            n_seeds=rows[0].shape[0],
        )
    return out


def _mb_label(mb: Any) -> str:
    return str(mb) if not isinstance(mb, str) else mb


def plot_separation_front(sweep: Dict[Tuple[str, Any], Dict[str, Any]],
                          arms: Sequence[Any] = SWEEP_ARMS, rep: str = 'Z',
                          figsize=None, vmax: float | None = None):
    """Item 1 — `cmi` as (mini-block since switch) x (training block), one panel per arm.

    If the separation front moves toward the switch with training, the bright region grows
    *downward* as x increases. The bottom row (mini-block 0) is boxed, because `cmi` there
    cannot distinguish held-context identity — see `sweep_miniblocks`.
    """
    cells = [z for z in arms if (rep, z) in sweep]
    if not cells:
        raise ValueError('Nothing to draw.')
    fig, axes = plt.subplots(1, len(cells), squeeze=False,
                             figsize=figsize or FigSize.custom(0.85 * len(cells) + 0.6, 1.25))
    vmax = vmax or max(float(np.nanmax(np.nanmean(sweep[(rep, z)]['cmi'], axis=1)))
                       for z in cells)
    im = None
    for j, z in enumerate(cells):
        d = sweep[(rep, z)]
        M = np.nanmean(d['cmi'], axis=1)                # (n_mb, n_block_windows)
        ax = axes[0][j]
        im = ax.imshow(M, cmap='viridis', origin='lower', aspect='auto',
                       vmin=0, vmax=vmax, interpolation='none', rasterized=True,
                       extent=(d['centre'][0] - 0.5, d['centre'][-1] + 0.5,
                               -0.5, len(d['mbs']) - 0.5))
        # Mini-block 0 is not interpretable as context identity; mark it rather than hide it.
        ax.axhline(0.5, color='w', lw=0.8, ls=(0, (2, 1.5)))
        ax.set_yticks(range(len(d['mbs'])))
        ax.set_yticklabels([_mb_label(mb) for mb in d['mbs']] if j == 0 else [])
        ax.set_title(ZLR_INFO[z].label)
        ax.set_xlabel('training block')
        if j == 0:
            ax.set_ylabel('mini-block since switch')
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.028, pad=0.02)
        cb.ax.tick_params(labelsize=4.5)
        cb.set_label('cmi (clustering, not identity)', fontsize=5)
    return fig


# ── The directional template index ────────────────────────────────────────────
#
# Why cmi is not enough at early windows, in one paragraph. An RDM is invariant to a global
# A<->B relabeling of the contexts, and under strict alternation "still holding the previous
# block's context" IS a global relabeling — so reliable perseveration and instant correct
# inference produce the *same* cmi (measured on synthetic data: 0.677 vs 0.684). A third
# reading is open as well, because the switch surprise is context-specific (A->B and B->A move
# the targets in opposite directions, and the transition is determined by the new context), so
# a checkerboard at mini-block 0 can also be the error signal alone. Three incompatible
# explanations, one number: harmless in the late window, fatal at mini-block 0.
#
# The index below is not label-invariant, so it resolves that. It is the representational twin
# of `norm_err` on the same 0 / 0.5 / 1 scale, which is what lets it be plotted directly beside
# the behavioural switch-aligned curve. The ratio form also partly cancels a perturbation that
# pushes activity away from *both* templates — something a difference-of-means contrast like
# cmi cannot do.

#: Windows the template analysis samples. Every mini-block scored, plus the `late` reference the
#: templates are built from. Deliberately stops at 4: in the shortest block (9 complete
#: mini-blocks, the geometric draw's floor) 'last' resolves to 8, so a window at 8 would collide
#: with it and the block would be dropped.
TEMPLATE_WINDOWS: Dict[str, Any] = {'mb0': 0, 'mb1': 1, 'mb2': 2, 'mb3': 3, 'mb4': 4,
                                    'late': 'last'}

#: Which window the settled templates are estimated from.
TEMPLATE_REFERENCE = 'late'


def zscore_patterns(X: np.ndarray) -> np.ndarray:
    """Per-feature z-score across the pattern set — the same normalization `build_rdm` applies.

    Shared so the template index and the RDM live in the same space; a few high-variance units
    must not dominate either one.
    """
    P = np.asarray(X, float)
    sd = P.std(axis=0, keepdims=True)
    return (P - P.mean(axis=0, keepdims=True)) / np.where(sd < 1e-12, 1.0, sd)


def _dist_to(X: np.ndarray, t: np.ndarray, metric: str) -> np.ndarray:
    if metric == 'euclidean':
        return np.linalg.norm(X - t, axis=1)
    if metric == 'correlation':
        Xc = X - X.mean(axis=1, keepdims=True)
        tc = t - t.mean()
        num = Xc @ tc
        den = np.linalg.norm(Xc, axis=1) * np.linalg.norm(tc)
        return 1.0 - np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    raise ValueError(f"Unknown metric '{metric}'; expected 'euclidean' or 'correlation'.")


def template_index(samples: Dict[str, Any], reference: str = TEMPLATE_REFERENCE,
                   metric: str = 'euclidean', zscore: bool = False) -> np.ndarray:
    """Per-trial directional index. 0 = at the new context, 0.5 = ambiguous, 1 = still the old.

        d_new = dist(x, settled(context of this block))
        d_old = dist(x, settled(the other context))
        index = d_new / (d_new + d_old)

    Templates are **causal and adjacent**: `settled_old` is the reference window of block b-1 and
    `settled_new` is the reference window of block b-2 (which carries the same context as b under
    strict alternation). Three reasons, and they are the whole design:

    1. **Symmetric.** The index is a ratio of distances to two templates, so they must be
       estimated the same way. Averaging n blocks shrinks template noise from s^2 to s^2/n and
       systematically reduces distances to that template, so an asymmetric pair biases the ratio.
       One block each is symmetric by construction.
    2. **Causal.** The index is read against training block, so a template drawn from later
       blocks would leak the very axis under study.
    3. **Exact, not a proxy.** Perseveration means "still in the state block b-1 left me in", and
       late(b-1) *is* that state rather than an estimate of a population template.

    Scoring a block's early trials against its own reference window would be circular — they
    share weights, drift point and noise realisation, so `d_new` would come out low whether or
    not the context was ever inferred. Hence b-1 / b-2, and hence the first two blocks score NaN.

    > **0.5 is NOT the right reference — the `'RNN'` arm is.** The two templates are symmetric in
    > estimation (one block each) but asymmetric in *time*: b-1 is nearer than b-2, so drift alone
    > pulls the index above 0.5. Measured on the post-gate hidden state, the `'RNN'` arm (which has
    > no context representation at all) runs 0.546 at mb0 down to 0.495 at `late` — the same shape
    > a real effect has. This cannot be symmetrised away, because under strict alternation the
    > same-context template is always an even block lag and therefore always further back in time.
    > Always plot or subtract the `'RNN'` arm at the matched mini-block. On the 2-dim Z the bias is
    > negligible against the measured range (0.89 -> 0.16); on a 64-unit state it is not.

    > **One residual assumption**, and it is checked rather than assumed: if a model settled into
    > the *wrong* context in every block, data-derived templates would absorb the swap and the
    > index would read 0. Behaviour rules it out — S1 pre-switch `norm_err` is 0.145-0.253 for
    > every arm, so every arm does end its blocks in the correct context.

    `zscore=False` by default, unlike `build_rdm`. Per-feature z-scoring equalises informative
    and uninformative dimensions, so a feature carrying no context signal has its *noise* scaled
    up to unit variance — which compresses the index toward 0.5 and costs exactly the 0 / 0.5 / 1
    scale that makes it comparable to `norm_err`. Measured on synthetic data with a known held
    context: raw gives 0.910 (perseverative) / 0.092 (correct), z-scored gives 0.746 / 0.249. On
    the 2-D softmax Z the two agree to three decimals (both dimensions carry signal, so z-scoring
    is a near-affine rescaling); the difference is expected to matter on the 64-unit hidden state.
    Both compress rather than flip, so the reading against 0.5 is robust either way.
    """
    X = zscore_patterns(samples['X']) if zscore else np.asarray(samples['X'], float)
    block, ctx, win = samples['block'], samples['context'], samples['window']
    blocks = np.unique(block)

    centroid = {}
    for b in blocks:
        sel = (block == b) & (win == reference)
        if sel.any():
            centroid[b] = X[sel].mean(axis=0)

    out = np.full(len(X), np.nan)
    for i, b in enumerate(blocks):
        if i < 2:
            continue                                  # no b-1 / b-2 yet
        b_old, b_new = blocks[i - 1], blocks[i - 2]
        if b_old not in centroid or b_new not in centroid:
            continue
        c_b = ctx[block == b][0]
        # A dropped block would break the alternation the b-2 trick relies on; verify rather
        # than assume, and skip the block if it does not hold.
        if ctx[block == b_old][0] == c_b or ctx[block == b_new][0] != c_b:
            continue
        rows = np.where(block == b)[0]
        d_new = _dist_to(X[rows], centroid[b_new], metric)
        d_old = _dist_to(X[rows], centroid[b_old], metric)
        den = d_new + d_old
        out[rows] = np.where(den > 0, d_new / np.where(den > 0, den, 1.0), np.nan)
    return out


def analyze_templates(z_lr: Any, rep: str = 'Z', frames: str = 'cue', stage: str = 'S1',
                      windows: Dict[str, Any] | None = None,
                      reference: str = TEMPLATE_REFERENCE, metric: str = 'euclidean',
                      zscore: bool = False,
                      runs: List[Tuple[Any, Any]] | None = None,
                      **load_kw) -> Dict[str, Any] | None:
    """One arm, every seed: the directional index per window, per block.

    Returns dict(idx={window: (n_seeds, n_blocks)}, blocks, window_names, ...). Unlike the RDM
    path nothing is averaged across seeds before the index is formed — the index is per trial,
    so each seed gets its own and they are pooled only at the reporting step.
    """
    if windows is None:
        windows = dict(TEMPLATE_WINDOWS)
    if runs is None:
        runs = load_arm_runs(z_lr, stage=stage, **load_kw)
    if not runs:
        return None

    per_seed = []
    for logger, config in runs:
        try:
            s = sample_trials(logger, config, rep=rep, frames=frames, windows=windows)
        except ValueError as exc:
            return dict(error=str(exc), z_lr=z_lr, rep=rep, frames=frames)
        if s is None:
            continue
        s['index'] = template_index(s, reference=reference, metric=metric, zscore=zscore)
        per_seed.append(s)
    if not per_seed:
        return None

    names = list(per_seed[0]['windows'])
    n_blocks = min(s['n_blocks'] for s in per_seed)
    idx: Dict[str, np.ndarray] = {}
    for w in names:
        rows = []
        for s in per_seed:
            blocks = np.unique(s['block'])[:n_blocks]
            # All-NaN is the right answer where there is no representation to score: the 'RNN'
            # arm's Z is exactly constant, so both templates coincide and the ratio is 0/0.
            # Suppress the empty-slice warning rather than let it imply something went wrong.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                rows.append([np.nanmean(s['index'][(s['block'] == b) & (s['window'] == w)])
                             for b in blocks])
        idx[w] = np.asarray(rows, dtype=float)        # (n_seeds, n_blocks)

    if all(np.all(np.isnan(v)) for v in idx.values()):
        return dict(z_lr=z_lr, rep=rep, frames=frames, stage=stage, metric=metric,
                    idx=idx, window_names=names, n_blocks=n_blocks, n_seeds=len(per_seed),
                    n_dropped=int(sum(s['n_dropped'] for s in per_seed)),
                    degenerate='representation is constant — both templates coincide')

    return dict(z_lr=z_lr, rep=rep, frames=frames, stage=stage, metric=metric,
                idx=idx, window_names=names, n_blocks=n_blocks, n_seeds=len(per_seed),
                n_dropped=int(sum(s['n_dropped'] for s in per_seed)))


def template_grid(arms: Sequence[Any] = Z_LR, rep: str = 'Z', **kw
                  ) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    """`analyze_templates` for every arm. Cells that cannot be read are reported and skipped."""
    out: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for z_lr in arms:
        res = analyze_templates(z_lr, rep=rep, **kw)
        if res is None:
            print(f'  {rep} {z_lr}: no runs on disk')
            continue
        if 'error' in res:
            print(f'  {rep} {z_lr}: {res["error"]}')
            continue
        out[(rep, z_lr)] = res
    return out


def summarize_templates(grid: Dict[Tuple[str, Any], Dict[str, Any]]) -> None:
    """Index per window, split into the first and second half of training.

    0 = switched to the new context, 0.5 = ambiguous, 1 = still in the old one. The first two
    blocks of every run are NaN by construction (no b-1 / b-2), so `n` is the scored count.
    """
    for (rep, z_lr), res in grid.items():
        names = res['window_names']
        print(f'\n  {rep}  Z_lr={z_lr}   ({res["n_seeds"]} seeds, {res["n_blocks"]} blocks, '
              f'metric={res["metric"]})')
        if 'degenerate' in res:
            print(f'      not scoreable: {res["degenerate"]}')
            continue
        print('      window ' + ' '.join(f'{w:>7}' for w in names))
        half = res['n_blocks'] // 2
        for label, sl in (('all', slice(None)), ('early tr', slice(0, half)),
                          ('late tr', slice(half, None))):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                vals = [np.nanmean(res['idx'][w][:, sl]) for w in names]
            print(f'  {label:>10} ' + ' '.join(f'{v:>7.3f}' for v in vals))


def behavioural_by_miniblock(z_lr: Any, stage: str = 'S1', n_colors: int = 5,
                             **load_kw) -> Dict[str, np.ndarray]:
    """Per-seed behavioural `norm_err` binned into mini-blocks since the switch.

    The representational index is on the same 0 / 0.5 / 1 scale as `norm_err`, so the two can be
    read on one axis. The alignment is off by one name and it matters: the behavioural
    `curve_summary` calls trials 1..n_colors `mb1`, which is mini-block index **0** here.

    The behavioural window stops at `AnalysisParams.n_post = 14` trials, so only mini-blocks 0
    and 1 are fully covered and 2 is partial; later mini-blocks come back NaN.
    """
    from rotation_curriculum_analysis import (AnalysisParams, _frac_for, switch_aligned_error,
                                              trial_axis)

    params = AnalysisParams()
    frac = _frac_for(stage, params)
    x = trial_axis(params)
    out: Dict[str, List[float]] = {}
    for logger, config in load_arm_runs(z_lr, stage=stage, **load_kw):
        arr = switch_aligned_error(logger, config, params, frac)
        if arr.shape[0] == 0 or np.all(np.isnan(arr)):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean = np.nanmean(arr, axis=0)
            for mb in range(3):
                lo, hi = mb * n_colors + 1, (mb + 1) * n_colors
                sel = (x >= lo) & (x <= hi)
                out.setdefault(f'mb{mb}', []).append(
                    float(np.nanmean(mean[sel])) if sel.any() else np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def plot_template_index(grid: Dict[Tuple[str, Any], Dict[str, Any]],
                        arms: Sequence[Any] = Z_LR, rep: str = 'Z',
                        behaviour: bool = True, figsize=None, **load_kw):
    """Items 3 and 4 — the directional index, and whether perseveration develops.

    Left: index against mini-block since switch, one line per arm. 0 = at the new context,
    0.5 = ambiguous, 1 = still in the old one. Behavioural `norm_err` over the same mini-blocks
    is overlaid dashed in the same colour where the behavioural window covers it, which is the
    point of matching the scale.

    Right: the mini-block-0 index against training block — does the model commit to the *old*
    context more or less readily as training proceeds?

    The `'RNN'` arm is absent by construction, not omitted: its Z is constant, so both templates
    coincide and the ratio is 0/0 (`summarize_templates` reports it as not scoreable).
    """
    cells = [z for z in arms if (rep, z) in grid and 'degenerate' not in grid[(rep, z)]]
    if not cells:
        raise ValueError('Nothing to draw.')
    fig, axes = plt.subplots(1, 2, figsize=figsize or FigSize.row(2))
    ax, ax2 = axes

    n_seeds = 0
    for z in cells:
        res = grid[(rep, z)]
        names = res['window_names']
        n_seeds = max(n_seeds, res['n_seeds'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mu = np.asarray([np.nanmean(res['idx'][w]) for w in names])
            se = np.asarray([np.nanstd(np.nanmean(res['idx'][w], axis=1))
                             / np.sqrt(res['n_seeds']) for w in names])
        xx = np.arange(len(names), dtype=float)
        info = ZLR_INFO[z]
        ax.plot(xx, mu, color=info.color, label=info.label, marker='o')
        ax.fill_between(xx, mu - se, mu + se, color=info.color, alpha=0.22, lw=0)

        if behaviour:
            b = behavioural_by_miniblock(z, **load_kw)
            bx = [i for i, w in enumerate(names) if w in b]
            by = [float(np.nanmean(b[names[i]])) for i in bx]
            ax.plot(bx, by, color=info.color, ls=(0, (2, 1.5)), marker='^',
                    markerfacecolor='none', lw=0.7)

        # The first two blocks are all-NaN for every seed by construction (no b-1 / b-2), so
        # the leading columns are empty slices. Expected, not a problem.
        mb0 = res['idx'][names[0]]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            m0 = np.nanmean(mb0, axis=0)
            s0 = np.nanstd(mb0, axis=0) / np.sqrt(mb0.shape[0])
        xb = np.arange(len(m0))
        ax2.plot(xb, m0, color=info.color)
        ax2.fill_between(xb, m0 - s0, m0 + s0, color=info.color, alpha=0.22, lw=0)

    for a in axes:
        a.axhline(0.5, color='0.6', lw=0.6, zorder=0)     # ambiguous
        a.set_ylim(0, 1)
    names = grid[(rep, cells[0])]['window_names']
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.tick_params(axis='x', length=0)
    ax.set_xlabel('window (mini-blocks since switch)')
    ax.set_ylabel('index (1 = old context)')
    ax2.set_xlabel('training block')
    ax2.set_ylabel('index at mini-block 0')
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    fig.supxlabel(f'{rep}, cue frames · solid = representation, dashed = behaviour '
                  f'(norm_err) · mean ± SEM over {n_seeds} seeds')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    if not self_test():
        raise SystemExit('self-test failed — not running the analysis')
    main()


# ── Per-unit context discriminability ─────────────────────────────────────────
#
# Why this exists beside `cmi`. `cmi` is a *relative* distance contrast, so its scale depends on
# how many dimensions carry the effect. For unit-variance features with per-unit separation d',
# between-context squared distance is 2 + d'^2 per feature against 2 within, so
#
#     cmi ~= sqrt(1 + d'^2 / 2) - 1
#
# A 64-unit population at d' = 0.93 therefore gives cmi ~= 0.20 — which is exactly what the
# post-gate hidden state measures. Read naively against the 2-dimensional Z's cmi of ~1.1 that
# looks like a weak effect; it is not. **`cmi` must not be compared across representations of
# different dimensionality**, and d' is the scale that can be.

def unit_discriminability(samples: Dict[str, Any]) -> np.ndarray:
    """Per-unit |d'| between the two contexts, matched on colour.

    Matching on colour holds the stimulus constant, so what is left is context. Returns one
    value per feature; `nan` for a dead unit (zero variance within every colour).
    """
    X, ctx, col = samples['X'], samples['context'], samples['colour']
    rows = []
    for c in np.unique(col):
        k = col == c
        a, b = X[k & (ctx == 0)], X[k & (ctx == 1)]
        if len(a) < 3 or len(b) < 3:
            continue
        sd = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2.0)
        rows.append(np.abs(a.mean(axis=0) - b.mean(axis=0))
                    / np.where(sd < 1e-9, np.nan, sd))
    if not rows:
        return np.full(X.shape[1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(rows, axis=0)


def discriminability_grid(arms: Sequence[Any] = Z_LR, reps: Sequence[str] = ('Z', 'H', 'H_pre'),
                          frames: str = 'cue', thresh: float = 0.5,
                          **load_kw) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    """`unit_discriminability` per (rep, arm), averaged over seeds."""
    out: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for rep in reps:
        for z_lr in arms:
            runs = load_arm_runs(z_lr, stage='S1', **load_kw)
            if not runs:
                continue
            rows, dead = [], []
            for logger, config in runs:
                try:
                    s = sample_trials(logger, config, rep=rep, frames=frames)
                except (ValueError, KeyError):
                    rows = []
                    break
                if s is None:
                    continue
                rows.append(unit_discriminability(s))
                dead.append(int((s['X'].std(axis=0) < 1e-9).sum()))
            if not rows:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                d = np.nanmean(rows, axis=0)
                out[(rep, z_lr)] = dict(
                    d=d, mean=float(np.nanmean(d)), max=float(np.nanmax(d)),
                    n_above=int(np.nansum(d > thresh)), n_units=len(d),
                    n_dead=int(np.mean(dead)) if dead else 0, n_seeds=len(rows))
    return out


def summarize_discriminability(grid: Dict[Tuple[str, Any], Dict[str, Any]],
                               thresh: float = 0.5) -> None:
    """One row per (rep, arm). The `'RNN'` arm is the no-context floor to read the others against."""
    print(f'\n{"rep":>6} {"Z_lr":>6} {"mean|d|":>8} {"max|d|":>7} '
          f'{f"units>{thresh}":>11} {"dead":>5} {"units":>6}')
    print('-' * 56)
    for (rep, z_lr), r in grid.items():
        print(f'{rep:>6} {str(z_lr):>6} {r["mean"]:>8.3f} {r["max"]:>7.3f} '
              f'{r["n_above"]:>11} {r["n_dead"]:>5} {r["n_units"]:>6}')
