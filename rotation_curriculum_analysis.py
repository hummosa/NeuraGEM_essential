"""Switch-aligned adaptation analysis for the three-stage cued-context curriculum.

The question this module answers is the paper's: **after an unsignalled rotation switch, how fast
does the model's behaviour move from the old rotation to the new one, and does it get the whole
map back from a single observation?**

The metric is the normalized state error from Yu et al., already implemented in
rotating_targets_analysis._analyze_adaptation and reused here unchanged in spirit:

    norm_err = ||pred - target(current_rotation, colour)||
             / (||pred - target(current_rotation, colour)|| + ||pred - target(other, colour)||)

    0.0 = predicting the correct target for this trial's rotation
    0.5 = exactly between the two candidate targets (chance)
    1.0 = predicting the target the *other* rotation would put there

No belief head, no criterion runs, no perseveration/slip counting — just where the prediction
sits between the two rotations, trial by trial.

## The plot

x is trials, where one trial is a cue timestep plus its outcome timestep.

    x = 0   the LAST trial of the previous block   (the switch happens between 0 and 1)
    x < 0   earlier trials of the previous block   — the asymptotic, pre-switch state
    x = 1   the FIRST trial of the new block       — the model has seen no evidence yet, so this
                                                     is its prior: ~1.0 means fully committed to
                                                     the old rotation
    x = 2   the second trial, a colour that has NOT been seen under the new rotation. If the
            model infers the global rotation from trial 1's single observation, this is already
            low — that is the zero-shot reuse the task is built to test
    x = 3+  the rest, covering ~3 mini-blocks so every colour appears at least twice

Mini-block ends are marked: with n_colors=5 every colour has appeared exactly once by x=5.

## Windows

Flexibility is only meaningful once a model has learned the task, so S1 and S2 are restricted to
the **last third of their blocks**. S3 freezes the weights, so nothing is learned during it and
all of its blocks are used.

## Averaging

Two levels, so that adding seeds does the right thing automatically:
  1. within a run, mean over the switches in the window  -> one curve per seed
  2. across seeds, mean +/- SEM
With a single seed there is no across-seed spread, so the band falls back to SEM across switches
and `AdaptationCurve.sem_over` records which was used.

Usage:
    python rotation_curriculum_analysis.py
"""

from __future__ import annotations

try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic('load_ext', 'autoreload')
        _ip.run_line_magic('autoreload', '2')
except Exception:
    pass

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

import plot_style
from plot_style import FigSize

from rotating_targets_analysis import (
    _nearest_rotation_deg, flatten_logger, get_block_switches, get_target_positions,
)
from rotation_curriculum_config import (
    CUE_INFO, CUE_MODES, EXPORT_ROOT, HEADLINE_ZLR, PINNED_Z_INFO, ZLR_INFO,
    Z_LR, active_noise, active_seeds, result_path, zlr_label,
)

plot_style.set_plot_style()

HEADLINE_NOISE = active_noise()[0]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class AnalysisParams:
    n_seeds:  int   = field(default_factory=active_seeds)
    noise:    float = HEADLINE_NOISE

    # Trial window around the switch. n_pre=2 puts x at [-1, 0]; n_post=14 gives x at [1..14],
    # which is 2.8 mini-blocks at n_colors=5 — every colour appears at least twice.
    n_pre:    int   = 2
    n_post:   int   = 14

    # Which rotation counts as "current" when scoring a trial.
    #   'trial' — the rotation actually in effect on that trial. Pre-switch trials then sit near
    #             0 (the model is right about the old block), the curve spikes at x=1 and decays.
    #             This is what makes x<=0 readable as the asymptotic state.
    #   'new'   — always the post-switch rotation, so pre-switch sits near 1. Useful if you want
    #             a single monotone "distance from the new context" curve.
    anchor:   str   = 'trial' #'new'

    # Fraction of each stage's blocks to use, taken from the END of the stage. Flexibility is
    # only meaningful once the task is learned, and S1/S2 are still learning; S3 freezes the
    # weights, so all of its blocks are equivalent and all are used.
    block_frac: Dict[str, float] = field(
        default_factory=lambda: {'S1': 1 / 3, 'S2': 1 / 3, 'S3': 1.0})

    phase_name: str  = 'Learning and inference'
    dpi:        int  = 160
    show_plots: bool = True
    save_plots: bool = True
    linewidth:  float = 1.5
    alpha:      float = 0.9
    band_alpha: float = 0.18


def _frac_for(stage: Any, params: AnalysisParams) -> float:
    key = stage if isinstance(stage, str) else stage[0]
    return params.block_frac.get(key, 1.0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

Stage = Any   # 'S1' | ('S2', cue) | ('S3', cue) | ('S3pinned', cue)


def stage_label(stage: Stage) -> str:
    if stage == 'S1':
        return 'S1 uncued'
    if stage[0] == 'S2':
        return f'S2 {CUE_INFO[stage[1]].label.lower()}'
    if stage[0] == 'S3':
        return f'S3 {CUE_INFO[stage[1]].label.lower()}'
    return f'S3 {CUE_INFO[stage[1]].label.lower()} ({PINNED_Z_INFO.label.lower()})'


def _get_stage(payload, stage: Stage):
    """(logger, config) for one stage of one tree, or (None, None) if absent."""
    if stage == 'S1':
        return payload['S1'], payload['configs']['S1']
    kind, cue = stage[0], stage[1]
    key = 'S3_pinned' if kind == 'S3pinned' else kind
    return payload[key].get(cue), payload['configs'][key].get(cue)


def load_trees(params: AnalysisParams, noise: float | None = None) -> Dict[Tuple, Any]:
    """{(z_lr, noise, seed): payload} for every tree on disk."""
    noises = active_noise() if noise is None else [noise]
    cache, missing = {}, []
    for z_lr in Z_LR:
        for n in noises:
            for seed in range(params.n_seeds):
                path = result_path(z_lr, n, seed)
                if not path.exists():
                    missing.append((z_lr, n, seed))
                    continue
                with path.open('rb') as f:
                    cache[(z_lr, n, seed)] = pickle.load(f)
    print(f'Loaded {len(cache)} trees from {EXPORT_ROOT}')
    if missing:
        print(f'  missing {len(missing)}: {missing[:6]}{" ..." if len(missing) > 6 else ""}')
    return cache


def stage_runs(cache, z_lr: Any, stage: Stage,
               noise: float | None = None) -> List[Tuple[Any, Any]]:
    """[(logger, config), ...] over seeds for one (Z_lr, stage) cell."""
    out = []
    for (t, n, _), payload in sorted(cache.items(), key=lambda kv: kv[0][2]):
        if t != z_lr or (noise is not None and n != noise):
            continue
        logger, config = _get_stage(payload, stage)
        if logger is not None:
            out.append((logger, config))
    return out


# ---------------------------------------------------------------------------
# Switch-aligned normalized state error
# ---------------------------------------------------------------------------

def _phase_range(logger, phase_name: str) -> Tuple[int, int]:
    """[start, end) flat timestep indices of one named phase, or the whole log if absent."""
    n = sum(len(np.asarray(e).reshape(-1)) for e in logger.context_ids)
    for i, (name, ts) in enumerate(logger.phases):
        if name == phase_name:
            end = logger.phases[i + 1][1] if i + 1 < len(logger.phases) else n
            return int(ts), int(end)
    return 0, n


def trial_axis(params: AnalysisParams) -> np.ndarray:
    """Trial indices relative to the switch: [-(n_pre-1) .. 0, 1 .. n_post]."""
    return np.arange(-(params.n_pre - 1), params.n_post + 1)


def switch_aligned_error(logger, config, params: AnalysisParams,
                         block_frac: float = 1.0) -> np.ndarray:
    """Normalized state error per switch, per trial. Shape (n_switches, len(trial_axis)).

    A trial is a cue frame and its outcome frame. `oi[t]` is the prediction *of* frame `ii[t]`,
    so for a cue frame at `t` the predicted attack is `oi[t+1, -2:]` — the same indexing
    rotating_targets_analysis._analyze_adaptation uses.

    Trials are counted off the switch: x=1 is the first cue frame at or after it, x=0 the last
    cue frame before it. Trials that would fall outside the block they belong to (a block shorter
    than n_post trials, or the very start/end of the log) are left NaN rather than borrowed from
    a neighbouring block.
    """
    ii, oi, ll, _ = flatten_logger(logger, config)
    nc = config.n_colors
    x  = trial_axis(params)

    t0, t1 = _phase_range(logger, params.phase_name)
    switches = get_block_switches(ll, max(1, t0), t1)
    # Last `block_frac` of the blocks in this phase. Flexibility only means something once the
    # task is learned, and the first blocks of S1/S2 are still learning it.
    if 0 < block_frac < 1 and len(switches) > 1:
        switches = switches[int(round(len(switches) * (1 - block_frac))):]
    if not switches:
        return np.full((0, len(x)), np.nan)

    cue = np.flatnonzero(ii[:, :nc].sum(axis=1) > 0.5)
    out = np.full((len(switches), len(x)), np.nan)

    for si, t_sw in enumerate(switches):
        rot_new, rot_old = ll[t_sw], ll[t_sw - 1]
        deg_new = _nearest_rotation_deg(rot_new, config.train_rotations)
        deg_old = _nearest_rotation_deg(rot_old, config.train_rotations)
        if deg_new is None or deg_old is None or deg_new == deg_old:
            continue
        targets = {deg_new: get_target_positions(config, deg_new),
                   deg_old: get_target_positions(config, deg_old)}

        j = int(np.searchsorted(cue, t_sw))        # cue index of trial x = 1
        for col, k in enumerate(x):
            ci = j + k - 1                          # x=1 -> j, x=0 -> j-1, x=-1 -> j-2
            if ci < 0 or ci >= len(cue):
                continue
            t = int(cue[ci])
            if t + 1 >= len(ll) or t < t0 or t >= t1:
                continue
            # Stay inside the intended block: a short block must not borrow trials from the next.
            want = rot_new if k >= 1 else rot_old
            if abs(ll[t] - want) > 1e-5:
                continue

            if params.anchor == 'new':
                deg_cur, deg_alt = deg_new, deg_old
            elif params.anchor == 'trial':
                deg_cur, deg_alt = ((deg_new, deg_old) if k >= 1 else (deg_old, deg_new))
            else:
                raise ValueError(f"anchor must be 'trial' or 'new'; got {params.anchor!r}")

            colour = int(np.argmax(ii[t, :nc]))
            pred   = oi[t + 1, -2:]
            d_cur  = np.linalg.norm(pred - targets[deg_cur][colour])
            d_alt  = np.linalg.norm(pred - targets[deg_alt][colour])
            out[si, col] = d_cur / (d_cur + d_alt + 1e-9)

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class AdaptationCurve:
    x:          np.ndarray
    mean:       np.ndarray
    sem:        np.ndarray
    per_seed:   np.ndarray   # (n_seeds, n_trials) — each row already averaged over its switches
    n_seeds:    int
    n_switches: int
    sem_over:   str          # 'seeds' | 'switches'


def adaptation_curve(runs, params: AnalysisParams, block_frac: float = 1.0
                     ) -> AdaptationCurve | None:
    """Switch-aligned curve for a set of runs, averaged within seed then across seeds."""
    per_seed, pooled = [], []
    for logger, config in runs:
        arr = switch_aligned_error(logger, config, params, block_frac)
        if arr.shape[0] == 0 or np.all(np.isnan(arr)):
            continue
        with np.errstate(invalid='ignore'):
            per_seed.append(np.nanmean(arr, axis=0))
        pooled.append(arr)
    if not per_seed:
        return None

    per_seed = np.stack(per_seed)
    pooled_arr = np.concatenate(pooled, axis=0)
    mean = np.nanmean(per_seed, axis=0)

    if len(per_seed) > 1:
        n = np.sum(~np.isnan(per_seed), axis=0).clip(1)
        sem, sem_over = np.nanstd(per_seed, axis=0, ddof=1) / np.sqrt(n), 'seeds'
    else:
        # One seed: there is no across-seed spread to show. Fall back to the switch-level spread
        # and say so, rather than drawing a zero-width band that implies certainty.
        n = np.sum(~np.isnan(pooled_arr), axis=0).clip(1)
        sem, sem_over = np.nanstd(pooled_arr, axis=0, ddof=1) / np.sqrt(n), 'switches'

    return AdaptationCurve(x=trial_axis(params), mean=mean, sem=np.nan_to_num(sem),
                           per_seed=per_seed, n_seeds=len(per_seed),
                           n_switches=int(pooled_arr.shape[0]), sem_over=sem_over)


_CURVE_MEMO: Dict[Tuple, Any] = {}


def clear_memo() -> None:
    """Drop the curve memo. Call after reloading trees or changing AnalysisParams."""
    _CURVE_MEMO.clear()


def stage_curve(cache, z_lr: Any, stage: Stage,
                params: AnalysisParams) -> AdaptationCurve | None:
    """Memoized adaptation_curve for one (Z_lr, stage) cell."""
    key = (id(cache), z_lr, stage if isinstance(stage, str) else tuple(stage),
           params.noise, params.n_pre, params.n_post, params.anchor, _frac_for(stage, params))
    if key not in _CURVE_MEMO:
        runs = stage_runs(cache, z_lr, stage, params.noise)
        _CURVE_MEMO[key] = (adaptation_curve(runs, params, _frac_for(stage, params))
                            if runs else None)
    return _CURVE_MEMO[key]


# ---------------------------------------------------------------------------
# Scalar read-outs off a curve
# ---------------------------------------------------------------------------

def curve_summary(curve: AdaptationCurve, n_colors: int) -> Dict[str, float]:
    """The numbers the plot is read for.

        pre    mean over x <= 0 — the asymptotic, pre-switch state
        t1     x = 1, the first trial of the new block. No evidence has been seen yet, so this is
               the model's prior: ~1 means fully committed to the old rotation
        t2     x = 2, a colour never seen under the new rotation. Low here means the model
               generalised the rotation from ONE observation — the zero-shot reuse the task tests
        reuse  t1 - t2, how much that single observation bought
        mb1    mean over the first mini-block (x in [1, n_colors])
        mb2    mean over the second mini-block
        asym   mean over the last mini-block in the window
    """
    x, m = curve.x, curve.mean
    def _at(k):
        i = np.flatnonzero(x == k)
        return float(m[i[0]]) if len(i) and not np.isnan(m[i[0]]) else np.nan
    def _mean(lo, hi):
        sel = (x >= lo) & (x <= hi)
        return float(np.nanmean(m[sel])) if sel.any() else np.nan

    nc   = n_colors
    last = int(x.max())
    t1, t2 = _at(1), _at(2)
    return dict(
        pre   = _mean(int(x.min()), 0),
        t1    = t1,
        t2    = t2,
        reuse = t1 - t2 if not (np.isnan(t1) or np.isnan(t2)) else np.nan,
        mb1   = _mean(1, nc),
        mb2   = _mean(nc + 1, 2 * nc),
        asym  = _mean(max(1, last - nc + 1), last),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _row2():
    """Two panels side by side, with headroom for the legend row and the note beneath.

    FigSize.row(2) alone leaves the axes squeezed once an outside legend and a supxlabel are
    added — the drawing area, not the font, is what suffers. Fixing the squeeze is the remedy the
    style guide asks for; enlarging the whole figure to make text bigger is not.
    """
    w, h = FigSize.row(2)
    return FigSize.custom(w, h + 0.45)


def _decorate(ax, params: AnalysisParams, n_colors: int, ylabel: bool = True) -> None:
    """Chance line, the switch, mini-block ends, axis labels and ticks."""
    x = trial_axis(params)
    ax.axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.35)
    # The switch happens BETWEEN trial 0 (last of the old block) and trial 1 (first of the new).
    ax.axvline(0.5, color='k', linewidth=0.9, alpha=0.55)
    # Mini-block ends: by x = n_colors every colour has appeared exactly once under the new
    # rotation, so anything below chance before then is transfer rather than experience.
    for b in range(n_colors, params.n_post + 1, n_colors):
        ax.axvline(b + 0.5, color='0.6', linewidth=0.5, linestyle=(0, (2, 2)), alpha=0.6)
    ax.set_xlim(x.min() - 0.4, x.max() + 0.4)
    ax.set_ylim(-0.04, 1.04)
    # Pin the y ticks: these panels are short, and matplotlib's autoticker drops 0.5 — the one
    # value that has to be readable, since it is the chance level the curves are judged against.
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks([x.min(), 0] + list(range(n_colors, params.n_post + 1, n_colors)))
    ax.set_xlabel('Trial relative to switch', labelpad=1)
    if ylabel:
        ax.set_ylabel('Normalized state error\n(0 = new rotation, 1 = old)')


def plot_adaptation(ax, entries: Sequence[Tuple[str, Any, dict]], params: AnalysisParams,
                    n_colors: int, ylabel: bool = True) -> int:
    """Draw curves on one axis. `entries` is [(label, AdaptationCurve, style_kwargs), ...]."""
    n = 0
    for label, curve, style in entries:
        if curve is None:
            continue
        kw = dict(linewidth=params.linewidth, alpha=params.alpha, marker='o',
                  markersize=2.6, markeredgewidth=0)
        kw.update(style)
        color = kw.get('color', 'grey')
        ax.plot(curve.x, curve.mean, label=label, **kw)
        ax.fill_between(curve.x, curve.mean - curve.sem, curve.mean + curve.sem,
                        color=color, alpha=params.band_alpha, linewidth=0)
        n += 1
    _decorate(ax, params, n_colors, ylabel=ylabel)
    return n


def _n_colors(cache) -> int:
    for payload in cache.values():
        return payload['configs']['S1'].n_colors
    return 5


def _save(fig, export_dir: Path, name: str, params: AnalysisParams):
    if params.save_plots:
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / name
        fig.savefig(out, bbox_inches='tight')
        print(f'  Saved -> {out}')
    if params.show_plots:
        plt.show()
    else:
        plt.close(fig)


def _sem_note(curves) -> str:
    over = {c.sem_over for c in curves if c is not None}
    seeds = max([c.n_seeds for c in curves if c is not None], default=0)
    sw = min([c.n_switches for c in curves if c is not None], default=0)
    return (f'mean ± SEM over {"seeds" if over == {"seeds"} else "switches"}; '
            f'n_seeds={seeds}, ≥{sw} switches/curve')


def _annotate_note(fig, curves) -> None:
    """Put the averaging note below the axes.

    Not a suptitle: these figures carry an `outside upper center` legend, and a suptitle lands on
    top of it under constrained layout.
    """
    fig.supxlabel(_sem_note(curves), fontsize=5, color='0.4')


# ── Shared helper: one stage, every Z_lr overlaid ─────────────────────────────

def plot_stage_by_zlr(cache, params: AnalysisParams, export_dir: Path, stage_kind: str,
                      cue_mode: str | None, title: str, fname: str) -> plt.Figure:
    """One stage, every Z_lr value on one axis — the headline dose-response shape.

    `stage_kind` in {'S1', 'S2', 'S3'}. S1 does not fork on cue_mode, so pass `cue_mode=None`
    there. Every individual (Z_lr) ran its own complete S1->S2->S3, so this is simply "read off
    the curve at each Z_lr for this one stage" — no forking involved any more.
    """
    nc = _n_colors(cache)
    fig, ax = plt.subplots(figsize=FigSize.wide, dpi=params.dpi, layout='constrained')
    drawn, entries = [], []
    for z in Z_LR:
        stage = 'S1' if stage_kind == 'S1' else (stage_kind, cue_mode)
        c = stage_curve(cache, z, stage, params)
        info = ZLR_INFO[z]
        entries.append((info.label, c, dict(color=info.color)))
        drawn.append(c)
    plot_adaptation(ax, entries, params, nc)
    ax.set_title(title, fontsize=6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncols=min(len(entries), 8),
              fontsize=5, frameon=False, handlelength=1.0, columnspacing=0.7,
              handletextpad=0.3)
    _annotate_note(fig, drawn)
    _save(fig, export_dir, fname, params)
    return fig


# ── A/B/C: the headline dose-response, one figure per stage ──────────────────

def plot_s1_dose_response(cache, params: AnalysisParams, export_dir: Path) -> plt.Figure:
    """A — S1, uncued, every Z_lr. Shows which individuals fail to adapt after a switch at all."""
    return plot_stage_by_zlr(cache, params, export_dir, 'S1', None,
                             'S1: uncued baseline', 'A_s1_by_zlr.pdf')


def plot_s2_dose_response(cache, params: AnalysisParams, export_dir: Path) -> plt.Figure:
    """B — S2, cued, every Z_lr. With ground truth handed to Z there is nothing left to infer, so
    curves should collapse toward the cue's own floor; any residual spread is forgetting, not
    inference speed."""
    return plot_stage_by_zlr(cache, params, export_dir, 'S2', 'oracle_z',
                             'S2: cued', 'B_s2_by_zlr.pdf')


def plot_s3_dose_response(cache, params: AnalysisParams, export_dir: Path) -> plt.Figure:
    """C — S3, uncued again, weights frozen, every Z_lr. The real recovered dose-response: each
    individual's own Z_lr, read off its own behaviour."""
    return plot_stage_by_zlr(cache, params, export_dir, 'S3', 'oracle_z',
                             'S3: uncued, cue withdrawn', 'C_s3_by_zlr.pdf')


# ── F3-equivalent: S1 vs S3, before/after in one figure ───────────────────────

def plot_s1_vs_s3_by_zlr(cache, params: AnalysisParams, export_dir: Path,
                         cue_mode: str = 'oracle_z') -> plt.Figure:
    """Compact before/after companion to A/B/C: every Z_lr at S1 and at S3, side by side."""
    nc = _n_colors(cache)
    fig, axes = plt.subplots(1, 2, figsize=_row2(), dpi=params.dpi,
                             sharey=True, layout='constrained')
    drawn = []
    for ax, stage, title in ((axes[0], 'S1', 'S1 uncued'),
                             (axes[1], ('S3', cue_mode), 'S3 uncued, cue withdrawn')):
        entries = []
        for z in Z_LR:
            c = stage_curve(cache, z, stage, params)
            entries.append((ZLR_INFO[z].label, c, dict(color=ZLR_INFO[z].color)))
            drawn.append(c)
        plot_adaptation(ax, entries, params, nc, ylabel=(ax is axes[0]))
        ax.set_title(title, fontsize=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncols=min(len(handles), 8),
               fontsize=5, frameon=False, handlelength=1.0, columnspacing=0.7)
    _annotate_note(fig, drawn)
    _save(fig, export_dir, 'F_s1_vs_s3_by_zlr.pdf', params)
    return fig


# ── Internal sanity figure: the uncued-control arm ─────────────────────────────
#
# cue_mode='none' is OUR check that the cued arm is being compared against a fair amount of extra
# training, not a result to show an audience — it has no story of its own. Kept as one combined
# figure, not folded into the headline A/B/C set.

def plot_uncued_control_sanity(cache, params: AnalysisParams, export_dir: Path) -> plt.Figure:
    """Internal sanity check only: S1/S2/S3 for the cue_mode='none' control arm, every Z_lr.

    Not for the paper — this is the paired control that lets F1's cued curves be read as "what
    the cue did" rather than "what more training did".
    """
    nc = _n_colors(cache)
    fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3), dpi=params.dpi,
                             sharey=True, layout='constrained')
    drawn = []
    for ax, stage, title in ((axes[0], 'S1', 'S1 uncued'),
                             (axes[1], ('S2', 'none'), 'S2 uncued control'),
                             (axes[2], ('S3', 'none'), 'S3 uncued control')):
        entries = []
        for z in Z_LR:
            c = stage_curve(cache, z, stage, params)
            entries.append((ZLR_INFO[z].label, c, dict(color=ZLR_INFO[z].color)))
            drawn.append(c)
        plot_adaptation(ax, entries, params, nc, ylabel=(ax is axes[0]))
        ax.set_title(title, fontsize=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('INTERNAL SANITY CHECK — uncued control (cue_mode="none")', fontsize=6, y=1.08)
    fig.legend(handles, labels, loc='outside upper center', ncols=min(len(handles), 8),
               fontsize=5, frameon=False, handlelength=1.0, columnspacing=0.7)
    _annotate_note(fig, drawn)
    _save(fig, export_dir, 'sanity_uncued_control.pdf', params)
    return fig


# ── F4-equivalent: the two scalars the curve is read for ─────────────────────

_ZLR_X = {z: i for i, z in enumerate(Z_LR)}


def plot_reuse_summary(cache, params: AnalysisParams, export_dir: Path) -> plt.Figure:
    """The prior (trial 1) and the zero-shot reuse (trial 2) against Z_lr, cued vs control.

    Trial 1 is scored before any evidence from the new block, so it measures commitment to the
    old rotation. Trial 2 is the first colour that was never observed under the new rotation, so
    a low value there means one observation was enough to move the whole map. One line per
    cue_mode now — Z_lr is a single axis, not trait-vs-dial.
    """
    nc = _n_colors(cache)
    w, h = FigSize.small
    fig, axes = plt.subplots(2, 1, figsize=(w * 1.4, h * 2 * 1.1), dpi=params.dpi,
                             sharex=True, layout='constrained')
    for row, field_name, ylab in ((0, 't1', 'Trial 1\n(prior: 1 = old rotation)'),
                                  (1, 't2', 'Trial 2\n(zero-shot reuse)')):
        ax = axes[row]
        for cue in CUE_MODES:
            xs, ys = [], []
            for z in Z_LR:
                c = stage_curve(cache, z, ('S3', cue), params)
                if c is None:
                    continue
                v = curve_summary(c, nc)[field_name]
                if not np.isnan(v):
                    xs.append(_ZLR_X[z]); ys.append(v)
            if xs:
                ax.plot(xs, ys, marker='o', markersize=3.0,
                        color=CUE_INFO[cue].color, linewidth=params.linewidth,
                        alpha=0.8, markerfacecolor='none', markeredgewidth=0.9,
                        label=CUE_INFO[cue].label)
        ax.axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.35)
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel(ylab)
        if row == 1:
            ax.set_xticks(list(_ZLR_X.values()))
            ax.set_xticklabels([str(z) for z in Z_LR], fontsize=5)
            ax.set_xlabel(r'S3  $\alpha_z$', labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncols=len(handles),
               fontsize=5, frameon=False, handlelength=1.1, columnspacing=0.8)
    _save(fig, export_dir, 'F4_reuse_summary.pdf', params)
    return fig


# ── D: does the latent actually flip? ─────────────────────────────────────────

def z_separation(runs) -> Tuple[float, float]:
    """How differently the latent gate is set in the two rotations, in [0, 1].

    Mean softmax(Z)[:, 0] under rotation A minus the same under rotation B. Measured
    post-activation because the gate is what Z actually does to the hidden state. 0 means the
    latent carries no context at all.
    """
    vals = []
    for logger, config in runs:
        z = np.concatenate([np.asarray(e).reshape(-1, int(np.prod(config.latent_dims)))
                            for e in logger.latent_values])
        ll = np.concatenate([np.asarray(e).reshape(-1) for e in logger.context_ids])
        n = min(len(z), len(ll))
        z, ll = z[:n], ll[:n]
        e = np.exp(z - z.max(axis=1, keepdims=True))
        gate = (e / e.sum(axis=1, keepdims=True))[:, 0]
        rots = np.deg2rad(np.asarray(config.train_rotations, dtype=float))
        slot = np.argmin(np.abs(np.arctan2(np.sin(ll[:, None] - rots[None, :]),
                                           np.cos(ll[:, None] - rots[None, :]))), axis=1)
        if len(np.unique(slot)) < 2:
            continue
        vals.append(abs(float(np.mean(gate[slot == 0]) - np.mean(gate[slot == 1]))))
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), (float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                                  if len(vals) > 1 else 0.0)


def plot_z_separation(cache, params: AnalysisParams, export_dir: Path,
                      cue_mode: str = 'oracle_z') -> plt.Figure:
    """D — gate separation between the two rotations in S3, against Z_lr."""
    fig, ax = plt.subplots(figsize=FigSize.wide, dpi=params.dpi, layout='constrained')
    xs, ys, es = [], [], []
    for z in Z_LR:
        runs = stage_runs(cache, z, ('S3', cue_mode), params.noise)
        if not runs:
            continue
        m, e = z_separation(runs)
        if not np.isnan(m):
            xs.append(_ZLR_X[z]); ys.append(m); es.append(e)
    if xs:
        ax.errorbar(xs, ys, yerr=es, marker='o', markersize=3.0,
                    capsize=1.5, elinewidth=0.8, color=plot_style.get_model_color('NeuraGEM'),
                    linewidth=params.linewidth, alpha=0.8, markerfacecolor='none',
                    markeredgewidth=0.9)
    ax.set_xticks(list(_ZLR_X.values()))
    ax.set_xticklabels([str(z) for z in Z_LR])
    ax.set_xlabel(r'S3  $\alpha_z$', labelpad=1)
    ax.set_ylabel('Gate separation\nbetween rotations')
    ax.set_ylim(0, None)
    _save(fig, export_dir, 'D_z_separation.pdf', params)
    return fig


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(cache, params: AnalysisParams) -> None:
    """The scalars from every curve, one row per (Z_lr, stage)."""
    nc = _n_colors(cache)
    print(f'\n-- Switch-aligned normalized state error @ noise={params.noise} --')
    print('   pre = mean of x<=0 | t1 = first trial of new block | t2 = zero-shot reuse trial')
    print(f"{'Z_lr':<7}{'stage':<34}{'pre':>7}{'t1':>7}{'t2':>7}{'reuse':>8}"
          f"{'mb1':>7}{'mb2':>7}{'asym':>7}{'sw':>6}{'sem':>10}")
    for z in Z_LR:
        stages: List[Stage] = ['S1']
        for cue in CUE_MODES:
            stages.append(('S2', cue))
            stages.append(('S3', cue))
            stages.append(('S3pinned', cue))
        for stage in stages:
            c = stage_curve(cache, z, stage, params)
            if c is None:
                continue
            s = curve_summary(c, nc)
            print(f'{str(z):<7}{stage_label(stage):<34}'
                  f"{s['pre']:>7.3f}{s['t1']:>7.3f}{s['t2']:>7.3f}{s['reuse']:>8.3f}"
                  f"{s['mb1']:>7.3f}{s['mb2']:>7.3f}{s['asym']:>7.3f}"
                  f'{c.n_switches:>6}{c.sem_over:>10}')


def check_acceptance(cache, params: AnalysisParams) -> bool:
    """Sanity gates, re-expressed on the adaptation curve.

    Each is a way the headline comparison could be uninterpretable rather than merely
    disappointing, so they are worth checking before reading anything else.
    """
    nc = _n_colors(cache)
    print('\n-- Sanity checks --')
    ok = True

    def _s(z, stage):
        c = stage_curve(cache, z, stage, params)
        return curve_summary(c, nc) if c is not None else None

    # (a) The models learned the task at all: pre-switch error near 0 in S1.
    pres = [_s(z, 'S1')['pre'] for z in Z_LR if _s(z, 'S1')]
    worst = max(pres) if pres else np.nan
    p = not np.isnan(worst) and worst <= 0.25
    ok &= p
    print(f"  (a) S1 asymptote reached          worst pre={worst:.3f} <= 0.25   "
          f"{'PASS' if p else 'FAIL'}")

    # (b) The cue is used: with the rotation handed to Z there is nothing to infer, so the
    #     trial-1 switch cost should collapse relative to the uncued control.
    cued = [_s(z, ('S2', 'oracle_z'))['t1'] for z in Z_LR if _s(z, ('S2', 'oracle_z'))]
    ctrl = [_s(z, ('S2', 'none'))['t1'] for z in Z_LR if _s(z, ('S2', 'none'))]
    wc = max(cued) if cued else np.nan
    mc = float(np.mean(ctrl)) if ctrl else np.nan
    p = not np.isnan(wc) and wc <= 0.25
    ok &= p
    print(f"  (b) S2 cued has no switch cost    worst t1={wc:.3f} <= 0.25 "
          f"(uncued control {mc:.3f})   {'PASS' if p else 'FAIL'}")
    if not p:
        print('      -> the weights are inferring context rather than reading the gate; '
              'drop S2_BLOCK_SCALE and rerun.')

    # (c) Z is the only context channel in S3: with it pinned (S3_PINNED_Z_SANITY), nothing
    #     should recover.
    froz = [_s(z, ('S3pinned', 'oracle_z'))['asym'] for z in Z_LR
            if _s(z, ('S3pinned', 'oracle_z'))]
    lo = min(froz) if froz else np.nan
    p = not np.isnan(lo) and lo >= 0.4
    ok &= p
    print(f"  (c) S3 pinned-Z does not recover  min asym={lo:.3f} >= 0.4   "
          f"{'PASS' if p else 'FAIL'}")
    if not p:
        print('      -> something other than Z carries context across blocks; '
              'nothing downstream is interpretable.')

    # (d) Z_lr changes recovery at all, else there is no dose-response to read. 'RNN' is
    #     excluded — it is a qualitative no-latent baseline, not a point on the numeric dial.
    v = [_s(z, ('S3', 'oracle_z'))['mb1'] for z in Z_LR if z != 'RNN'
         and _s(z, ('S3', 'oracle_z'))]
    v = [q for q in v if not np.isnan(q)]
    best = (max(v) - min(v)) if len(v) > 1 else np.nan
    p = not np.isnan(best) and best >= 0.1
    ok &= p
    print(f"  (d) S3 separates by Z_lr          mb1 spread={best:.3f} >= 0.1   "
          f"{'PASS' if p else 'FAIL'}")

    print(f"\n  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return bool(ok)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

class _FakeLogger:
    """Minimal stand-in carrying only what flatten_logger and _phase_range read."""

    def __init__(self, ii, oi, ll):
        self.inputs            = [ii[:, None, :]]
        self.predicted_outputs = [oi[:, None, :]]
        self.context_ids       = [ll[:, None]]
        self.latent_values     = []
        self.phases            = [('Learning and inference', 0)]
        self.others            = {}


def _self_test() -> None:
    """Check the switch alignment and the metric against synthetic predictions.

    Two reference "models", neither of which needs training:
      perfect  — predicts the correct target for the rotation in force. Must score 0 everywhere.
      lag-one  — uses the rotation of the *previous trial*. Inside a block that is the same
                 rotation, so it is correct; the one place it is wrong is the first trial after a
                 switch. It must therefore spike to exactly 1 at x=1 and sit at 0 at every other
                 position, which is the alignment claim in full: the switch falls between x=0 and
                 x=1, and x=2 is already inside the new block.
    """
    from datasets import RotatingTargetsDataset
    from rotation_curriculum_config import make_base_config

    cfg = make_base_config(noise_std=0.0)
    cfg.block_size   = 14 * cfg.n_colors * 2
    cfg.no_of_blocks = 8
    ds  = RotatingTargetsDataset(cfg)
    ii  = np.array(ds.data_sequence, dtype=float)
    ll  = np.array(ds.llcid_sequence, dtype=float)
    nc  = cfg.n_colors
    params = AnalysisParams(n_pre=3, n_post=14, n_seeds=1)

    cue = np.flatnonzero(ii[:, :nc].sum(axis=1) > 0.5)
    colour = np.argmax(ii[cue, :nc], axis=1)
    rots = {float(d): get_target_positions(cfg, float(d)) for d in cfg.train_rotations}

    def _deg(rad):
        return _nearest_rotation_deg(rad, cfg.train_rotations)

    # Perfect: predict the current rotation's target for the cued colour.
    oi_perfect = np.zeros((len(ll), cfg.output_size))
    oi_perfect[cue + 1, -2:] = [rots[_deg(ll[t])][c] for t, c in zip(cue, colour)]
    curve = adaptation_curve([(_FakeLogger(ii, oi_perfect, ll), cfg)], params)
    assert np.nanmax(curve.mean) < 1e-6, f'perfect model should score 0, got {np.nanmax(curve.mean)}'
    print(f'  perfect prediction -> 0 everywhere   : max {np.nanmax(curve.mean):.2e}   OK')

    # Lag-one: use the rotation of the previous trial. Wrong only on the first trial of a block.
    rot_lag = np.concatenate([[ll[cue[0]]], ll[cue[:-1]]])
    oi_lag = np.zeros((len(ll), cfg.output_size))
    oi_lag[cue + 1, -2:] = [rots[_deg(r)][c] for r, c in zip(rot_lag, colour)]
    curve = adaptation_curve([(_FakeLogger(ii, oi_lag, ll), cfg)], params)
    x, m = curve.x, curve.mean
    assert abs(m[x == 1][0] - 1) < 1e-6, f'lag-one should be fully wrong at x=1, got {m[x == 1]}'
    assert np.all(m[x != 1] < 1e-6), (
        f'lag-one should be correct everywhere but x=1; got {dict(zip(x[m > 1e-6], m[m > 1e-6]))}')
    print('  lag-one prediction -> spike at x=1 only: 1.000 at x=1, 0 elsewhere   OK')

    # anchor='new' scores every trial against the post-switch rotation, so the lag-one model is
    # wrong for x<=1 (still on the old rotation) and right from x=2 on.
    p_new = AnalysisParams(n_pre=3, n_post=14, n_seeds=1, anchor='new')
    c_new = adaptation_curve([(_FakeLogger(ii, oi_lag, ll), cfg)], p_new)
    xn, mn = c_new.x, c_new.mean
    assert np.all(mn[xn <= 1] > 1 - 1e-6) and np.all(mn[xn >= 2] < 1e-6), mn
    print("  anchor='new' -> 1 for x<=1, 0 after    : OK")

    # Every trial position must be filled: blocks here are 14 mini-blocks long, so a 14-trial
    # window can never run off the end of a block.
    assert not np.isnan(curve.mean).any(), f'unfilled trial positions: {curve.x[np.isnan(curve.mean)]}'
    print(f'  all {len(curve.x)} trial positions filled       : {curve.n_switches} switches   OK')

    s = curve_summary(curve, nc)
    assert abs(s['pre']) < 1e-6, s
    assert abs(s['t1'] - 1) < 1e-6 and abs(s['t2']) < 1e-6, s
    assert abs(s['reuse'] - 1) < 1e-6, s          # one observation buys the whole map back
    assert abs(s['mb1'] - 1 / nc) < 1e-6, s       # exactly one wrong trial in the first mini-block
    print('  curve_summary on the lag-one model    : OK')
    print('\nSELF-TEST PASSED')


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> dict:
    global tree_cache
    params     = AnalysisParams()
    export_dir = EXPORT_ROOT / 'figures'

    if not globals().get('tree_cache'):
        tree_cache = load_trees(params)
        clear_memo()
    if not tree_cache:
        print(f'No trees under {EXPORT_ROOT}. Run rotation_curriculum_sweep.py first.')
        return {}

    plot_uncued_control_sanity(tree_cache, params, export_dir)
    plot_s1_dose_response(tree_cache, params, export_dir)
    plot_s2_dose_response(tree_cache, params, export_dir)
    plot_s3_dose_response(tree_cache, params, export_dir)
    plot_s1_vs_s3_by_zlr(tree_cache, params, export_dir)
    plot_reuse_summary(tree_cache, params, export_dir)
    plot_z_separation(tree_cache, params, export_dir)
    summarize(tree_cache, params)
    check_acceptance(tree_cache, params)
    return tree_cache


if __name__ == '__main__':
    print('Self-test (synthetic, no trained model needed):')
    _self_test()
    tree_cache = main()
