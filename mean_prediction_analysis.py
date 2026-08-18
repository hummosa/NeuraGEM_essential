"""Analysis of mean-prediction task sweep results.

Loads pickles produced by mean_prediction_sweep.py, computes per-block
side-correct rate (fraction of timesteps where the predicted mean is on the
correct side of the context midpoint), averages over seeds, and plots all
conditions on shared axes.

Usage:
    python mean_prediction_analysis.py
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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import plot_style
from plot_style import FigSize
from mean_prediction_sweep_config import ConditionInfo, CONDITION_INFO, EXPORT_ROOT

plot_style.set_plot_style()


# ---------------------------------------------------------------------------
# Global analysis parameters
# ---------------------------------------------------------------------------

@dataclass
class AnalysisParams:
    n_seeds:                   int   = 10
    last_ts_in_a_block:        int   = 15      # tail timesteps used per block
    phases_to_include:         str   = 'Learning and inference'
    aggregate_blocks:          int | None = 3  # None = one point per block
    dpi:                       int   = 160
    show_plots:                bool  = True
    save_plots:                bool  = True
    skip_first_blocks:         int   = 0
    block_group_max:           int | None = 50
    asymptotic_n_last_groups:  int   = 1
    # "learning speed": block-group where side-correct first exceeds this fraction
    # of the (asymptote − initial) gain (measured from initial ~0.5 chance level).
    adaptation_threshold:      float = 0.5
    criterion_n:               int   = 3   # consecutive correct steps to declare new context acquired
    linewidth:                 float = 1.75
    alpha:                     float = .80


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_loggers(condition_name: str, n_seeds: int) -> List[Any]:
    """Load train_loggers for all seeds of one condition."""
    folder = EXPORT_ROOT / condition_name / "default"
    loggers, missing = [], []
    for seed in range(n_seeds):
        filepath = folder / f"results_seed-{seed}.pkl"
        if filepath.exists():
            with filepath.open("rb") as f:
                payload = pickle.load(f)
            loggers.append(payload["train_logger"])
        else:
            missing.append(seed)
    if missing:
        print(f"  [{condition_name}] missing seeds: {missing}")
    print(f"  [{condition_name}] loaded {len(loggers)}/{n_seeds} seeds")
    return loggers


def load_all_loggers(params: AnalysisParams) -> Dict[str, List]:
    print("Loading loggers...")
    return {name: load_loggers(name, params.n_seeds) for name in CONDITION_INFO}


# ---------------------------------------------------------------------------
# Metric: per-block side-correct rate
# ---------------------------------------------------------------------------

def extract_block_side_correct(
    logger,
    last_ts_in_a_block: int,
    phases_to_include: str,
    training_data_means: Sequence[float] = (0.2, 0.8),
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Fraction of tail timesteps in each block where pred_mean is on the correct side.

    Returns (block_means, block_sems) arrays of shape (n_blocks,), or (None, None).
    pred_mean = logger.predicted_outputs[:, 1]  (the mean-prediction output dim)
    true_mean = logger.context_ids[:, 0]             (ground-truth context mean)
    """
    if not logger.predicted_outputs:
        return None, None

    oi_full  = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1, 2)
    ll_full  = np.concatenate(logger.context_ids,            axis=0).reshape(-1)

    # Filter to requested phase(s)
    if phases_to_include is not None:
        include = [phases_to_include] if isinstance(phases_to_include, str) else phases_to_include
        phase_windows = []
        for i, (name, ts) in enumerate(logger.phases):
            end = logger.phases[i + 1][1] if i + 1 < len(logger.phases) else len(ll_full)
            if name in include:
                phase_windows.append((ts, end))
        if phase_windows:
            oi = np.concatenate([oi_full[s:e] for s, e in phase_windows], axis=0)
            ll = np.concatenate([ll_full[s:e] for s, e in phase_windows], axis=0)
        else:
            oi, ll = oi_full, ll_full
    else:
        oi, ll = oi_full, ll_full

    midpoint   = float(np.mean(training_data_means))
    pred_mean  = oi[:, 1]
    correct    = ((pred_mean - midpoint) * (ll - midpoint) > 0).astype(float)

    T           = len(ll)
    block_starts = [0] + [t for t in range(1, T) if ll[t] != ll[t - 1]]
    block_ends   = block_starts[1:] + [T]

    block_means = []
    for start, end in zip(block_starts, block_ends):
        tail_start = max(start, end - last_ts_in_a_block)
        tail = correct[tail_start:end]
        if len(tail) > 0:
            block_means.append(float(tail.mean()))

    if not block_means:
        return None, None

    arr = np.array(block_means)
    return arr, np.zeros_like(arr)   # per-block, no SEM at this level; seed-level SEM computed upstream


def compute_mean_sem_curve(
    loggers: Sequence[Any],
    params: AnalysisParams,
    training_data_means: Sequence[float] = (0.2, 0.8),
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Average side-correct curves across seeds. Returns (x, mean, sem)."""
    all_curves = []
    for logger in loggers:
        bm, _ = extract_block_side_correct(
            logger,
            last_ts_in_a_block=params.last_ts_in_a_block,
            phases_to_include=params.phases_to_include,
            training_data_means=training_data_means,
        )
        if bm is not None and len(bm) > 0:
            all_curves.append(bm)

    if not all_curves:
        return None, None, None

    min_len = min(len(b) for b in all_curves)
    arr     = np.stack([b[:min_len] for b in all_curves])   # (n_seeds, n_blocks)

    if params.aggregate_blocks is not None:
        n_agg    = int(params.aggregate_blocks)
        n_groups = min_len // n_agg
        agg = np.stack(
            [arr[:, g * n_agg:(g + 1) * n_agg].mean(axis=1) for g in range(n_groups)],
            axis=1,
        )
        x          = np.arange(n_groups)
        mean_curve = agg.mean(axis=0)
        sem_curve  = agg.std(axis=0, ddof=1) / np.sqrt(len(loggers))
    else:
        x          = np.arange(min_len)
        mean_curve = arr.mean(axis=0)
        sem_curve  = arr.std(axis=0, ddof=1) / np.sqrt(len(loggers))

    return x, mean_curve, sem_curve


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _prune_legend_handles(handles, labels, max_entries=4):
    """If legend has >max_entries items, keep only first, middle, and last.
    Viewers extrapolate the gradient; the intermediate entries add clutter."""
    n = len(handles)
    if n <= max_entries:
        return handles, labels
    indices = [0,1,  n // 2, n - 1]
    return [handles[i] for i in indices], [labels[i] for i in indices]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_canonical(
    params: AnalysisParams,
    export_dir: Path,
    loggers_cache: Dict[str, List],
) -> plt.Figure:
    """Side-correct learning curves + asymptotic summary panel."""
    pw, ph = FigSize.wide
    sw = FigSize.small[0]
    fig = plt.figure(
        figsize=(pw + sw, ph),
        dpi=params.dpi,
        layout='constrained',
    )
    gs = gridspec.GridSpec(1, 2, width_ratios=[pw, sw], figure=fig)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_asy  = fig.add_subplot(gs[0, 1])

    asymptotic_pts: List[Tuple[str, float, float]] = []

    for cond_name in CONDITION_INFO:
        info    = CONDITION_INFO[cond_name]
        loggers = loggers_cache.get(cond_name, [])
        if not loggers:
            continue

        x, mean_curve, sem_curve = compute_mean_sem_curve(loggers, params)
        if x is None:
            continue

        skip = params.skip_first_blocks
        if skip > 0 and len(x) > skip:
            x, mean_curve, sem_curve = x[skip:], mean_curve[skip:], sem_curve[skip:]
        if params.block_group_max is not None:
            x          = x[:params.block_group_max]
            mean_curve = mean_curve[:params.block_group_max]
            sem_curve  = sem_curve[:params.block_group_max]

        ax_main.plot(x, mean_curve, color=info.color, linewidth=params.linewidth, alpha=params.alpha, label=info.label)
        ax_main.fill_between(x, mean_curve - sem_curve, mean_curve + sem_curve,
                             color=info.color, alpha=0.2, linewidth=0)

        n_last    = min(params.asymptotic_n_last_groups, len(mean_curve))
        asym_mean = float(mean_curve[-n_last:].mean())
        asym_sem  = float(sem_curve[-n_last:].mean())
        asymptotic_pts.append((cond_name, asym_mean, asym_sem))

        # Learning speed: first block-group where side-correct exceeds threshold
        # fraction of the (asymptote − 0.5) gain above chance.
        gain = asym_mean - 0.5
        if gain > 0.02:
            thresh = 0.5 + gain * params.adaptation_threshold
            crossings = np.where(mean_curve >= thresh)[0]
            if len(crossings) > 0:
                ax_main.axvline(float(x[crossings[0]]), color=info.color,
                                linewidth=0.8, linestyle=':', alpha=0.55, zorder=0)

    x_label = (
        f'Block group (×{params.aggregate_blocks})'
        if params.aggregate_blocks is not None
        else 'Block'
    )
    ax_main.axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.3)
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(f'Side-correct rate  (tail {params.last_ts_in_a_block} ts)')
    ax_main.set_ylim(0.35, 1.05)
    handles, labels = ax_main.get_legend_handles_labels()
    handles, labels = _prune_legend_handles(handles, labels)
    # ax_main.legend(handles, labels, loc='lower right')

    # Asymptotic summary panel
    if asymptotic_pts:
        a_conds, a_means, a_sems = zip(*asymptotic_pts)
        xs = np.arange(len(a_conds))
        ax_asy.plot(xs, a_means, color='0.65', linewidth=1.0, zorder=1)
        for i, (cond_name, mean_val, sem_val) in enumerate(asymptotic_pts):
            info = CONDITION_INFO.get(cond_name, ConditionInfo(label=cond_name, color='grey'))
            ax_asy.errorbar(xs[i], mean_val, yerr=sem_val, fmt='o',
                            color=info.color, capsize=3, linewidth=1.5, markersize=6, zorder=2)
        ax_asy.axhline(0.5, color='k', linewidth=0.6, linestyle=':', alpha=0.3)
        ax_asy.set_xticks(xs[::2]) # show every other label to avoid crowding

        ax_asy.set_xticklabels(
            [CONDITION_INFO.get(c, ConditionInfo(label=c, color='grey')).label for c in a_conds][::2],
            rotation=45, ha='right',
        )
        n_last = params.asymptotic_n_last_groups
        group_word = 'group' if n_last == 1 else 'groups'
        ax_asy.set_ylabel(f'Asymptotic side-correct\n(last {n_last} {group_word})')

    if params.save_plots:
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / "side_correct_by_condition.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved → ./{out}")

    if params.show_plots:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Perseveration & context-slip metrics
# ---------------------------------------------------------------------------

def _phase_filter(oi_full: np.ndarray, ll_full: np.ndarray, logger, phases_to_include):
    """Return (oi, ll) restricted to the requested training phase(s)."""
    if phases_to_include is None:
        return oi_full, ll_full
    include = [phases_to_include] if isinstance(phases_to_include, str) else list(phases_to_include)
    windows = []
    for i, (name, ts) in enumerate(logger.phases):
        end = logger.phases[i + 1][1] if i + 1 < len(logger.phases) else len(ll_full)
        if name in include:
            windows.append((ts, end))
    if not windows:
        return oi_full, ll_full
    oi = np.concatenate([oi_full[s:e] for s, e in windows], axis=0)
    ll = np.concatenate([ll_full[s:e] for s, e in windows], axis=0)
    return oi, ll


def _find_criterion(block_correct: np.ndarray, criterion_n: int) -> int | None:
    """Return index of the first step AFTER the first run of criterion_n consecutive
    correct predictions, or None if the criterion is never reached in the block."""
    for t in range(len(block_correct) - criterion_n + 1):
        if block_correct[t:t + criterion_n].all():
            return t + criterion_n
    return None


def extract_block_criterion_metrics(
    logger,
    phases_to_include: str,
    criterion_n: int = 3,
    training_data_means: Sequence[float] = (0.2, 0.8),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-block perseveration errors, context slips, and trials-to-criterion.

    For each block after the first (every block follows a context switch):
      Perseveration errors — wrong-side predictions before criterion is reached.
        If criterion is never reached, all errors in the block count as perseveration.
      Context slips — wrong-side predictions after criterion is reached.
        NaN when criterion was never reached (model never adapted).
      Trials to criterion — timesteps elapsed until the criterion run ends.
        NaN when criterion was never reached.

    Returns three arrays of shape (n_blocks_in_phase − 1,).
    """
    if not logger.predicted_outputs:
        return np.array([]), np.array([]), np.array([])

    oi_full = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1, 2)
    ll_full = np.concatenate(logger.context_ids,            axis=0).reshape(-1)
    oi, ll  = _phase_filter(oi_full, ll_full, logger, phases_to_include)

    midpoint  = float(np.mean(training_data_means))
    pred_mean = oi[:, 1]
    correct   = (pred_mean - midpoint) * (ll - midpoint) > 0

    T            = len(ll)
    block_starts = [0] + [t for t in range(1, T) if ll[t] != ll[t - 1]]
    block_ends   = block_starts[1:] + [T]

    persev_list, slips_list, ttc_list = [], [], []

    for i, (start, end) in enumerate(zip(block_starts, block_ends)):
        if i == 0:
            continue  # first block: no prior context to perseverate from
        block_correct = correct[start:end]
        t_crit = _find_criterion(block_correct, criterion_n)

        if t_crit is not None:
            persev_list.append(float(np.sum(~block_correct[:t_crit])))
            slips_list.append(float(np.sum(~block_correct[t_crit:])))
            ttc_list.append(float(t_crit))
        else:
            # never adapted: all errors are perseverative; no post-criterion period
            persev_list.append(float(np.sum(~block_correct)))
            slips_list.append(np.nan)
            ttc_list.append(np.nan)

    return np.array(persev_list), np.array(slips_list), np.array(ttc_list)


def _aggregate_per_block_curves(
    per_block_arrs: List[np.ndarray],
    aggregate_blocks: int | None,
    skip_first_blocks: int = 0,
    block_group_max: int | None = None,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Average per-block arrays (one array per seed) into a mean±SEM curve.

    NaN values (e.g., TTC or slips when criterion not reached) are handled via
    nanmean/nanstd so that seeds which did adapt contribute to the estimate.
    """
    if not per_block_arrs:
        return None, None, None
    min_len = min(len(b) for b in per_block_arrs)
    if min_len == 0:
        return None, None, None

    arr = np.stack([b[:min_len] for b in per_block_arrs])   # (n_seeds, n_blocks)
    if skip_first_blocks > 0:
        arr = arr[:, skip_first_blocks:]

    if aggregate_blocks is not None:
        n_agg    = int(aggregate_blocks)
        n_groups = arr.shape[1] // n_agg
        if n_groups == 0:
            return None, None, None
        agg = np.stack(
            [arr[:, g * n_agg:(g + 1) * n_agg].mean(axis=1) for g in range(n_groups)],
            axis=1,
        )
        x = np.arange(n_groups)
    else:
        agg = arr
        x   = np.arange(arr.shape[1])

    if block_group_max is not None:
        x   = x[:block_group_max]
        agg = agg[:, :block_group_max]

    n_valid    = np.sum(~np.isnan(agg), axis=0).clip(min=1)
    mean_curve = np.nanmean(agg, axis=0)
    sem_curve  = np.nanstd(agg, axis=0, ddof=1) / np.sqrt(n_valid)
    return x, mean_curve, sem_curve


def _asymptote_panel(ax, asym_pts, ylabel, n_last, condition_info=None, note=None):
    """Shared helper: draw the asymptotic summary dots-and-line panel.

    condition_info defaults to this module's CONDITION_INFO; pass another mapping of
    {condition_name: ConditionInfo} to reuse the panel from a different sweep
    (rotation_slips_perseveration_analysis does).

    note overrides the "(last N groups)" line under the y-label, for callers that summarise
    over a different window (e.g. the mean of the whole curve rather than its tail).
    """
    if not asym_pts:
        return
    info_map = CONDITION_INFO if condition_info is None else condition_info
    a_conds, a_means, a_sems = zip(*asym_pts)
    xs = np.arange(len(a_conds))
    ax.plot(xs, a_means, color='0.65', linewidth=1.0, zorder=1)
    for i, (cond_name, mean_val, sem_val) in enumerate(asym_pts):
        info = info_map.get(cond_name, ConditionInfo(label=cond_name, color='grey'))
        ax.errorbar(xs[i], mean_val, yerr=sem_val, fmt='o',
                    color=info.color, capsize=3, linewidth=1.5, markersize=6, zorder=2)
    ax.set_xticks(xs[::2]) # show every other label to avoid crowding
    ax.set_xticklabels(
        [info_map.get(c, ConditionInfo(label=c, color='grey')).label for c in a_conds][::2],
        rotation=45, ha='right',
    )
    if note is None:
        group_word = 'group' if n_last == 1 else 'groups'
        note = f'last {n_last} {group_word}'
    ax.set_ylabel(f'{ylabel}\n({note})')


def plot_perseveration_and_slips(
    params: AnalysisParams,
    export_dir: Path,
    loggers_cache: Dict[str, List],
) -> plt.Figure:
    """Temporal evolution of perseveration errors and context slips over training.

    Layout (2 × 2):
      [A] Perseveration errors per block  |  [C] Asymptotic perseveration
      [B] Context slips per block         |  [D] Asymptotic slips

    Perseveration errors: wrong-side predictions before the model reaches criterion
        (params.criterion_n consecutive correct) after each context switch.
    Context slips: wrong-side predictions after criterion is reached in the same block.

    Blocks where criterion is never reached contribute NaN to slips and TTC;
    all their errors go to perseveration. nanmean is used across seeds.
    """
    pw, ph = FigSize.wide
    sw = FigSize.small[0]
    fig = plt.figure(
        figsize=(pw + sw, ph * 2),
        dpi=params.dpi,
        layout='constrained',
    )
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[pw, sw],
        figure=fig,
    )
    ax_persev = fig.add_subplot(gs[0, 0])
    ax_slips  = fig.add_subplot(gs[1, 0])
    ax_asy_p  = fig.add_subplot(gs[0, 1])
    ax_asy_s  = fig.add_subplot(gs[1, 1])

    asym_persev: List[Tuple[str, float, float]] = []
    asym_slips:  List[Tuple[str, float, float]] = []

    x_label = (
        f'Block group (×{params.aggregate_blocks})'
        if params.aggregate_blocks is not None
        else 'Block'
    )

    for cond_name in CONDITION_INFO:
        info    = CONDITION_INFO[cond_name]
        loggers = loggers_cache.get(cond_name, [])
        if not loggers:
            continue

        persev_arrs, slips_arrs = [], []
        for logger in loggers:
            p, s, _ = extract_block_criterion_metrics(
                logger,
                phases_to_include=params.phases_to_include,
                criterion_n=params.criterion_n,
            )
            if len(p) > 0:
                persev_arrs.append(p)
                slips_arrs.append(s)

        for ax, arrs, asym_list in (
            (ax_persev, persev_arrs, asym_persev),
            (ax_slips,  slips_arrs,  asym_slips),
        ):
            x, mean_c, sem_c = _aggregate_per_block_curves(
                arrs,
                params.aggregate_blocks,
                params.skip_first_blocks,
                params.block_group_max,
            )
            if x is None:
                continue

            ax.plot(x, mean_c, color=info.color, linewidth=params.linewidth, alpha=params.alpha, label=info.label)
            ax.fill_between(x, mean_c - sem_c, mean_c + sem_c,
                            color=info.color, alpha=0.2, linewidth=0)

            n_last    = min(params.asymptotic_n_last_groups, len(mean_c))
            asym_mean = float(np.nanmean(mean_c[-n_last:]))
            asym_sem  = float(np.nanmean(sem_c[-n_last:]))
            asym_list.append((cond_name, asym_mean, asym_sem))

    ax_persev.set_ylabel(f'Perseveration errors / block')
    ax_persev.set_xlabel(x_label)
    handles, labels = ax_persev.get_legend_handles_labels()
    handles, labels = _prune_legend_handles(handles, labels)
    fig.legend(handles, labels, loc='outside upper left', ncols=len(handles)//2
    , fontsize='small', frameon=False, bbox_to_anchor=(0.05, 1.099))

    ax_slips.set_ylabel('Context slips / block')
    ax_slips.set_xlabel(x_label)

    _asymptote_panel(ax_asy_p, asym_persev, 'Perseveration errors',
                     params.asymptotic_n_last_groups)
    _asymptote_panel(ax_asy_s, asym_slips,  'Context slips',
                     params.asymptotic_n_last_groups)

    if params.save_plots:
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / "perseveration_and_slips.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved → {out}")

    if params.show_plots:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> dict:
    global loggers_cache
    analysis_params = AnalysisParams()
    export_dir      = EXPORT_ROOT / "figures"

    if globals().get('loggers_cache') is None:
        loggers_cache = load_all_loggers(analysis_params)

    plot_canonical(analysis_params, export_dir, loggers_cache)
    plot_perseveration_and_slips(analysis_params, export_dir, loggers_cache)
    return loggers_cache


if __name__ == "__main__":
    loggers_cache = main()
