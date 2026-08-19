"""Analysis of canonical-conditions sweep results.

Loads pickles produced by cst_correlated_noise_sweep.py (50 jobs: 5 conditions × 10
seeds), extracts per-block |pred - mean| curves via extract_block_corrects(), averages
over seeds, and plots all 5 conditions on shared axes.

Usage:
    python cst_correlated_noise_analysis.py
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
from sklearn.linear_model import LinearRegression

import plot_style
from functions_and_utils import extract_block_corrects
from cst_correlated_noise_config import ConditionInfo, CONDITION_INFO, RUN_NAME, EXPORT_ROOT

plot_style.set_plot_style()


# ---------------------------------------------------------------------------
# Global analysis parameters
# ---------------------------------------------------------------------------

@dataclass
class AnalysisParams:
    n_seeds: int = 10
    last_ts_in_a_block: int = 15
    phases_to_include: str = 'Learning and inference'
    aggregate_blocks: int | None = 5    # None = one point per block
    subplot_width: float = 5.0
    subplot_height: float = 2.5
    dpi: int = 100
    show_plots: bool = True
    save_plots: bool = True
    skip_first_blocks: int = 0      
        # drop warm-up blocks from display
    block_group_max: int | None = 50    # cap x-axis of block-level plots (None = all)
    asymptotic_n_last_groups: int = 1   # how many trailing block-groups define "asymptote"
    adaptation_threshold: float = 0.5  # learning speed: fraction of (initial−asymptote) late-error reduction
    adaptation_switch_threshold: float = 0.85  # adaptation speed: fraction of (peak−floor) reduction post-switch
    switch_pre_window:  int = 4         # timesteps before switch to include
    switch_post_window: int = 8        # timesteps after switch to include


# ---------------------------------------------------------------------------
# Comparison specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalSpec:
    key: str
    title: str
    conditions: Sequence[str] = field(default_factory=lambda: list(CONDITION_INFO.keys()))


CANONICAL_SPECS: List[CanonicalSpec] = [
    CanonicalSpec(
        key="all_conditions",
        title="RNN vs NeuraGEM variants (no correlated noise)",
        conditions=list(CONDITION_INFO.keys()),
    ),
]


# ---------------------------------------------------------------------------
# Loading — path mirrors cst_correlated_noise_sweep.py exactly
# ---------------------------------------------------------------------------

def load_loggers(condition_name: str, n_seeds: int) -> List[Any]:
    """Load train_loggers for all seeds of one condition.

    Sweep saves to EXPORT_ROOT/<condition_name>/default/results_seed-<N>.pkl
    because combo_key({}) = "default" when there are no non-seed params.
    """
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


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_mean_sem_curve(
    loggers: Sequence[Any],
    last_ts_in_a_block: int,
    phases_to_include: str,
    aggregate_blocks: int | None,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Average extract_block_corrects across seeds.

    Returns (x, mean_curve, sem_curve) or (None, None, None) if no data.
    """
    all_block_means = []
    for logger in loggers:
        bm, _ = extract_block_corrects(
            logger,
            last_ts_in_a_block=last_ts_in_a_block,
            phases_to_include=phases_to_include,
        )
        if bm is not None and len(bm) > 0:
            all_block_means.append(bm)

    if not all_block_means:
        return None, None, None

    min_len = min(len(b) for b in all_block_means)
    arr = np.stack([b[:min_len] for b in all_block_means])  # (n_seeds, n_blocks)

    if aggregate_blocks is not None:
        n_agg = int(aggregate_blocks)
        n_groups = min_len // n_agg
        agg = np.stack(
            [arr[:, g * n_agg:(g + 1) * n_agg].mean(axis=1) for g in range(n_groups)],
            axis=1,
        )
        x = np.arange(n_groups)
        mean_curve = agg.mean(axis=0)
        sem_curve = agg.std(axis=0, ddof=1) / np.sqrt(len(loggers))
    else:
        x = np.arange(min_len)
        mean_curve = arr.mean(axis=0)
        sem_curve = arr.std(axis=0, ddof=1) / np.sqrt(len(loggers))

    return x, mean_curve, sem_curve


def _estimate_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(x, y)
    slope = float(reg.coef_[0])
    resid = y - reg.predict(x)
    sigma2 = np.sum(resid ** 2) / max(len(x) - 2, 1)
    xvar = float(np.var(x, ddof=1))
    slope_se = float(np.sqrt(sigma2 / (len(x) * xvar))) if xvar > 0 else 0.0
    return slope, slope_se


def compute_block_end_lr(
    loggers: Sequence[Any],
    params: AnalysisParams,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Estimate behavioral LR from the tail of each block, pooled across seeds."""
    per_seed_block_pairs: List[List[Tuple[np.ndarray, np.ndarray]]] = []

    for logger in loggers:
        ll  = np.concatenate(logger.context_ids,            axis=0).reshape(-1)
        oi  = np.concatenate(logger.predicted_outputs,  axis=0).reshape(-1)
        inp = np.concatenate(logger.inputs,             axis=0).reshape(-1)

        T = len(ll)
        block_starts = [0] + [t for t in range(1, T) if ll[t] != ll[t - 1]]
        block_ends   = block_starts[1:] + [T]

        block_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for start, end in zip(block_starts, block_ends):
            tail_start = max(start, end - params.last_ts_in_a_block)
            idx = np.arange(tail_start, end - 1)
            if len(idx) == 0:
                block_pairs.append((np.array([]), np.array([])))
                continue
            pred_err   = np.abs(inp[idx]     - oi[idx])
            delta_pred = np.abs(oi[idx + 1]  - oi[idx])
            block_pairs.append((pred_err, delta_pred))

        per_seed_block_pairs.append(block_pairs)

    if not per_seed_block_pairs:
        return None, None, None

    n_blocks_min = min(len(bp) for bp in per_seed_block_pairs)
    skip = params.skip_first_blocks

    if params.aggregate_blocks is None:
        x       = np.arange(n_blocks_min)
        mean_lr = np.full(n_blocks_min, np.nan)
        sem_lr  = np.full(n_blocks_min, np.nan)

        for b in range(n_blocks_min):
            errs   = [bp[b][0] for bp in per_seed_block_pairs if len(bp[b][0]) > 0]
            deltas = [bp[b][1] for bp in per_seed_block_pairs if len(bp[b][1]) > 0]
            if not errs:
                continue
            all_errs   = np.concatenate(errs)
            all_deltas = np.concatenate(deltas)
            if len(all_errs) < 2:
                continue
            slope, slope_se = _estimate_slope(all_errs, all_deltas)
            mean_lr[b] = slope
            sem_lr[b]  = slope_se

        if skip > 0:
            x, mean_lr, sem_lr = x[skip:], mean_lr[skip:], sem_lr[skip:]
        return x, mean_lr, sem_lr

    else:
        n_agg    = int(params.aggregate_blocks)
        n_groups = (n_blocks_min - skip) // n_agg
        if n_groups == 0:
            return None, None, None

        per_seed_group_lr: List[List[float]] = []
        for block_pairs in per_seed_block_pairs:
            seed_lr: List[float] = []
            for g in range(n_groups):
                b0 = skip + g * n_agg
                b1 = b0 + n_agg
                errs   = [block_pairs[b][0] for b in range(b0, b1)
                          if b < len(block_pairs) and len(block_pairs[b][0]) > 0]
                deltas = [block_pairs[b][1] for b in range(b0, b1)
                          if b < len(block_pairs) and len(block_pairs[b][1]) > 0]
                if not errs:
                    seed_lr.append(np.nan)
                    continue
                all_errs   = np.concatenate(errs)
                all_deltas = np.concatenate(deltas)
                if len(all_errs) < 2:
                    seed_lr.append(np.nan)
                    continue
                slope, _ = _estimate_slope(all_errs, all_deltas)
                seed_lr.append(slope)
            per_seed_group_lr.append(seed_lr)

        arr     = np.array(per_seed_group_lr)
        n_valid = np.sum(~np.isnan(arr), axis=0).clip(min=1)
        mean_lr = np.nanmean(arr, axis=0)
        sem_lr  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_valid)
        return np.arange(n_groups), mean_lr, sem_lr


# ---------------------------------------------------------------------------
# Switch-aligned analysis
# ---------------------------------------------------------------------------

def extract_switch_aligned_windows(
    loggers: Sequence[Any],
    pre_window: int,
    post_window: int,
    condition_name: str,
) -> np.ndarray | None:
    """Per-timestep |input - pred| windows aligned to block switches.

    Uses only switches in the last third of blocks. Discards any switch where
    the new block has fewer than post_window timesteps or there are fewer than
    pre_window timesteps available before the switch.

    Returns array of shape (n_windows, pre_window + post_window), or None.
    """
    all_windows: List[np.ndarray] = []

    for logger in loggers:
        ll  = np.concatenate(logger.context_ids,           axis=0).reshape(-1)
        oi  = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1)
        inp = np.concatenate(logger.inputs,            axis=0).reshape(-1)

        metric = np.abs(inp - oi)
        T      = len(ll)

        block_starts = [0] + [t for t in range(1, T) if ll[t] != ll[t - 1]]
        block_ends   = block_starts[1:] + [T]
        n_blocks     = len(block_starts)

        first_third_end = max(2, n_blocks // 3)

        for b in range(1, first_third_end):
            switch_t  = block_starts[b]
            block_end = block_ends[b]

            if switch_t - pre_window < 0:
                continue
            if block_end - switch_t < post_window:
                continue

            all_windows.append(metric[switch_t - pre_window : switch_t + post_window])

    n = len(all_windows)
    if n == 0:
        print(f"  WARNING [{condition_name}]: no valid switch windows found.")
        return None
    if n < 10:
        print(f"  WARNING [{condition_name}]: only {n} switch windows available (< 10), treat with caution.")

    return np.stack(all_windows)  # (n_windows, pre_window + post_window)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_canonical(
    spec: CanonicalSpec,
    params: AnalysisParams,
    export_dir: Path,
    loggers_cache: Dict[str, List],
) -> plt.Figure:
    """Training curves (|pred−mean|, behavioral LR) + asymptotic + adaptation-speed panels."""
    summary_w = params.subplot_height  # square-ish panels
    fig = plt.figure(
        figsize=(params.subplot_width + summary_w, params.subplot_height * 2),
        dpi=params.dpi,
        layout='constrained',
    )
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[params.subplot_width, summary_w],
        figure=fig,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])
    ax_asy = fig.add_subplot(gs[0, 1])
    ax_spd = fig.add_subplot(gs[1, 1])

    asymptotic_pts: List[Tuple[str, float, float]] = []  # (cond_name, mean, sem)
    adaptation_pts: List[Tuple[str, float | None]] = []  # (cond_name, x_at_crossing | None)

    for cond_name in spec.conditions:
        info = CONDITION_INFO.get(cond_name, ConditionInfo(label=cond_name, color="grey"))
        loggers = loggers_cache.get(cond_name, [])
        if not loggers:
            continue

        x, mean_curve, sem_curve = compute_mean_sem_curve(
            loggers,
            last_ts_in_a_block=params.last_ts_in_a_block,
            phases_to_include=params.phases_to_include,
            aggregate_blocks=params.aggregate_blocks,
        )
        if x is None:
            continue

        skip = params.skip_first_blocks
        if skip > 0:
            if len(x) <= skip:
                continue
            x, mean_curve, sem_curve = x[skip:], mean_curve[skip:], sem_curve[skip:]
        if params.block_group_max is not None:
            x, mean_curve, sem_curve = x[:params.block_group_max], mean_curve[:params.block_group_max], sem_curve[:params.block_group_max]

        ax_top.plot(x, mean_curve, color=info.color, linewidth=1.75, label=info.label)
        ax_top.fill_between(x, mean_curve - sem_curve, mean_curve + sem_curve,
                            color=info.color, alpha=0.2, linewidth=0)

        n_last = min(params.asymptotic_n_last_groups, len(mean_curve))
        asym_mean = float(mean_curve[-n_last:].mean())
        asym_sem  = float(sem_curve[-n_last:].mean())
        asymptotic_pts.append((cond_name, asym_mean, asym_sem))

        # Adaptation speed: first block-group where error drops by adaptation_threshold
        # fraction of its own (initial − asymptote) range.
        initial = float(mean_curve[0])
        if initial > asym_mean:
            thresh = asym_mean + (initial - asym_mean) * (1.0 - params.adaptation_threshold)
            crossings = np.where(mean_curve <= thresh)[0]
            if len(crossings) > 0:
                cross_x = float(x[crossings[0]])
                adaptation_pts.append((cond_name, cross_x))
                ax_top.axvline(cross_x, color=info.color, linewidth=0.8,
                               linestyle=':', alpha=0.55, zorder=0)
            else:
                adaptation_pts.append((cond_name, None))
                print(f"  [{cond_name}] adaptation threshold not reached in window")
        else:
            adaptation_pts.append((cond_name, None))

    x_label = (
        f'Block group (×{params.aggregate_blocks})'
        if params.aggregate_blocks is not None
        else 'Block'
    )
    ax_top.set_xlabel(x_label)
    ax_top.set_ylabel(f'|pred − mean|  (tail {params.last_ts_in_a_block} ts)')
    ax_top.legend(fontsize=7, loc='upper right')

    for cond_name in spec.conditions:
        info    = CONDITION_INFO.get(cond_name, ConditionInfo(label=cond_name, color="grey"))
        loggers = loggers_cache.get(cond_name, [])
        if not loggers:
            continue

        x_lr, mean_lr, sem_lr = compute_block_end_lr(loggers, params)
        if x_lr is None:
            continue
        if params.block_group_max is not None:
            x_lr, mean_lr, sem_lr = x_lr[:params.block_group_max], mean_lr[:params.block_group_max], sem_lr[:params.block_group_max]

        ax_bot.plot(x_lr, mean_lr, color=info.color, linewidth=1.75, label=info.label)
        ax_bot.fill_between(x_lr, mean_lr - sem_lr, mean_lr + sem_lr,
                            color=info.color, alpha=0.2, linewidth=0)

    ax_bot.set_xlabel(x_label)
    ax_bot.set_ylabel('Behavioral LR')
    ax_bot.legend(fontsize=7, loc='upper right')

    # --- Asymptotic summary panel (top-right) ---
    if asymptotic_pts:
        a_conds, a_means, a_sems = zip(*asymptotic_pts)
        xs = np.arange(len(a_conds))
        ax_asy.plot(xs, a_means, color='0.65', linewidth=1.0, zorder=1)
        for i, (cond_name, mean_val, sem_val) in enumerate(asymptotic_pts):
            info = CONDITION_INFO.get(cond_name, ConditionInfo(label=cond_name, color='grey'))
            ax_asy.errorbar(
                xs[i], mean_val, yerr=sem_val,
                fmt='o', color=info.color,
                capsize=3, linewidth=1.5, markersize=6, zorder=2,
            )
        ax_asy.set_xticks(xs)
        ax_asy.set_xticklabels(
            [CONDITION_INFO.get(c, ConditionInfo(label=c, color='grey')).label
             for c in a_conds],
            rotation=45, ha='right', fontsize=7,
        )
        n_last = params.asymptotic_n_last_groups
        group_word = 'group' if n_last == 1 else 'groups'
        ax_asy.set_ylabel(f'Asymptotic error\n(last {n_last} {group_word})', fontsize=8)
        ax_asy.set_title('Asymptote', fontsize=9)

    # --- Adaptation speed panel (bottom-right) ---
    # Shows the block-group index at which each condition's mean curve first drops
    # by adaptation_threshold fraction of its (initial − asymptote) range.
    # Dotted vertical lines on ax_top mark the same crossing points.
    valid_spd = [(c, v) for c, v in adaptation_pts if v is not None]
    if valid_spd:
        spd_x = np.arange(len(adaptation_pts))
        spd_y = [v for _, v in adaptation_pts]
        # connecting line skips None gaps
        valid_mask = np.array([v is not None for _, v in adaptation_pts])
        spd_y_arr = np.array([v if v is not None else np.nan for v in spd_y], dtype=float)
        if valid_mask.sum() > 1:
            ax_spd.plot(
                spd_x[valid_mask],
                spd_y_arr[valid_mask],
                color='0.65', linewidth=1.0, zorder=1,
            )
        for i, (cond_name, speed) in enumerate(adaptation_pts):
            info = CONDITION_INFO.get(cond_name, ConditionInfo(label=cond_name, color='grey'))
            if speed is not None:
                ax_spd.plot(spd_x[i], speed, 'o', color=info.color, markersize=6, zorder=2)
            else:
                ax_spd.plot(spd_x[i], ax_spd.get_ylim()[1] if ax_spd.get_ylim()[1] != 1.0 else 0,
                            '^', color=info.color, markersize=6, alpha=0.4, zorder=2)
        ax_spd.set_xticks(np.arange(len(adaptation_pts)))
        ax_spd.set_xticklabels(
            [CONDITION_INFO.get(c, ConditionInfo(label=c, color='grey')).label
             for c, _ in adaptation_pts],
            rotation=45, ha='right', fontsize=7,
        )
        frac_pct = int(params.adaptation_threshold * 100)
        ax_spd.set_ylabel(x_label, fontsize=8)
        ax_spd.set_title(f'Learning speed\n({frac_pct}% late-error reduction)', fontsize=9)

    fig.suptitle(spec.title, fontsize=10)

    if params.save_plots:
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / f"{spec.key}.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved → {out}")

    if params.show_plots:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_switch_aligned(
    params: AnalysisParams,
    export_dir: Path,
    loggers_cache: Dict[str, List],
) -> plt.Figure:
    """Switch-aligned |pred − input| curves + adaptation-speed summary panel.

    Floor  = mean error over the pre-switch window (t < 0).
    Peak   = error at t = 0 (the switch).
    Threshold = floor + (peak − floor) × (1 − adaptation_switch_threshold).
    Adaptation speed = timesteps post-switch to first reach threshold.
    """
    pre  = params.switch_pre_window
    post = params.switch_post_window
    x    = np.arange(-pre, post)

    summary_w = params.subplot_height * 0.9
    fig = plt.figure(
        figsize=(params.subplot_width + summary_w, params.subplot_height),
        dpi=params.dpi,
        layout='constrained',
    )
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[params.subplot_width, summary_w],
        figure=fig,
    )
    ax     = fig.add_subplot(gs[0, 0])
    ax_spd = fig.add_subplot(gs[0, 1])

    adaptation_pts: List[Tuple[str, float | None]] = []

    for cond_name, info in CONDITION_INFO.items():
        loggers = loggers_cache.get(cond_name, [])
        if not loggers:
            continue

        windows = extract_switch_aligned_windows(loggers, pre, post, cond_name)
        if windows is None:
            continue

        mean = windows.mean(axis=0)
        sem  = windows.std(axis=0, ddof=1) / np.sqrt(len(windows))

        ax.plot(x, mean, color=info.color, linewidth=1.75, label=info.label)
        ax.fill_between(x, mean - sem, mean + sem,
                        color=info.color, alpha=0.2, linewidth=0)

        # Adaptation speed: how many post-switch timesteps to drop
        # adaptation_switch_threshold of the (peak − floor) range.
        floor = float(mean[:pre].mean())   # pre-switch steady state
        peak  = float(mean[pre])           # error at the switch (t=0)
        if peak > floor:
            thresh = floor + (peak - floor) * (1.0 - params.adaptation_switch_threshold)
            crossings = np.where(mean[pre:] <= thresh)[0]
            if len(crossings) > 0:
                speed = float(crossings[0])
                adaptation_pts.append((cond_name, speed))
                ax.axvline(speed, color=info.color, linewidth=0.8,
                           linestyle=':', alpha=0.55, zorder=0)
            else:
                adaptation_pts.append((cond_name, None))
                print(f"  [{cond_name}] switch adaptation threshold not reached in post-window")
        else:
            adaptation_pts.append((cond_name, None))

    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xlabel('Timestep relative to block switch')
    ax.set_ylabel('|pred − input|')
    ax.set_title('Adaptation around block switch  (first ⅓ of training)', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')

    # --- Adaptation speed summary ---
    valid_spd = [(c, v) for c, v in adaptation_pts if v is not None]
    if valid_spd:
        spd_x     = np.arange(len(adaptation_pts))
        valid_mask = np.array([v is not None for _, v in adaptation_pts])
        spd_y_arr  = np.array([v if v is not None else np.nan for _, v in adaptation_pts],
                               dtype=float)
        if valid_mask.sum() > 1:
            ax_spd.plot(spd_x[valid_mask], spd_y_arr[valid_mask],
                        color='0.65', linewidth=1.0, zorder=1)
        for i, (cond_name, speed) in enumerate(adaptation_pts):
            info = CONDITION_INFO.get(cond_name, ConditionInfo(label=cond_name, color='grey'))
            if speed is not None:
                ax_spd.plot(spd_x[i], speed, 'o', color=info.color, markersize=6, zorder=2)
            else:
                ax_spd.plot(spd_x[i], post, '^', color=info.color,
                            markersize=6, alpha=0.4, zorder=2)
        ax_spd.set_xticks(spd_x)
        ax_spd.set_xticklabels(
            [CONDITION_INFO.get(c, ConditionInfo(label=c, color='grey')).label
             for c, _ in adaptation_pts],
            rotation=45, ha='right', fontsize=7,
        )
        frac_pct = int(params.adaptation_switch_threshold * 100)
        ax_spd.set_ylabel('Timesteps post-switch', fontsize=8)
        ax_spd.set_title(f'Adaptation speed\n({frac_pct}% reduction)', fontsize=9)

    if params.save_plots:
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / "switch_aligned.pdf"
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

def load_all_loggers(params: AnalysisParams) -> Dict[str, List]:
    """Load loggers for every condition once and return a shared cache."""
    print("Loading loggers...")
    cache = {}
    for cond_name in CONDITION_INFO:
        cache[cond_name] = load_loggers(cond_name, params.n_seeds)
    return cache


def main() -> dict:
    global loggers_cache
    analysis_params = AnalysisParams()
    export_dir      = EXPORT_ROOT / "figures"

    if 'loggers_cache' not in globals() or loggers_cache is None:
        loggers_cache = load_all_loggers(analysis_params)

    for spec in CANONICAL_SPECS:
        print(f"\n=== {spec.key} ===")
        plot_canonical(spec, analysis_params, export_dir, loggers_cache)

    print("\n=== switch_aligned ===")
    plot_switch_aligned(analysis_params, export_dir, loggers_cache)

    return loggers_cache


if __name__ == "__main__":
    loggers_cache = main()
