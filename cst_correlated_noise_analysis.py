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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import plot_style
from functions_and_utils import extract_block_corrects

plot_style.set_plot_style()
COLOR_SCHEME = plot_style.Color_scheme()


# ---------------------------------------------------------------------------
# Condition metadata — mirrors cst_correlated_noise_sweep.CONDITIONS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionInfo:
    label: str
    color: Any


CONDITION_INFO: Dict[str, ConditionInfo] = {
    "rnn":               ConditionInfo("RNN",                         COLOR_SCHEME.short_horizon_rnn),
    "neuragem":          ConditionInfo("NeuraGEM",                    COLOR_SCHEME.neuragem),
    "neuragem_lu_first": ConditionInfo("NeuraGEM LU-first",           plt.cm.tab10(2)),   # green
    "neuragem_slow":     ConditionInfo(r"NeuraGEM slow (LU_lr=0.2)",  plt.cm.tab10(1)),   # orange
    "neuragem_fast":     ConditionInfo(r"NeuraGEM fast (LU_lr=0.7)",  plt.cm.tab10(3)),   # red
}


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
    skip_first_blocks: int = 0          # drop warm-up blocks from display


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

RUN_NAME    = "canonical_conditions"
EXPORT_ROOT = Path(f"./exports/canonical/{RUN_NAME}")


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
        ll  = np.concatenate(logger.llcids,            axis=0).reshape(-1)
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
# Plotting
# ---------------------------------------------------------------------------

def plot_canonical(
    spec: CanonicalSpec,
    params: AnalysisParams,
    export_dir: Path,
) -> plt.Figure:
    """Two stacked subplots (|pred−mean|, behavioral LR), one line per condition."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(params.subplot_width, params.subplot_height * 2),
        dpi=params.dpi,
        constrained_layout=True,
    )

    loggers_cache: Dict[str, List] = {}

    for cond_name in spec.conditions:
        info = CONDITION_INFO.get(cond_name, ConditionInfo(label=cond_name, color="grey"))
        loggers = load_loggers(cond_name, params.n_seeds)
        loggers_cache[cond_name] = loggers
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

        ax_top.plot(x, mean_curve, color=info.color, linewidth=1.5, label=info.label)
        ax_top.fill_between(x, mean_curve - sem_curve, mean_curve + sem_curve,
                            color=info.color, alpha=0.2, linewidth=0)

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

        ax_bot.plot(x_lr, mean_lr, color=info.color, linewidth=1.5, label=info.label)
        ax_bot.fill_between(x_lr, mean_lr - sem_lr, mean_lr + sem_lr,
                            color=info.color, alpha=0.2, linewidth=0)

    ax_bot.set_xlabel(x_label)
    ax_bot.set_ylabel('Behavioral LR')
    ax_bot.legend(fontsize=7, loc='upper right')

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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    analysis_params = AnalysisParams()
    export_dir = EXPORT_ROOT / "figures"

    for spec in CANONICAL_SPECS:
        print(f"\n=== {spec.key} ===")
        plot_canonical(spec, analysis_params, export_dir)


if __name__ == "__main__":
    main()
