"""Analysis of correlated-noise sweep results.

Loads pickles produced by cst_correlated_noise_sweep.py, extracts per-block
|pred - mean| curves via extract_block_corrects(), averages over seeds, and
plots one subplot per model with one line per noise_correlation_tau value.

Usage (interactive or script):
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import plot_style
from functions_and_utils import extract_block_corrects

plot_style.set_plot_style()
COLOR_SCHEME = plot_style.Color_scheme()


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInfo:
    """Display metadata for one model variant."""
    label: str
    color: str
    base_params: Mapping[str, Any] = field(default_factory=dict)


MODEL_INFO: Dict[str, ModelInfo] = {
    "rnn_short": ModelInfo(
        label=r"RNN$^{\mathrm{short}}$",
        color=COLOR_SCHEME.short_horizon_rnn,
        base_params={},
    ),
    "rnn_long": ModelInfo(
        label=r"RNN$^{\mathrm{long}}$",
        color=COLOR_SCHEME.long_horizon_rnn,
        base_params={},
    ),
    "neuragem": ModelInfo(
        label="NeuraGEM",
        color=COLOR_SCHEME.neuragem,
        base_params={},
    ),
}


@dataclass(frozen=True)
class ComparisonSpec:
    """One figure: which models to include and which extra params each carries."""
    key: str
    title: str
    # model_name → extra params beyond noise_correlation_tau (e.g. l2_loss, LU_lr)
    model_params: Mapping[str, Mapping[str, Any]]
    tau_values: Sequence[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass(frozen=True)
class HyperparamSweepSpec:
    """Orthogonal to ComparisonSpec: fix tau, vary hyperparam combos on one subplot.

    ComparisonSpec answers "how does model X behave across noise levels?" — one
    subplot per model, one line per tau.

    HyperparamSweepSpec answers "which hyperparams work best at a given noise
    level?" — one subplot total, one line per (label, params) variant, tau fixed.
    Each variant is a (line_label, extra_params_beyond_tau) tuple so multiple
    combos of the same model_name can coexist without dict-key collisions.
    """
    key: str
    title: str
    model_name: str                                      # must exist in MODEL_INFO
    fixed_tau: int                                       # single noise_correlation_tau
    # Ordered list of (legend label, extra params).  Order sets legend order.
    variants: Sequence[Tuple[str, Mapping[str, Any]]]


@dataclass
class AnalysisParams:
    """Global knobs for the analysis pipeline."""
    n_seeds: int = 10
    last_ts_in_a_block: int = 15
    phases_to_include: str = 'Learning and inference'
    aggregate_blocks: int | None = 5   # None = one point per block
    subplot_width: float = 3.0
    subplot_height: float = 2.5
    dpi: int = 100
    show_plots: bool = True
    save_plots: bool = True
    # Drop the first N blocks from the plot (useful to skip warm-up / pre-convergence).
    # Data is still loaded in full; slicing happens at render time so x-axis labels
    # reflect true block numbers (e.g. blocks 10..N, not 0..N-10).
    skip_first_blocks: int = 0

# ---------------------------------------------------------------------------
# Comparison specifications — edit here to change what gets plotted
# ---------------------------------------------------------------------------

COMPARISON_SPECS: List[ComparisonSpec] = [
    ComparisonSpec(
        key="all_models_l2_0p0001_LU_0p1", # a unique name for saving the figures.
        title="Correlated noise: all models",
        model_params={
            "rnn_short":  {},
            "rnn_long":   {},
            "neuragem":   {"l2_loss": 0.0001, "LU_lr": 0.3},
        },
    ),
]


RUN_NAME   = "paired_lr_sweep_lu_first"
EXPORT_ROOT = Path(f"./exports/correlated_noise/{RUN_NAME}")

# HyperparamSweepSpec figures are generated independently of COMPARISON_SPECS —
# adding entries here does not affect any existing ComparisonSpec output.
# Use this to put multiple NeuraGEM hyperparam combos on the same subplot at a
# chosen tau, which a ComparisonSpec cannot do (dict keys must be unique).
HYPERPARAM_SWEEP_SPECS: List[HyperparamSweepSpec] = [
    HyperparamSweepSpec(
        key="neuragem_tau1_hyperparam", # a unique name for saving the figures.
        title="NeuraGEM hyperparam sweep at τ=1",
        model_name="neuragem",
        fixed_tau=1,
        variants=[
            ("l2=5e-5, LU_lr=0.1",  {"l2_loss": 0.00005, "LU_lr": 0.1}),
            ("l2=1e-4, LU_lr=0.3",  {"l2_loss": 0.0001,  "LU_lr": 0.3}),
            ("l2=3e-4, LU_lr=0.5",  {"l2_loss": 0.0003,  "LU_lr": 0.5}),
            ("l2=8e-4, LU_lr=0.7",  {"l2_loss": 0.0008,  "LU_lr": 0.7}),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Filename construction (must mirror cst_correlated_noise_sweep.py exactly)
# ---------------------------------------------------------------------------

def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def _combo_key(params: Mapping[str, Any]) -> str:
    """Stable folder-name key from params (excluding 'seed')."""
    filtered = {k: v for k, v in params.items() if k != "seed"}
    return "_".join(
        f"{k}-{_format_param_value(v)}" for k, v in sorted(filtered.items())
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_loggers(
    model_name: str,
    params: Mapping[str, Any],
    n_seeds: int,
) -> List[Any]:
    """Load train_loggers for all seeds of one param combination.

    Filenames are constructed explicitly — no directory globbing.
    """
    key = _combo_key(params)
    folder = EXPORT_ROOT / model_name / key
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
        print(f"  [{model_name}/{key}] missing seeds: {missing}")
    print(f"  [{model_name}/{key}] loaded {len(loggers)}/{n_seeds} seeds")
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
        )  # (n_seeds, n_groups)
        x = np.arange(n_groups)
        mean_curve = agg.mean(axis=0)
        sem_curve = agg.std(axis=0, ddof=1) / np.sqrt(len(loggers))
    else:
        x = np.arange(min_len)
        mean_curve = arr.mean(axis=0)
        sem_curve = arr.std(axis=0, ddof=1) / np.sqrt(len(loggers))

    return x, mean_curve, sem_curve


def _estimate_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Linear regression slope and its SE (y ~ β·x + intercept)."""
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
    params: 'AnalysisParams',
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Estimate behavioral LR from the tail of each block, pooled across seeds.

    For each block, the last `params.last_ts_in_a_block` timesteps supply
    consecutive (pred_err, delta_pred) pairs.  Pairs are pooled across seeds
    (and across blocks within a group when aggregate_blocks is set) before a
    linear regression slope is fit.  The slope is the behavioral learning rate.

    Returns (x, mean_lr, sem_lr) or (None, None, None) when no data is available.
      aggregate_blocks is None  →  one slope per block;  sem = regression SE.
      aggregate_blocks = N      →  one slope per N-block group; sem = cross-seed SEM.
    """
    per_seed_block_pairs: List[List[Tuple[np.ndarray, np.ndarray]]] = []

    for logger in loggers:
        ll  = np.concatenate(logger.llcids,           axis=0).reshape(-1)
        oi  = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1)
        inp = np.concatenate(logger.inputs,            axis=0).reshape(-1)

        T = len(ll)
        block_starts = [0] + [t for t in range(1, T) if ll[t] != ll[t - 1]]
        block_ends   = block_starts[1:] + [T]

        block_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for start, end in zip(block_starts, block_ends):
            tail_start = max(start, end - params.last_ts_in_a_block)
            # pairs (i, i+1) must both sit within [start, end)
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
        # ---- one slope per block, pooled across seeds ----
        x        = np.arange(n_blocks_min)
        mean_lr  = np.full(n_blocks_min, np.nan)
        sem_lr   = np.full(n_blocks_min, np.nan)

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
        # ---- one slope per N-block group, SEM across seeds ----
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

        arr     = np.array(per_seed_group_lr)          # (n_seeds, n_groups)
        n_valid = np.sum(~np.isnan(arr), axis=0).clip(min=1)
        mean_lr = np.nanmean(arr, axis=0)
        sem_lr  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_valid)
        return np.arange(n_groups), mean_lr, sem_lr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    spec: ComparisonSpec,
    params: AnalysisParams,
    export_dir: Path,
) -> plt.Figure:
    """One subplot per model, one line per tau value."""
    model_names = list(spec.model_params.keys())
    n_models = len(model_names)
    tau_values = list(spec.tau_values)
    tau_cmap = plt.cm.viridis
    tau_colors = [tau_cmap(i / max(len(tau_values) - 1, 1)) for i in range(len(tau_values))]

    fig, axes_2d = plt.subplots(
        2, n_models,
        figsize=(params.subplot_width * n_models, params.subplot_height * 2),
        dpi=params.dpi,
        constrained_layout=True,
        squeeze=False,
    )
    axes_top = axes_2d[0]
    axes_bot = axes_2d[1]

    loggers_cache: Dict[Tuple, List] = {}

    for ax, model_name in zip(axes_top, model_names):
        info = MODEL_INFO.get(model_name, ModelInfo(label=model_name, color="grey"))

        for color, tau in zip(tau_colors, tau_values):
            combo_params = {**spec.model_params[model_name], "noise_correlation_tau": tau}
            cache_key = (model_name, tau)
            if cache_key not in loggers_cache:
                loggers_cache[cache_key] = load_loggers(model_name, combo_params, params.n_seeds)
            loggers = loggers_cache[cache_key]
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

            # Trim warm-up blocks from the display window (x labels stay correct).
            skip = params.skip_first_blocks
            if skip > 0:
                if len(x) <= skip:
                    continue
                x, mean_curve, sem_curve = x[skip:], mean_curve[skip:], sem_curve[skip:]

            ax.plot(x, mean_curve, color=color, linewidth=1.5, label=f'τ={tau}')
            ax.fill_between(x, mean_curve - sem_curve, mean_curve + sem_curve,
                            color=color, alpha=0.2, linewidth=0)

        ax.set_title(info.label, fontsize=9)
        ax.set_xlabel(
            'Block' if params.aggregate_blocks is None
            else f'Block group (×{params.aggregate_blocks})'
        )
        ax.set_ylabel('|pred − mean|')
        ax.legend(fontsize=7, loc='upper right', title='τ', title_fontsize=7)

    for ax_lr, model_name in zip(axes_bot, model_names):
        info = MODEL_INFO.get(model_name, ModelInfo(label=model_name, color="grey"))

        for color, tau in zip(tau_colors, tau_values):
            loggers = loggers_cache.get((model_name, tau), [])
            if not loggers:
                continue

            x_lr, mean_lr, sem_lr = compute_block_end_lr(loggers, params)
            if x_lr is None:
                continue

            ax_lr.plot(x_lr, mean_lr, color=color, linewidth=1.5, label=f'τ={tau}')
            ax_lr.fill_between(x_lr, mean_lr - sem_lr, mean_lr + sem_lr,
                               color=color, alpha=0.2, linewidth=0)

        ax_lr.set_title(info.label, fontsize=9)
        ax_lr.set_xlabel(
            'Block' if params.aggregate_blocks is None
            else f'Block group (×{params.aggregate_blocks})'
        )
        ax_lr.set_ylabel('Behavioral LR')
        ax_lr.legend(fontsize=7, loc='upper right', title='τ', title_fontsize=7)

    fig.suptitle(spec.title, fontsize=10)

    if params.save_plots:
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / f"{spec.key}_tau_sweep.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved → {out}")

    if params.show_plots:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_hyperparam_sweep(
    spec: HyperparamSweepSpec,
    params: AnalysisParams,
    export_dir: Path,
) -> plt.Figure:
    """Single subplot: one line per hyperparam variant of one model at a fixed tau.

    This mirrors the inner loop of plot_comparison (load → compute → draw) but
    iterates over hyperparam variants instead of tau values.  A qualitative
    colormap (tab10) keeps adjacent lines maximally distinct regardless of how
    many variants are listed.
    """
    info = MODEL_INFO.get(spec.model_name, ModelInfo(label=spec.model_name, color="grey"))
    n_variants = len(spec.variants)
    # tab10 has 10 perceptually distinct hues — sufficient for hyperparam comparisons.
    colors = [plt.cm.tab10(i / 10) for i in range(n_variants)]

    fig, axes_2d = plt.subplots(
        2, 1,
        figsize=(params.subplot_width, params.subplot_height * 2),
        dpi=params.dpi,
        constrained_layout=True,
        squeeze=False,
    )
    ax    = axes_2d[0, 0]
    ax_lr = axes_2d[1, 0]

    hyperparam_loggers_cache: Dict[str, List] = {}

    for color, (line_label, extra_params) in zip(colors, spec.variants):
        combo_params = {**extra_params, "noise_correlation_tau": spec.fixed_tau}
        loggers = load_loggers(spec.model_name, combo_params, params.n_seeds)
        hyperparam_loggers_cache[line_label] = loggers
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
                print(f"  [{spec.model_name}/{line_label}] fewer blocks than skip_first_blocks={skip}, skipping")
                continue
            x, mean_curve, sem_curve = x[skip:], mean_curve[skip:], sem_curve[skip:]

        ax.plot(x, mean_curve, color=color, linewidth=1.5, label=line_label)
        ax.fill_between(x, mean_curve - sem_curve, mean_curve + sem_curve,
                        color=color, alpha=0.2, linewidth=0)

    ax.set_title(f"{info.label}  (τ={spec.fixed_tau})", fontsize=9)
    ax.set_xlabel(
        'Block' if params.aggregate_blocks is None
        else f'Block group (×{params.aggregate_blocks})'
    )
    ax.set_ylabel('|pred − mean|')
    ax.legend(fontsize=7, loc='upper right')

    for color, (line_label, extra_params) in zip(colors, spec.variants):
        loggers = hyperparam_loggers_cache.get(line_label, [])
        if not loggers:
            continue

        x_lr, mean_lr, sem_lr = compute_block_end_lr(loggers, params)
        if x_lr is None:
            continue

        ax_lr.plot(x_lr, mean_lr, color=color, linewidth=1.5, label=line_label)
        ax_lr.fill_between(x_lr, mean_lr - sem_lr, mean_lr + sem_lr,
                           color=color, alpha=0.2, linewidth=0)

    ax_lr.set_title(f"{info.label}  (τ={spec.fixed_tau})", fontsize=9)
    ax_lr.set_xlabel(
        'Block' if params.aggregate_blocks is None
        else f'Block group (×{params.aggregate_blocks})'
    )
    ax_lr.set_ylabel('Behavioral LR')
    ax_lr.legend(fontsize=7, loc='upper right')

    fig.suptitle(spec.title, fontsize=10)

    if params.save_plots:
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / f"{spec.key}_hyperparam_sweep.pdf"
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

    for spec in COMPARISON_SPECS:
        print(f"\n=== {spec.key} ===")
        plot_comparison(spec, analysis_params, export_dir)

    for spec in HYPERPARAM_SWEEP_SPECS:
        print(f"\n=== {spec.key} ===")
        plot_hyperparam_sweep(spec, analysis_params, export_dir)


if __name__ == "__main__":
    main()
