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
    tau_values: Sequence[int] = field(default_factory=lambda: [1, 2, 4, 6, 10])


@dataclass
class AnalysisParams:
    """Global knobs for the analysis pipeline."""
    n_seeds: int = 20
    last_ts_in_a_block: int = 15
    phases_to_include: str = 'Learning and inference'
    aggregate_blocks: int | None = None   # None = one point per block
    subplot_width: float = 3.0
    subplot_height: float = 2.5
    dpi: int = 100
    show_plots: bool = True
    save_plots: bool = True


EXPORT_ROOT = Path("./exports/correlated_noise")

# ---------------------------------------------------------------------------
# Comparison specifications — edit here to change what gets plotted
# ---------------------------------------------------------------------------

COMPARISON_SPECS: List[ComparisonSpec] = [
    ComparisonSpec(
        key="all_models_l2_0p0001_LU_0p1",
        title="Correlated noise: all models",
        model_params={
            "rnn_short":  {},
            "rnn_long":   {},
            "neuragem":   {"l2_loss": 0.0001, "LU_lr": 0.1},
        },
    ),
    # Uncomment to compare NeuraGEM hyperparams:
    # ComparisonSpec(
    #     key="neuragem_hyperparam_sweep",
    #     title="NeuraGEM hyperparam comparison",
    #     model_params={
    #         "neuragem": {"l2_loss": 0.0001, "LU_lr": 0.1},
    #         "neuragem": {"l2_loss": 0.0008, "LU_lr": 0.05},
    #     },
    # ),
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

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(params.subplot_width * n_models, params.subplot_height),
        dpi=params.dpi,
        constrained_layout=True,
    )
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        info = MODEL_INFO.get(model_name, ModelInfo(label=model_name, color="grey"))

        for color, tau in zip(tau_colors, tau_values):
            combo_params = {**spec.model_params[model_name], "noise_correlation_tau": tau}
            loggers = load_loggers(model_name, combo_params, params.n_seeds)
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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    analysis_params = AnalysisParams()
    export_dir = EXPORT_ROOT / "figures"

    for spec in COMPARISON_SPECS:
        print(f"\n=== {spec.key} ===")
        plot_comparison(spec, analysis_params, export_dir)


if __name__ == "__main__":
    main()
