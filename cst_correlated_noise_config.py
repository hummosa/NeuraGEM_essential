"""Shared configuration for the correlated-noise sweep and analysis scripts.

Edit this file to change conditions, seeds, training overrides, export paths,
or display metadata. Both cst_correlated_noise_sweep.py and
cst_correlated_noise_analysis.py import exclusively from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

import plot_style

COLOR_SCHEME = plot_style.Color_scheme()


# ── Sweep conditions ──────────────────────────────────────────────────────────

# Z_decay scales as lr^2 (power-law fit to hand-tuned values; rough approximation).
_l2_for_lr = lambda lr: 1e-3 * lr ** 2

_NG_LRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CONDITIONS: Dict[str, Dict[str, Any]] = {
    "rnn": dict(no_of_steps_in_latent_space=0),
    **{
        rf"NG$\alpha_z={lr}$": dict(
            no_of_steps_in_latent_space=1,
            Z_lr=lr,
            Z_decay=_l2_for_lr(lr),
        )
        for lr in _NG_LRS
    },
}

# Alternative: original canonical conditions (uncomment and reassign CONDITIONS above)
# CONDITIONS_CANONICAL: Dict[str, Dict[str, Any]] = {
#     "rnn":               dict(no_of_steps_in_latent_space=0),
#     "neuragem":          dict(no_of_steps_in_latent_space=1),
#     "neuragem_lu_first": dict(no_of_steps_in_latent_space=1, update_latent_before_weights=True),
#     "neuragem_slow":     dict(no_of_steps_in_latent_space=1, Z_lr=0.2, Z_decay=5e-5),
#     "neuragem_fast":     dict(no_of_steps_in_latent_space=1, Z_lr=0.7, Z_decay=8e-4),
# }

DEFAULT_SEEDS: int = 10

# Shared overrides applied to every condition during training.
TRAIN_OVERRIDES: Dict[str, Any] = {
    "blocked_phase_length": 5000,
    "correlated_noise":     False,
    "default_std":          0.1,
}

SKIP_EXISTING: bool = False


# ── Export paths ──────────────────────────────────────────────────────────────

RUN_NAME    = "varying_alpha_z"
EXPORT_ROOT = Path(f"./exports/canonical/{RUN_NAME}")


# ── Condition display metadata ────────────────────────────────────────────────

@dataclass(frozen=True)
class ConditionInfo:
    label: str
    color: Any


CONDITION_INFO: Dict[str, ConditionInfo] = {
    "rnn": ConditionInfo("RNN", COLOR_SCHEME.contextB),
    **{
        rf"NG$\alpha_z={lr}$": ConditionInfo(
            rf"NG $\alpha_z={lr}$",
            plt.cm.viridis(lr),
        )
        for lr in _NG_LRS
    },
}

# Alternative: canonical-condition display metadata
# CONDITION_INFO_CANONICAL: Dict[str, ConditionInfo] = {
#     "rnn":               ConditionInfo("RNN",                        COLOR_SCHEME.short_horizon_rnn),
#     "neuragem":          ConditionInfo("NeuraGEM",                   COLOR_SCHEME.neuragem),
#     "neuragem_lu_first": ConditionInfo("NeuraGEM LU-first",          plt.cm.tab10(2)),
#     "neuragem_slow":     ConditionInfo(r"NeuraGEM slow (Z_lr=0.2)", plt.cm.tab10(1)),
#     "neuragem_fast":     ConditionInfo(r"NeuraGEM fast (Z_lr=0.7)", plt.cm.tab10(3)),
# }
