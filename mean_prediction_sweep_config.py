"""Shared configuration for the mean-prediction task sweep and analysis scripts.

Sweeps NeuraGEM over a range of Z_lr values (with Z_decay paired via a power-law
formula), plus a plain RNN baseline. Structure mirrors cst_correlated_noise_config.py.

Edit this file to change conditions, seeds, training overrides, or export paths.
Both mean_prediction_sweep.py and mean_prediction_analysis.py import exclusively
from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

import plot_style

COLOR_SCHEME = plot_style.Color_scheme()


# ── Sweep conditions ──────────────────────────────────────────────────────────

# Z_decay scales as Z_lr^2 (power-law fit to hand-tuned values; rough approximation).
_z_decay_for_lr = lambda lr: 1e-3 * lr ** 2

import numpy as np
_Z_LRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# replace with something log spaced that passes near 0.4
# _Z_LRS = np.logspace(np.log10(0.1), np.log10(0.9), num=9)
# Round to 2 decimal places for display purposes
# _Z_LRS = [round(lr, 2) for lr in _Z_LRS]

CONDITIONS: Dict[str, Dict[str, Any]] = {
    "rnn": dict(no_of_steps_in_latent_space=0),
    **{
        rf"NG$\alpha_z={lr}$": dict(
            no_of_steps_in_latent_space=1,
            Z_lr=lr,
            Z_decay=_z_decay_for_lr(lr),
        )
        for lr in _Z_LRS
    },
}

DEFAULT_SEEDS: int = 10

# Shared overrides applied to every condition during training.
TRAIN_OVERRIDES: Dict[str, Any] = {
    "blocked_phase_length": 400,
    "default_std":          0.4,
    "test_block_size": 40,
    'seq_len': 4,
}

SKIP_EXISTING: bool = False


# ── Export paths ──────────────────────────────────────────────────────────────

# RUN_NAME    = "initial_runs_1000_training_length"
RUN_NAME    = f"initial_runs_seq_len_{TRAIN_OVERRIDES['seq_len']}_blocked_phase_{TRAIN_OVERRIDES['blocked_phase_length']}"
EXPORT_ROOT = Path(f"./exports/mean_prediction/{RUN_NAME}")


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
        for lr in _Z_LRS
    },
}
