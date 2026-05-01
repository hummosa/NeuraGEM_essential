"""Sweep correlated noise effects on RNN (short), MRNN (long), and NeuraGEM.

noise_correlation_tau is swept over [1, 2, 4, 6, 10] with 20 seeds per combination.
NeuraGEM additionally sweeps l2_loss and LU_lr.

Run locally (sequential over all combinations):
    python cst_correlated_noise_sweep.py

Run on SLURM (one job per array element):
    #SBATCH --array=0-<N-1>
    python cst_correlated_noise_sweep.py
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from configs import ContextualSwitchingTaskConfig
from train_and_infer_functions import train_model


# ── Experiment configuration ──────────────────────────────────────────────────

DEFAULT_SEEDS = 10

# Grid convention:
#   string key  → one scalar parameter swept independently (full Cartesian product).
#   tuple key   → multiple parameters swept together as atomic units (not crossed).
#
# Example of the tuple form used below:
#   ("l2_loss", "LU_lr"): [(0.00005, 0.1), (0.0001, 0.3), ...]
#   means run (l2_loss=0.00005, LU_lr=0.1) as one job, (l2_loss=0.0001, LU_lr=0.3)
#   as another, etc. — never the cross-product of the two lists.
#   The tuple still participates in the Cartesian product with all other keys
#   (e.g. noise_correlation_tau and seed), so each paired combo is run for every
#   tau and every seed.
PARAM_GRIDS: Dict[str, Any] = {
    "neuragem": {
        "noise_correlation_tau":  [1, ],
        ("l2_loss", "LU_lr"):     [(0.00005, 0.1), (0.0001, 0.3), (0.0003, 0.5), (0.0008, 0.7)],
        "seed":                   list(range(DEFAULT_SEEDS)),
    },
    "rnn_short": {
        "noise_correlation_tau": [1, ],
        "seed": list(range(DEFAULT_SEEDS)),
    },
    "rnn_long": {
        "noise_correlation_tau": [1, ],
        "seed": list(range(DEFAULT_SEEDS)),
    },
}

TRAIN_OVERRIDES: Dict[str, Any] = {
    "blocked_phase_length": 5000,
    "correlated_noise": False,
    "update_latent_before_weights": False,

}

RUN_NAME   = "not_correlated"
EXPORT_ROOT = f"./exports/correlated_noise/{RUN_NAME}"
SKIP_EXISTING = False  # Whether to skip jobs with existing results (based on filepath presence).

# ── Job dataclass & helpers ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentJob:
    model_name: str
    param_combination: Dict[str, Any]


def generate_jobs() -> List[ExperimentJob]:
    jobs = []
    for model_name, grid in PARAM_GRIDS.items():
        keys = list(grid.keys())
        for combo in product(*grid.values()):
            params: Dict[str, Any] = {}
            for key, val in zip(keys, combo):
                if isinstance(key, tuple):
                    # Tuple key: expand each paired param individually into the flat dict.
                    params.update(zip(key, val))
                else:
                    params[key] = val
            jobs.append(ExperimentJob(model_name=model_name, param_combination=params))
    return jobs


def combo_key(params: Dict[str, Any]) -> str:
    """Stable string key from non-seed params, used for folder naming."""
    filtered = {k: v for k, v in params.items() if k != "seed"}
    return "_".join(f"{k}-{v}" for k, v in sorted(filtered.items()))


# ── Single job execution ──────────────────────────────────────────────────────

def run_job(job: ExperimentJob, job_index: int | None = None, total: int | None = None) -> None:
    params = job.param_combination
    seed = params.get("seed", 0)
    key = combo_key(params)

    export_path = os.path.join(EXPORT_ROOT, job.model_name, key)
    os.makedirs(export_path, exist_ok=True)
    filepath = os.path.join(export_path, f"results_seed-{seed}.pkl")

    prefix = f"[{job_index+1}/{total}] " if job_index is not None else ""

    if SKIP_EXISTING and os.path.exists(filepath):
        print(f"{prefix}Skipping (exists): {filepath}")
        return

    print(f"{prefix}{job.model_name} | {params}")

    config = ContextualSwitchingTaskConfig(experiment_to_run="figure")

    # Apply sweep params (except seed)
    for k, v in params.items():
        if k == "seed":
            continue
        setattr(config, k, v)
    config.env_seed = seed

    # Apply shared training overrides
    for k, v in TRAIN_OVERRIDES.items():
        setattr(config, k, v)

    # Model-specific architecture
    if job.model_name == "rnn_short":
        config.no_of_steps_in_latent_space = 0
    elif job.model_name == "rnn_long":
        config.seq_len = 50
        config.no_of_steps_in_latent_space = 0
    # neuragem: default seq_len and latent updates

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_logger, model, config, _ = train_model(
        config, seed=seed, run_test_phase=False,
        save_models=False, load_models=False,
    )

    with open(filepath, "wb") as f:
        pickle.dump({"train_logger": train_logger, "config": config}, f)
    print(f"  Saved → {filepath}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    jobs = generate_jobs()
    total = len(jobs)
    print(f"Total jobs: {total}")

    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_str is None:
        print("No SLURM_ARRAY_TASK_ID — running all jobs sequentially.")
        for idx, job in enumerate(jobs):
            run_job(job, job_index=idx, total=total)
    else:
        task_id = int(task_id_str)
        if task_id < 0 or task_id >= total:
            raise ValueError(f"Task id {task_id} out of range [0, {total-1}].")
        run_job(jobs[task_id], job_index=task_id, total=total)


if __name__ == "__main__":
    main()
