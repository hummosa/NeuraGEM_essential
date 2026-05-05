"""Sweep canonical RNN vs NeuraGEM conditions over multiple seeds.

Conditions mirror run_comparisons.py exactly:
  rnn               : no_of_steps_in_latent_space=0
  neuragem          : no_of_steps_in_latent_space=1
  neuragem_lu_first : update_latent_before_weights=True
  neuragem_slow     : LU_lr=0.2, l2_loss=5e-5
  neuragem_fast     : LU_lr=0.7, l2_loss=8e-4

Correlated noise is disabled. No tau sweep.

Run locally (sequential):
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
from typing import Any, Dict, List

import numpy as np
import torch

from configs import ContextualSwitchingTaskConfig
from train_and_infer_functions import train_model


# ── Canonical conditions (mirrors run_comparisons.py) ────────────────────────

CONDITIONS: Dict[str, Dict[str, Any]] = {
    "rnn":               dict(no_of_steps_in_latent_space=0),
    "neuragem":          dict(no_of_steps_in_latent_space=1),
    "neuragem_lu_first": dict(no_of_steps_in_latent_space=1, update_latent_before_weights=True),
    "neuragem_slow":     dict(no_of_steps_in_latent_space=1, LU_lr=0.2,  l2_loss=5e-5),
    "neuragem_fast":     dict(no_of_steps_in_latent_space=1, LU_lr=0.7,  l2_loss=8e-4),
}

DEFAULT_SEEDS = 10

# Each condition is swept only over seeds — no other grid axes.
PARAM_GRIDS: Dict[str, Any] = {
    name: {"seed": list(range(DEFAULT_SEEDS))}
    for name in CONDITIONS
}

# Shared overrides applied to every condition (correlated noise is off).
TRAIN_OVERRIDES: Dict[str, Any] = {
    "blocked_phase_length": 5000,
    "correlated_noise":     False,
    "default_std":          0.1,
}

RUN_NAME    = "canonical_conditions"
EXPORT_ROOT = f"./exports/canonical/{RUN_NAME}"
SKIP_EXISTING = False


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
                    params.update(zip(key, val))
                else:
                    params[key] = val
            jobs.append(ExperimentJob(model_name=model_name, param_combination=params))
    return jobs


def combo_key(params: Dict[str, Any]) -> str:
    filtered = {k: v for k, v in params.items() if k != "seed"}
    return "_".join(f"{k}-{v}" for k, v in sorted(filtered.items())) or "default"


# ── Single job execution ──────────────────────────────────────────────────────

def run_job(job: ExperimentJob, job_index: int | None = None, total: int | None = None) -> None:
    params = job.param_combination
    seed   = params.get("seed", 0)
    key    = combo_key(params)

    export_path = os.path.join(EXPORT_ROOT, job.model_name, key)
    os.makedirs(export_path, exist_ok=True)
    filepath = os.path.join(export_path, f"results_seed-{seed}.pkl")

    prefix = f"[{job_index+1}/{total}] " if job_index is not None else ""

    if SKIP_EXISTING and os.path.exists(filepath):
        print(f"{prefix}Skipping (exists): {filepath}")
        return

    print(f"{prefix}{job.model_name} | {params}")

    config = ContextualSwitchingTaskConfig(experiment_to_run="figure")

    # Apply shared training overrides first.
    for k, v in TRAIN_OVERRIDES.items():
        setattr(config, k, v)

    # Apply condition-specific overrides (may override shared defaults).
    for k, v in CONDITIONS[job.model_name].items():
        setattr(config, k, v)

    config.env_seed = seed
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
    jobs  = generate_jobs()
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
