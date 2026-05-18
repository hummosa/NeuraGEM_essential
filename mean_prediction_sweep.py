"""Sweep RNN vs NeuraGEM conditions on the mean-prediction task over multiple seeds.

All configurable state (conditions, seeds, overrides, export paths) lives in
mean_prediction_sweep_config.py — edit that file to change what is run.

Run locally (sequential):
    python mean_prediction_sweep.py

Run on SLURM (one job per array element):
    #SBATCH --array=0-<N-1>
    python mean_prediction_sweep.py
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List

import numpy as np
import torch

from configs import MeanPredictionConfig
from train_and_infer_functions import train_model
from mean_prediction_sweep_config import (
    CONDITIONS, DEFAULT_SEEDS, TRAIN_OVERRIDES, SKIP_EXISTING, RUN_NAME, EXPORT_ROOT,
)


PARAM_GRIDS: Dict[str, Any] = {
    name: {"seed": list(range(DEFAULT_SEEDS))}
    for name in CONDITIONS
}


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

    config = MeanPredictionConfig(experiment_to_run="figure")

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
