"""Sweep RNN vs NeuraGEM vs Oracle on the rotating-targets task, crossed with observation noise.

All configurable state (conditions, noise levels, seeds, base config, export paths) lives in
rotation_slips_perseveration_config.py — edit that file to change what is run.

Only Phase 2 is trained (run_test_phase=False): this experiment is about perseveration and
context slips *during* blocked training, not about generalization to novel rotations, which
run_rotating_targets_comparison.py already covers.

Run locally (sequential):
    python rotation_slips_perseveration_sweep.py

Run on SLURM (one job per array element):
    ./submit_job.sh <N-1> rotation_slips
"""

from __future__ import annotations

import os

# Must precede the torch import (via configs -> functions_and_utils). batch_size=1 with a
# 64-unit LSTM is far too small to benefit from intra-op threading, and on a busy shared node
# the threads contend and stall. Measured single-threaded: ~122 it/s.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import pickle
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List

import numpy as np
import torch

from train_and_infer_functions import train_model
from rotation_slips_perseveration_config import (
    CONTEXT_OUTPUT_ENCODING, EXPORT_ROOT, PILOT, PILOT_INCLUDE_NO_HEAD_CONTROL,
    NOISE_LEVELS, SKIP_EXISTING, active_conditions, active_seeds, make_base_config,
)


CONDITIONS = active_conditions()
N_SEEDS    = active_seeds()

# 'context_encoding' is only a grid axis in pilot mode, where the no-head control arm answers
# "does the belief head change the primary task?". In the full sweep every run has the head.
_ENCODINGS = ([CONTEXT_OUTPUT_ENCODING, None]
              if (PILOT and PILOT_INCLUDE_NO_HEAD_CONTROL) else [CONTEXT_OUTPUT_ENCODING])

PARAM_GRIDS: Dict[str, Any] = {
    name: {
        "seed":             list(range(N_SEEDS)),
        "noise_std":        list(NOISE_LEVELS),
        "context_encoding": list(_ENCODINGS),
    }
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
    """Directory name for one parameter cell. Everything except the seed, which names the file."""
    filtered = {k: v for k, v in params.items() if k != "seed"}
    return "_".join(f"{k}-{v}" for k, v in sorted(filtered.items())) or "default"


# ── Logger compaction ─────────────────────────────────────────────────────────

# The logger appends one tiny (1, 1, D) array per timestep per field. At ~20k timesteps the
# per-array pickle overhead dominates the actual numbers, so a raw pickle runs 15-20 MB and the
# full sweep would be several GB. Collapsing each field to a single concatenated float32 array
# wrapped in a one-element list keeps every consumer working unchanged — they all do
# np.concatenate(field, axis=0).reshape(-1, D) (see rotating_targets_analysis.flatten_logger).
_KEEP_FIELDS = ('inputs', 'predicted_outputs', 'context_ids', 'hlcids',
                'latent_values', 'training_losses')
_DROP_FIELDS = ('training_batches', 'training_losses_before_latent_optimization',
                'hidden_states', 'gradients_corrections', 'latent_gradients',
                'latent_updating_losses', 'latent_updating_latents',
                'latent_updating_combined_inputs', 'latent_updating_outputs',
                'latent_updating_grad_model_outputs', 'input_attention_weights',
                'gradients_max_entropy', 'prediction_losses',
                'testing_batches', 'testing_losses')


def compact_logger(logger):
    """Shrink a logger in place for storage. `phases` and `others` are left untouched.

    Surviving fields are _KEEP_FIELDS; everything in _DROP_FIELDS becomes []. Panels that read a
    dropped field (e.g. plot_logger_panels' 'gradients', which wants gradients_corrections) will
    come out empty on a compacted logger — that is expected, not a bug.
    """
    # rotating_targets_analysis.get_phase3a_range falls back to len(logger.inputs) as a timestep
    # count, which compaction reduces to 1. Phase 3 is not run by this sweep, so the path is
    # never reached; fail loudly rather than silently return a nonsense range if that changes.
    assert not any(name in ('Inference only', 'No inference nor learning')
                   for name, _ in logger.phases), (
        'compact_logger breaks len(logger.inputs) as a timestep count, which '
        'get_phase3a_range relies on. Do not compact a logger that has Phase 3 data.')
    for field in _KEEP_FIELDS:
        entries = getattr(logger, field, None)
        if entries:
            setattr(logger, field, [np.concatenate(entries, axis=0).astype(np.float32)])
    for field in _DROP_FIELDS:
        if hasattr(logger, field):
            setattr(logger, field, [])
    # logger.config holds a full Config; it is pickled separately under 'config'.
    if hasattr(logger, 'config'):
        del logger.config
    return logger


# ── Single job execution ──────────────────────────────────────────────────────

def run_job(job: ExperimentJob, job_index: int | None = None, total: int | None = None) -> None:
    params   = job.param_combination
    seed     = params.get("seed", 0)
    noise    = params["noise_std"]
    encoding = params["context_encoding"]
    key      = combo_key(params)

    export_path = os.path.join(EXPORT_ROOT, job.model_name, key)
    os.makedirs(export_path, exist_ok=True)
    filepath = os.path.join(export_path, f"results_seed-{seed}.pkl")

    prefix = f"[{job_index+1}/{total}] " if job_index is not None else ""

    if SKIP_EXISTING and os.path.exists(filepath):
        print(f"{prefix}Skipping (exists): {filepath}")
        return

    print(f"{prefix}{job.model_name} | {params}", flush=True)

    config = make_base_config(noise_std=noise, context_output_encoding=encoding)

    # Condition-specific overrides last, so they win over the shared defaults.
    for k, v in CONDITIONS[job.model_name].items():
        setattr(config, k, v)

    config.env_seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # save_models defaults to True, and the model filename does not include the condition, so
    # conditions would overwrite one another. run_test_phase=False: Phase 2 only.
    train_logger, model, config, _ = train_model(
        config, seed=seed, run_test_phase=False,
        save_models=False, load_models=False,
    )

    with open(filepath, "wb") as f:
        pickle.dump({"train_logger": compact_logger(train_logger), "config": config}, f)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"  Saved → {filepath}  ({size_mb:.1f} MB)", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    jobs  = generate_jobs()
    total = len(jobs)
    print(f"{'PILOT: ' if PILOT else ''}Total jobs: {total}  "
          f"({len(CONDITIONS)} conditions x {len(NOISE_LEVELS)} noise x "
          f"{len(_ENCODINGS)} encodings x {N_SEEDS} seeds)")
    print(f"Export root: {EXPORT_ROOT}")

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
