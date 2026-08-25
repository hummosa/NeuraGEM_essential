"""Run the flanker pipeline over seeds x noise levels.

Each seed is a synthetic subject: it gets its own Stage-1 pretraining, cached to disk,
and is then tested with the weights frozen. `arrow_noise_std` is a stimulus parameter, so
each noise level needs its own trained model per seed. All configurable state lives in
flanker_sweep_config.py.

Run locally (sequential):
    python flanker_sweep.py

Run on SLURM (one job per array element):
    #SBATCH --array=0-<N-1>
    python flanker_sweep.py

Loading results for analysis:
    from flanker_sweep import load_result, load_condition
    res = load_result(seed=0, variant='noise10')        # dict with train_logger, config
    results = load_condition('noise10')                 # all seeds present on disk
"""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from configs import FlankerTaskConfig, FlankerRandomTrialsConfig
from train_and_infer_functions import train_model
from flanker_analyses import sync_gating, mirror_to_model, reset_Z_uniform
from flanker_sweep_config import (
    SEEDS, N_PRETRAIN_TRIALS, N_TEST_TRIALS, P_CONGRUENT,
    GATING, Z_INIT_SCALE, PRETRAIN_OVERRIDES, TEST_OVERRIDES, VARIANTS,
    RUN_NAME, EXPORT_ROOT, SKIP_EXISTING,
)


# ── Which sweep run to read ───────────────────────────────────────────────────
#
# RUN_NAME (from flanker_sweep_config) is the single source of truth for which sweep every
# loader and figure script reads. Change it there to change what a fresh run writes; use
# `use_run(...)` or an entry point's --run flag to read a different one without editing
# the config. Reading the wrong run silently is easy and has bitten this project, so every
# entry point prints the run it used.

SWEEP_RUNS = {
    'sweep_noise_pt4k': 'The noise ladder with 4000 Stage-1 trials (was 2000). Same four '
                        'noise levels, 20 seeds, everything else fixed — so the pair with '
                        'sweep_noise isolates how much longer pretraining changes the '
                        'behavioural signatures.',
    'sweep_noise': 'The noise ladder: arrow_noise_std 1.3 / 1.0 / 0.7 / 0.4, everything '
                   'else fixed (SGD latent optimizer, temporal decay 0.3, steep spatial '
                   'gradient, p_congruent 0.5). Tests whether stimulus noise is what '
                   'breaks the post-error signatures.',
}


@contextlib.contextmanager
def use_run(name):
    """
    Temporarily read from a different sweep run. Restores the previous name on exit.

        with use_run('sweep_noise'):
            low_noise = load_condition('noise04')
    """
    global RUN_NAME
    previous = RUN_NAME
    RUN_NAME = name or previous
    try:
        yield RUN_NAME
    finally:
        RUN_NAME = previous


def describe_runs(root=None):
    """
    Every sweep on disk, read from the stored configs rather than from SWEEP_RUNS.

    Returns a DataFrame: the parameters that actually differ between runs, how many
    conditions and seeds are present, and the prose note if there is one. Notes go stale;
    the configs do not.
    """
    import glob
    import pandas as pd

    root = root or os.path.join(EXPORT_ROOT)
    keys = ['Z_optimizer', 'Z_lr', 'Z_decay', 'Z_decay_mode', 'arrow_noise_std',
            'temporal_decay_factor', 'n_trials', 'arrows_duration']
    rows = []
    for run in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        run_dir = os.path.join(root, run)
        conds = sorted(d for d in os.listdir(run_dir) if not d.startswith('pretrain'))
        if not conds:
            continue
        files = glob.glob(os.path.join(run_dir, conds[0], 'results_seed-*.pkl'))
        if not files:
            continue
        with open(files[0], 'rb') as fh:
            config = pickle.load(fh)['config']
        row = {'run': run, 'conditions': len(conds), 'seeds': len(files),
               'current': run == RUN_NAME}
        row.update({k: getattr(config, k, None) for k in keys})
        row['note'] = SWEEP_RUNS.get(run, '(not in SWEEP_RUNS)')
        rows.append(row)
    return pd.DataFrame(rows)


# ── Safe writes ───────────────────────────────────────────────────────────────

def _atomic_save(write_fn, path: str, suffix: str = '') -> None:
    """Write through a unique temp file, then rename into place.

    On a cluster, several array elements can share a seed and reach the pretraining
    step simultaneously. Renaming is atomic, so the worst case becomes duplicated work
    rather than a half-written checkpoint that every later job then fails to load.
    """
    tmp = f'{path}.tmp-{os.getpid()}{suffix}'
    write_fn(tmp)
    os.replace(tmp, path)


# ── Paths ─────────────────────────────────────────────────────────────────────

def pretrain_tag(variant: str) -> str:
    """Which pretrained model set a variant needs.

    Variants that only change the test stage share the baseline models. Variants that
    override Stage 1 (e.g. the spatial gradient) need their own, so they get their own
    cache folder keyed by the variant name.
    """
    spec = VARIANTS.get(variant, {})
    return variant if spec.get('pretrain_overrides') else 'shared'


def _pretrain_folder(tag: str) -> str:
    name = 'pretrained' if tag == 'shared' else f'pretrained__{tag}'
    folder = os.path.join(EXPORT_ROOT, RUN_NAME, name)
    os.makedirs(folder, exist_ok=True)
    return folder


def pretrain_path(seed: int, tag: str = 'shared') -> str:
    return os.path.join(_pretrain_folder(tag), f'model_seed-{seed}.pt')


def pretrain_curve_path(seed: int, tag: str = 'shared') -> str:
    """Per-trial training record, so the learning curve can be plotted across seeds
    without keeping the (large) training logger."""
    return os.path.join(_pretrain_folder(tag), f'curve_seed-{seed}.npz')


def result_path(seed: int, variant: str) -> str:
    """Result location: one folder per variant, one pickle per seed."""
    folder = os.path.join(EXPORT_ROOT, RUN_NAME, variant)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'results_seed-{seed}.pkl')


# ── Job definition ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentJob:
    seed: int
    variant: str


def generate_jobs() -> List[ExperimentJob]:
    """Seed-major ordering, so consecutive SLURM array elements reuse one pretrained model."""
    return [ExperimentJob(seed=s, variant=name)
            for s in range(SEEDS)
            for name in VARIANTS]


def pretrain_jobs() -> List[tuple]:
    """Unique (seed, tag) pairs — one job per pretrained model the sweep needs.

    Run this array first on a cluster so the test array finds every model already
    cached, instead of many jobs racing to train the same seed.
    """
    tags = sorted({pretrain_tag(v) for v in VARIANTS})
    return [(seed, tag) for tag in tags for seed in range(SEEDS)]


# ── Stage 1: pretraining (cached per seed) ────────────────────────────────────

def build_pretrain_config(seed: int, tag: str = 'shared') -> FlankerTaskConfig:
    config = FlankerTaskConfig(experiment_to_run='default')
    config.run_name   = f'{RUN_NAME}_pretrain_{tag}_seed-{seed}'
    config.env_seed   = seed
    # Use the setter rather than assigning blocked_phase_length: the config derives
    # n_training_contexts and no_of_blocks from n_pretrain_trials, and setting the length
    # alone leaves those stale — the model would train for a different number of blocks
    # than the length implies.
    config.set_n_pretrain_trials(N_PRETRAIN_TRIALS)
    config.pre_gating  = (GATING == 'pre')
    config.post_gating = (GATING == 'post')
    for key, value in PRETRAIN_OVERRIDES.items():
        setattr(config, key, value)
    # Variant Stage-1 overrides last, so they win over the shared defaults.
    for key, value in VARIANTS.get(tag, {}).get('pretrain_overrides', {}).items():
        setattr(config, key, value)
    return config


def ensure_pretrained(seed: int, tag: str = 'shared') -> None:
    """Train and cache the Stage-1 model for one seed, plus its learning curve.

    Pretraining is deterministic given the seed (train_model seeds torch and numpy,
    and the dataset draws from config.env_seed), so regenerating a missing curve
    reproduces the same model.
    """
    path, curve = pretrain_path(seed, tag), pretrain_curve_path(seed, tag)
    if SKIP_EXISTING and os.path.exists(path) and os.path.exists(curve):
        return

    print(f'  Pretraining seed {seed} [{tag}] ({N_PRETRAIN_TRIALS} trials)...')
    config = build_pretrain_config(seed, tag)
    logger, model, config, _ = train_model(
        config, seed=seed, save_models=False, load_models=False, run_test_phase=False,
    )
    _atomic_save(lambda tmp: torch.save(model, tmp), path)

    from flanker_analyses import extract_trials
    trials = extract_trials(logger, config)
    _atomic_save(lambda tmp: np.savez_compressed(
        tmp,
        correct=trials['correct_at_decision'],
        rt=trials['rt_interp'],
        # Stage 1 hlcids are the congruency flag (1.0 congruent / 0.0 incongruent)
        congruent=trials['trial_type'],
        correct_by_timestep=trials['correct'],
    ), curve, suffix='.npz')
    print(f'  Cached pretrained model -> {path}')


def load_pretrain_curve(seed: int, tag: str = 'shared'):
    """Per-trial training record for one seed, or None if not saved yet."""
    path = pretrain_curve_path(seed, tag)
    return dict(np.load(path)) if os.path.exists(path) else None


def load_pretrained(seed: int, tag: str = 'shared'):
    """Load a fresh copy of the seed's pretrained model.

    Reloaded per job rather than reused in-process, so a test session can never
    inherit Z state or config edits from the previous one. weights_only=False is
    required: these are whole pickled models, and torch >= 2.6 defaults it to True.
    """
    return torch.load(pretrain_path(seed, tag), weights_only=False)


# ── Stage 2: one test session ─────────────────────────────────────────────────

def run_job(job: ExperimentJob, job_index: Optional[int] = None,
            total: Optional[int] = None) -> None:
    prefix = f'[{job_index + 1}/{total}] ' if job_index is not None else ''
    filepath = result_path(job.seed, job.variant)

    # Before the skip check, so a resumed run still backfills any missing pretraining
    # artifact (e.g. a learning curve added after the models were first cached).
    tag = pretrain_tag(job.variant)
    ensure_pretrained(job.seed, tag)

    if SKIP_EXISTING and os.path.exists(filepath):
        print(f'{prefix}Skipping (exists): {filepath}')
        return

    spec           = VARIANTS.get(job.variant, {})
    overrides      = spec.get('overrides', {})
    stim_overrides = spec.get('pretrain_overrides', {})
    print(f'{prefix}seed={job.seed}  variant={job.variant}'
          + (f'  {overrides}' if overrides else ''))
    model = load_pretrained(job.seed, tag)

    test_config = FlankerRandomTrialsConfig(experiment_to_run='default')
    test_config.run_name    = f'{RUN_NAME}_{job.variant}_seed-{job.seed}'
    test_config.env_seed    = job.seed
    test_config.p_congruent = P_CONGRUENT
    test_config.set_n_trials(N_TEST_TRIALS)
    for key, value in TEST_OVERRIDES.items():
        setattr(test_config, key, value)
    # Stage-1 overrides are stimulus/task parameters shared by both stages, so they
    # must apply here too. p_corr_by_distance happens to be unused at test, but
    # something like arrow_noise_std would otherwise train and test at different
    # values without any warning.
    for key, value in stim_overrides.items():
        setattr(test_config, key, value)
    # Test-stage overrides last, so they win.
    for key, value in overrides.items():
        setattr(test_config, key, value)

    # model.config is the pretraining config the model was built with; the gating
    # choice lives there, and a freshly constructed stage config would revert it.
    sync_gating(test_config, model.config)
    mirror_to_model(model, test_config)
    reset_Z_uniform(model, scale=Z_INIT_SCALE, seed=job.seed)

    torch.manual_seed(job.seed)
    np.random.seed(job.seed)

    train_logger, _, test_config, _ = train_model(
        test_config, seed=job.seed, save_models=False, load_models=False,
        pretrained_model=model, run_test_phase=False,
    )

    with open(filepath, 'wb') as fh:
        pickle.dump({'train_logger': train_logger,
                     'config': test_config,
                     'seed': job.seed,
                     'p_congruent': P_CONGRUENT,
                     'variant': job.variant,
                     'overrides': dict(overrides),
                     'pretrain_overrides': dict(stim_overrides)}, fh)
    print(f'  Saved -> {filepath}')


# ── Result loading (for analysis scripts) ─────────────────────────────────────

def load_result(seed: int, variant: str) -> Optional[Dict[str, Any]]:
    """Load one (seed, variant) result, or None if it has not been run."""
    path = result_path(seed, variant)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def load_condition(variant: str, seeds: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Load every available seed for one variant, reporting what is missing."""
    seeds = list(range(SEEDS)) if seeds is None else seeds
    label = variant
    results, missing = [], []
    for seed in seeds:
        res = load_result(seed, variant)
        (results.append(res) if res is not None else missing.append(seed))
    if missing:
        print(f'  [{label}] missing seeds: {missing}')
    print(f'  [{label}] loaded {len(results)}/{len(seeds)} seeds')
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main(mode: str = 'full') -> None:
    task_id_str = os.environ.get('SLURM_ARRAY_TASK_ID')

    if mode == 'pretrain':
        jobs = pretrain_jobs()
        print(f'Pretraining jobs: {len(jobs)}  '
              f'({SEEDS} seeds x {len(jobs)//max(SEEDS,1)} model sets)')
        selected = (list(enumerate(jobs)) if task_id_str is None
                    else [(int(task_id_str), jobs[int(task_id_str)])])
        for idx, (seed, tag) in selected:
            print(f'[{idx+1}/{len(jobs)}] pretrain seed={seed} tag={tag}')
            ensure_pretrained(seed, tag)
        return

    jobs  = generate_jobs()
    total = len(jobs)
    print(f'Total jobs: {total}  ({SEEDS} seeds x {len(VARIANTS)} variants)')

    if task_id_str is None:
        print('No SLURM_ARRAY_TASK_ID — running all jobs sequentially.')
        for idx, job in enumerate(jobs):
            run_job(job, job_index=idx, total=total)
    else:
        task_id = int(task_id_str)
        if task_id < 0 or task_id >= total:
            raise ValueError(f'Task id {task_id} out of range [0, {total-1}].')
        run_job(jobs[task_id], job_index=task_id, total=total)


if __name__ == '__main__':
    main('pretrain' if 'pretrain' in sys.argv[1:] else 'full')
