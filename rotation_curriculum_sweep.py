"""Run the three-stage cued-context curriculum on the rotating-targets task.

All configurable state lives in rotation_curriculum_config.py — edit that file to change what
is run.

One job = one (Z_lr, noise_std, seed) cell — one "individual" — and it runs the whole stage tree
in memory, at the SAME Z_lr from start to finish:

    S1  uncued, self-Z, WU+LU @ Z_lr                          1 run
     |__ for cue_mode in CUE_MODES:
           S2  from deepcopy(model_S1)                        1 run each
            |__ S3        from deepcopy(model_S2), W frozen, same Z_lr        1 run each
            |__ S3_pinned from deepcopy(model_S2), W frozen, Z pinned         1 run each
                                                              (sanity only, S3_PINNED_Z_SANITY)

S1 does not depend on cue_mode, so the cued and control arms fork from *identical* S1 weights.
S3 continues each individual at the Z_lr it trained and was cued with — there is no separate
"test" Z_lr any more, which is the whole point: an individual's Z_lr is fixed all along, and S3
is where it is recovered from behaviour, not a grid of mismatched values tried against one
model's weights. S3_pinned is the one exception, a small bounded fork off the same S2 checkpoint
that pins Z instead of carrying it forward, kept only as the "does anything besides Z carry
context" sanity check.

Run locally (sequential):
    python rotation_curriculum_sweep.py

Run on SLURM (one job per array element):
    ./submit_job.sh <N-1> curriculum
"""

from __future__ import annotations

import os

# Must precede the torch import (via configs -> functions_and_utils). batch_size=1 with a
# 64-unit LSTM is far too small to benefit from intra-op threading, and on a busy shared node
# the threads contend and stall.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import copy
import pickle
import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List

import numpy as np
import torch

from train_and_infer_functions import train_model
from rotation_slips_perseveration_sweep import compact_logger
from rotation_curriculum_config import (
    CUE_MODES, PASSIVE_LENGTH, PILOT, S1_LENGTH, S2_BLOCK_SCALE, S2_LENGTH, S3_LENGTH,
    S3_PINNED_Z_SANITY, S3_Z_INIT, SKIP_EXISTING, STAGE_SEED_OFFSETS, Z_LR,
    active_noise, active_seeds, make_base_config, result_path, z_decay_for,
)


# ── Job enumeration ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CurriculumJob:
    z_lr: Any
    noise_std: float
    seed: int


def generate_jobs() -> List[CurriculumJob]:
    """One job per (Z_lr, noise, seed) — one individual. Cue and stage axes run inside a job."""
    return [CurriculumJob(z_lr=z, noise_std=n, seed=s)
            for z, n, s in product(Z_LR, active_noise(), range(active_seeds()))]


# ── Stage configs ─────────────────────────────────────────────────────────────

def _apply_zlr(cfg, z_lr: Any) -> None:
    """Latent settings for the one Z_lr this run carries through every stage."""
    if z_lr == 'RNN':
        # No latent inference at all. Z stays at its zero init, so softmax gives a constant
        # uniform gate and the weights learn without any context channel — which is what makes
        # this the strongest impairment arm when S2 suddenly hands it a varying gate.
        cfg.no_of_steps_in_latent_space = 0
    else:
        cfg.no_of_steps_in_latent_space = 1
        cfg.Z_lr    = z_lr
        cfg.Z_decay = z_decay_for(z_lr)


def _round_block_size(block_size: int, n_colors: int) -> int:
    """Nearest block size that is a whole number of mini-blocks.

    The dataset derives its mini-block count as block_size // (n_colors * 2), so a size that is
    not a multiple of that silently truncates part of the block.
    """
    unit = n_colors * 2
    return max(unit, int(round(block_size / unit)) * unit)


def make_stage_configs(z_lr: Any, noise_std: float) -> Dict[str, Any]:
    """Base config plus the S1 config. S2/S3 configs are derived per branch at run time."""
    base = make_base_config(noise_std=noise_std)
    _apply_zlr(base, z_lr)

    cfg_s1 = copy.deepcopy(base)
    cfg_s1.blocked_phase_length      = S1_LENGTH
    cfg_s1.add_passive_learning_phase = True
    cfg_s1.passive_phase_length      = PASSIVE_LENGTH
    cfg_s1.what_latent_to_use        = 'self'
    cfg_s1.no_of_steps_in_weight_space = 1
    return dict(base=base, S1=cfg_s1)


def make_s2_config(base, z_lr: Any, cue_mode: str):
    cfg = copy.deepcopy(base)
    cfg.blocked_phase_length       = S2_LENGTH
    cfg.add_passive_learning_phase = False
    cfg.no_of_steps_in_weight_space = 1
    cfg.block_size = _round_block_size(cfg.block_size * S2_BLOCK_SCALE, cfg.n_colors)

    if cue_mode == 'oracle_z':
        # The cue: the true rotation is injected as a one-hot into Z, which multiplicatively
        # gates the hidden state. LU off — Z is given, never fitted, so the weights are forced to
        # read the gate rather than treat Z as another free parameter.
        cfg.what_latent_to_use          = 'context_ids'
        cfg.no_of_steps_in_latent_space = 0
    elif cue_mode == 'none':
        # Control: S2 continues exactly as S1, so the cued arm is compared against the same
        # amount of extra training rather than against nothing.
        cfg.what_latent_to_use = 'self'
        _apply_zlr(cfg, z_lr)
    else:
        raise ValueError(f"Unknown cue_mode {cue_mode!r}; choose from {CUE_MODES}.")
    return cfg


def make_s3_config(base, z_lr: Any, pin_z: bool = False):
    """S3: uncued again, weights frozen. Carries the SAME z_lr the individual trained/was cued
    with, unless `pin_z` — the one sanity-check exception, which pins Z instead (LU off)."""
    cfg = copy.deepcopy(base)
    cfg.blocked_phase_length       = S3_LENGTH
    cfg.add_passive_learning_phase = False
    cfg.what_latent_to_use         = 'self'
    # The whole design: with the weights frozen, Z is the only adaptive variable, so recovery
    # speed after a switch can only be a function of Z_lr.
    cfg.no_of_steps_in_weight_space = 0
    if pin_z:
        cfg.no_of_steps_in_latent_space = 0     # sanity branch: Z pinned, not part of the Z_LR axis
    else:
        _apply_zlr(cfg, z_lr)
    return cfg


# ── Stage-boundary bookkeeping ────────────────────────────────────────────────

def _last_block_rotation_rad(logger) -> float:
    """Rotation (radians) of the final block in a logger's context_ids stream."""
    ll = np.concatenate([np.asarray(e).reshape(-1) for e in logger.context_ids])
    return float(ll[-1])


def _init_Z_for_s3(model, cfg, last_rot_rad: float) -> None:
    """Put Z somewhere in-distribution before S3 hands control back to self-inference.

    S2's oracle gate is softmax(one_hot), i.e. [0.731, 0.269] at two rotations — the weights
    never saw the uniform [0.5, 0.5] that a fresh Z would produce. 'last_cued' therefore sets the
    pre-softmax logits to the one-hot of the last S2 block's rotation, which reproduces a gate
    the weights have actually seen. Under 'random_no_repeat' that rotation is uncorrelated with
    S3's first block, so about half the runs start in the wrong context and the first switch is a
    real perseveration test.

    The uncued control arm gets the identical rule even though it was never cued, so the two arms
    enter S3 matched.
    """
    if S3_Z_INIT == 'zeros':
        return
    z = torch.zeros_like(model.Z)
    if S3_Z_INIT == 'uniform':
        z.fill_(1.0 / model.Z_dim)
    elif S3_Z_INIT == 'last_cued':
        rots = np.deg2rad(np.asarray(cfg.train_rotations, dtype=float))
        d = np.abs(np.arctan2(np.sin(last_rot_rad - rots), np.cos(last_rot_rad - rots)))
        z[..., int(np.argmin(d))] = 1.0
    else:
        raise ValueError(f"Unknown S3_Z_INIT {S3_Z_INIT!r}; "
                         "choose 'last_cued', 'uniform' or 'zeros'.")
    model.set_Z(z)


def _hand_over(model, cfg) -> None:
    """Point a carried-over model at its new stage config and rebuild the Z optimizer.

    Both halves are load-bearing and neither happens on its own:
      - train_model never touches model.config, and model.config is a live reference used by the
        LU path and by _build_Z_optimizer.
      - _rebuild_Z_optimizer only fires when Z's *shape* changes, so a new Z_lr would otherwise
        be ignored and the old Adam moments would keep driving the step size. This is the
        reference implementation's explicit `m.Z_optimizer = m._build_Z_optimizer()` line.
    """
    model.config = cfg
    model._rebuild_Z_optimizer()


def _run_stage(cfg, model, seed: int, stage: str, label: str):
    """One stage. Returns (compacted logger, model); the model is carried forward.

    The logger is compacted immediately rather than at the end: a tree holds 17 of them, and the
    raw form is one tiny (1, 1, D) array per field per timestep, whose per-array overhead
    dominates the numbers by an order of magnitude.
    """
    print(f'  [{label}]  blocks={int(cfg.blocked_phase_length / cfg.block_size)} '
          f'latent={cfg.what_latent_to_use} LU={cfg.no_of_steps_in_latent_space} '
          f'WU={cfg.no_of_steps_in_weight_space} Z_lr={cfg.Z_lr:g}', flush=True)
    logger, model, _, _ = train_model(
        cfg, seed=seed + STAGE_SEED_OFFSETS[stage], run_test_phase=False,
        save_models=False, load_models=False, pretrained_model=model,
    )
    return compact_logger(logger), model


# ── One curriculum tree ───────────────────────────────────────────────────────

def run_tree(job: CurriculumJob, job_index: int | None = None, total: int | None = None) -> None:
    prefix = f'[{job_index + 1}/{total}] ' if job_index is not None else ''
    out = result_path(job.z_lr, job.noise_std, job.seed)

    if SKIP_EXISTING and out.exists():
        print(f'{prefix}Skipping (exists): {out}')
        return

    print(f'{prefix}Z_lr={job.z_lr} noise={job.noise_std} seed={job.seed}', flush=True)
    t0 = time.time()

    cfgs = make_stage_configs(job.z_lr, job.noise_std)

    # ── S1: uncued, from scratch ──────────────────────────────────────────────
    torch.manual_seed(job.seed)
    np.random.seed(job.seed)
    logger_s1, model_s1 = _run_stage(cfgs['S1'], None, job.seed, 'S1', 'S1 uncued')

    payload: Dict[str, Any] = {
        'meta': dict(z_lr=job.z_lr, noise_std=job.noise_std, seed=job.seed,
                     s1_length=S1_LENGTH, s2_length=S2_LENGTH, s3_length=S3_LENGTH,
                     s2_block_scale=S2_BLOCK_SCALE, s3_z_init=S3_Z_INIT,
                     cue_modes=list(CUE_MODES), s3_pinned_z_sanity=S3_PINNED_Z_SANITY),
        'S1': logger_s1,
        'S2': {}, 'S3': {}, 'S3_pinned': {},
        'configs': {'S1': cfgs['S1'], 'S2': {}, 'S3': {}, 'S3_pinned': {}},
    }

    for cue_mode in CUE_MODES:
        # ── S2: cued (or the matched uncued control) ──────────────────────────
        cfg_s2   = make_s2_config(cfgs['base'], job.z_lr, cue_mode)
        model_s2 = copy.deepcopy(model_s1)
        _hand_over(model_s2, cfg_s2)
        logger_s2, model_s2 = _run_stage(cfg_s2, model_s2, job.seed, 'S2', f'S2 {cue_mode}')
        payload['S2'][cue_mode] = logger_s2
        payload['configs']['S2'][cue_mode] = cfg_s2
        last_rot = _last_block_rotation_rad(logger_s2)

        # ── S3: uncued again, weights frozen, SAME Z_lr as S1/S2 ──────────────
        cfg_s3   = make_s3_config(cfgs['base'], job.z_lr)
        model_s3 = copy.deepcopy(model_s2)
        model_s3.config = cfg_s3            # set_Z reads it if it has to reallocate
        _init_Z_for_s3(model_s3, cfg_s3, last_rot)
        _hand_over(model_s3, cfg_s3)        # rebuild the optimizer against the final Z
        logger_s3, _ = _run_stage(cfg_s3, model_s3, job.seed, 'S3', f'S3 {cue_mode} z_lr={job.z_lr}')
        payload['S3'][cue_mode] = logger_s3
        payload['configs']['S3'][cue_mode] = cfg_s3

        # ── S3_pinned: same fork point, Z pinned instead — the sanity check ───
        if S3_PINNED_Z_SANITY:
            cfg_s3p   = make_s3_config(cfgs['base'], job.z_lr, pin_z=True)
            model_s3p = copy.deepcopy(model_s2)
            model_s3p.config = cfg_s3p
            _init_Z_for_s3(model_s3p, cfg_s3p, last_rot)
            _hand_over(model_s3p, cfg_s3p)
            logger_s3p, _ = _run_stage(cfg_s3p, model_s3p, job.seed, 'S3',
                                       f'S3pinned {cue_mode}')
            payload['S3_pinned'][cue_mode] = logger_s3p
            payload['configs']['S3_pinned'][cue_mode] = cfg_s3p

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        pickle.dump(payload, f)
    print(f'  Saved -> {out}  ({out.stat().st_size / 1e6:.1f} MB, '
          f'{(time.time() - t0) / 60:.1f} min)', flush=True)
    return payload


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    jobs  = generate_jobs()
    total = len(jobs)
    # Per cue_mode: S2 + S3, plus S3_pinned when the sanity branch is on.
    n_runs = 1 + len(CUE_MODES) * (2 + (1 if S3_PINNED_Z_SANITY else 0))
    print(f"{'PILOT: ' if PILOT else ''}Total jobs: {total}  "
          f"({len(Z_LR)} Z_lr values x {len(active_noise())} noise x "
          f"{active_seeds()} seeds), {n_runs} training runs each")
    print(f'Export root: {result_path(Z_LR[0], active_noise()[0], 0).parents[2]}')

    task_id_str = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id_str is None:
        print('No SLURM_ARRAY_TASK_ID — running all jobs sequentially.')
        for idx, job in enumerate(jobs):
            payload = run_tree(job, job_index=idx, total=total)
    else:
        task_id = int(task_id_str)
        if task_id < 0 or task_id >= total:
            raise ValueError(f'Task id {task_id} out of range [0, {total - 1}].')
        payload = run_tree(jobs[task_id], job_index=task_id, total=total)
    
    return payload  


if __name__ == '__main__':
    payload = main()
