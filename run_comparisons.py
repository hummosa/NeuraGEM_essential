"""
run_comparisons.py — Compare RNN vs NeuraGEM variants on the same pred-mean subplot.

Comparisons
───────────
  RNN               : no_of_steps_in_latent_space = 0
  NeuraGEM          : no_of_steps_in_latent_space = 1  (standard defaults)
  NeuraGEM LU-first : update_latent_before_weights=True, latent_aggregation_op='none'
  NeuraGEM slow     : Z_lr=0.05,  Z_decay=1e-5   (too slow)
  NeuraGEM fast     : Z_lr=0.7,   Z_decay=8e-4   (too fast)

Shared sweep params (edit below) are applied to every condition.
"""

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

#%%
import numpy as np
import matplotlib.pyplot as plt

import plot_style
plot_style.set_plot_style()

from functions_and_utils import *
from configs import *
from train_and_infer_functions import *


# ── Shared params ──────────────────────────────────────────────────────────────
# These are applied to every comparison condition below.

CORRELATED_NOISE       = True
NOISE_CORRELATION_TAU  = 1       # 1 ≈ white noise; larger → smoother correlations
BLOCKED_PHASE_LENGTH   = 5000    # length of training (blocked) phase
DEFAULT_STD            = 0.1     # observation noise


# ── Comparison conditions ──────────────────────────────────────────────────────
# Each entry is (label, dict-of-overrides).  Overrides are applied on top of
# the shared params above and the ContextualSwitchingTaskConfig defaults.

COMPARISONS = [
    (
        'RNN',
        dict(no_of_steps_in_latent_space=0),
    ),
    (
        'NeuraGEM',
        dict(no_of_steps_in_latent_space=1),
    ),
    (
        'NeuraGEM LU-first',
        dict(
            no_of_steps_in_latent_space=1,
            update_latent_before_weights=True,
            # latent_aggregation_op='none',
        ),
    ),
    (
        'NeuraGEM slow (Z_lr=0.2)',
        dict(
            no_of_steps_in_latent_space=1,
            Z_lr=0.2,
            Z_decay=5e-5,
        ),
    ),
    (
        'NeuraGEM fast (Z_lr=0.7)',
        dict(
            no_of_steps_in_latent_space=1,
            Z_lr=0.7,
            Z_decay=8e-4,
        ),
    ),
]


# ── Helper: build a fresh config with shared + per-condition overrides ─────────

def make_config(overrides: dict):
    cfg = ContextualSwitchingTaskConfig(experiment_to_run='figure')
    cfg.default_std            = DEFAULT_STD
    cfg.save_model             = False
    cfg.correlated_noise       = CORRELATED_NOISE
    cfg.noise_correlation_tau  = NOISE_CORRELATION_TAU
    cfg.blocked_phase_length   = BLOCKED_PHASE_LENGTH
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


PHASES_TO_INCLUDE  = 'Learning and inference'  # None = all phases
LAST_TS_IN_BLOCK   = 15    # how many tail timesteps per block to average


# ── Run all conditions ─────────────────────────────────────────────────────────

results = []   # list of (label, block_means, block_sems)

for label, overrides in COMPARISONS:
    print(f'\n=== {label} ===')
    cfg = make_config(overrides)
    logger, model, cfg, _ = train_model(cfg, seed=cfg.env_seed, save_models=False, load_models=False)

    panel_order = ['behavior', 'latent_2d', 'corrects']
    fig = plot_logger_panels(logger, cfg, panel_order, x2=None, annotate_phases='behavior', width=5)
    fig.suptitle(label, fontsize=9, fontweight='bold', y=1.01)
    plt.show()

    block_means, block_sems = extract_block_corrects(
        logger,
        last_ts_in_a_block=LAST_TS_IN_BLOCK,
        phases_to_include=PHASES_TO_INCLUDE,
    )
    results.append((label, block_means, block_sems))
    print(f'  done — {len(block_means)} blocks')


# ── Plot ───────────────────────────────────────────────────────────────────────
#%%
AGGREGATE_BLOCKS   = 30     # group this many consecutive blocks into one point (None = per-block)

fig, ax = plt.subplots(figsize=(7, 3), dpi=120)

n_agg = int(AGGREGATE_BLOCKS) if AGGREGATE_BLOCKS is not None else 1

for label, block_means, block_sems in results:
    if n_agg > 1:
        n_groups = len(block_means) // n_agg
        x_vals   = [g + 0.5 for g in range(n_groups)]
        grp_means = [block_means[g*n_agg:(g+1)*n_agg].mean() for g in range(n_groups)]
        grp_sems  = [block_means[g*n_agg:(g+1)*n_agg].std() / np.sqrt(n_agg) for g in range(n_groups)]
        ax.errorbar(x_vals, grp_means, yerr=grp_sems,
                    fmt='-o', capsize=3, linewidth=1.2, markersize=4, label=label, alpha=0.85)
    else:
        ax.errorbar(range(len(block_means)), block_means, yerr=block_sems,
                    fmt='-o', capsize=2, linewidth=1.2, markersize=3, label=label, alpha=0.85)

x_label = f'Block group (×{n_agg})' if n_agg > 1 else 'Block'
ax.set_xlabel(x_label)
ax.set_ylabel(f'|pred − mean|  (tail {LAST_TS_IN_BLOCK} ts)')
ax.set_title(
    f'correlated_noise={CORRELATED_NOISE}, τ={NOISE_CORRELATION_TAU}, '
    f'blocked_phase={BLOCKED_PHASE_LENGTH}'
)
ax.legend(fontsize=7, loc='upper right')
plt.tight_layout()
plt.show()
