"""
run_rotating_targets_comparison.py — Adaptation speed comparison on the rotating-targets task.

Trains three model conditions and compares how quickly each adapts after an
unsignalled block switch in the last 1/3 of Phase 2 training:

  NeuraGEM  : self-inferred Z updated each batch via LU
  RNN       : Z frozen at initialisation — weights adapt but context does not
  Oracle Z  : true rotation angle injected as task ID (upper-bound ceiling)

After Phase 2, Phase 3a immediately runs with novel rotations (45°, 135°) and
weights kept plastic (test_no_of_steps_in_weight_space=1), measuring how quickly
each model's context inference generalizes to unseen angles.

Produces:
  P1 — Euclidean error per trial position in first mini-block after switch
  P2 — Normalized state error per trial position  (0=correct, 0.5=chance, 1=wrong)
  P3 — Per-mini-block learning curve after switch (trained rotations)
  P4 — Z-trajectory (NeuraGEM only): displacement from old → new Z centroid
  P5 — Novel rotation normalized error per trial position
  P6 — Novel rotation mini-block learning curve
  P7 — Arena plots: qualitative predictions vs observations (last 2 trained blocks)
"""

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

import plot_style
plot_style.set_plot_style()

import numpy as np
import matplotlib.pyplot as plt

from functions_and_utils import *
from configs import RotatingTargetsConfig
from train_and_infer_functions import train_model
from functions_and_utils import plot_logger_panels
from run_2D_predicitve_task import plot_arena_trials
from rotating_targets_analysis import (
    analyze_block_switch_adaptation,
    analyze_novel_rotation_test,
    analyze_rotation_sweep,
    analyze_z_trajectory,
    get_phase3a_range,
    plot_adaptation_comparison,
    plot_miniblock_norm_comparison,
    plot_novel_rotation_comparison,
    plot_rotation_angle_sweep,
    plot_z_by_rotation,
)

# ── Experiment parameters ──────────────────────────────────────────────────────

BLOCKED_PHASE_LENGTH = 3000   # longer → more block switches in last 1/3
PASSIVE_PHASE_LENGTH = 500
NOVEL_ROTATIONS      = list(range(0, 360, 15))  # all 24 angles in 15° steps
NOVEL_N_BLOCKS       = 50              # ~2 samples per angle; use block_range to compare early vs late
SEED                 = 42
N_MINIBLOCKS_TRACK   = 6
Z_WINDOW             = 40

'''
test all rotations in 15 deg increments
Parametrize Z dims and see if it learns a more continuous representation of rotation space (e.g. circular manifold) 
Parametrize Z lr to see the effects of faster vs slower context inference 
test contexts that are not rotations, but randomly sampled shield targets for each colors.

'''
# ── Shared config factory ──────────────────────────────────────────────────────

def make_base_config():
    cfg = RotatingTargetsConfig()
    cfg.train_rotations = [0.0, 70.0, 115.0]  
    # cfg.train_rotations = [0.0, 90.0]  

    cfg.add_passive_learning_phase = True
    cfg.passive_phase_length  = PASSIVE_PHASE_LENGTH
    cfg.blocked_phase_length  = BLOCKED_PHASE_LENGTH

    cfg.n_miniblocks_per_state_block = 7  # USING DOUBLE BECAUSE RNN UNABLE TO SWITCH FULLY
    # The paper uses 6-8 miniblocks per state block
    # total timesteps= 7*n_colors*2 (for cue and outcome) = 60 per block
    cfg.block_size = cfg.n_miniblocks_per_state_block * cfg.n_colors * 2 # 120

    cfg.WU_lr    = 0.001
    cfg.pass_previous_latent = True
    
    cfg.pre_gating  = False
    cfg.post_gating = True

    if cfg.pre_gating:
        cfg.LU_lr = 0.1
        cfg.l2_loss = 0.000_0011
    else:
        cfg.LU_lr    = 0.5
        cfg.l2_loss  = 0.000_1


    # Novel rotation test (Phase 3a) — run immediately after Phase 2
    # reconfigure_for_prediction uses these to set up Phase 3:
    cfg.test_rotations               = NOVEL_ROTATIONS
    cfg.test_no_of_steps_in_weight_space = 0   # weights plastic — models learn novel rotations over 50 blocks
    cfg.test_no_of_blocks            = NOVEL_N_BLOCKS

    return cfg


# ── Model conditions ───────────────────────────────────────────────────────────
#
#  During Phase 2:
#    Oracle Z: taskID converts llcid (radians) to int → one-hot [1,0] / [0,1].
#              Works naturally with latent_dims=[2].
#
#  During Phase 3 (novel rotations):
#    reconfigure_for_prediction forces what_latent_to_use='self' for ALL models.
#    So Oracle Z also uses self-inferred Z — its advantage is solely in its weights
#    (trained with ground-truth context labels).

COMPARISONS = [
    ('NeuraGEM',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self')),
    ('RNN',       dict(no_of_steps_in_latent_space=0, what_latent_to_use='self',)), # WU_lr=0.01)), Can increase lr to give it more realisdtic adaptation time.
    # ('Oracle Z',  dict(no_of_steps_in_latent_space=0, what_latent_to_use='taskID')),
]


# ── Train all conditions ───────────────────────────────────────────────────────

results       = {}   # Phase-2 analysis
novel_results = {}   # Phase-3a novel rotation analysis
sweep_results = {}   # Phase-3a per-rotation-angle error sweep

for label, overrides in COMPARISONS:
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    cfg = make_base_config()
    for k, v in overrides.items():
        setattr(cfg, k, v)

    logger, model, cfg, _ = train_model(
        cfg, seed=SEED,
        save_models=False, load_models=False,
        run_test_phase=True,   # runs Phase 3a with novel rotations
    )

    # ── Phase-2 analysis (trained rotations, last 1/3) ─────────────────────
    res = analyze_block_switch_adaptation(logger, cfg, n_miniblocks_to_track=N_MINIBLOCKS_TRACK)
    if label == 'NeuraGEM':
        traj, _ = analyze_z_trajectory(logger, cfg, window=Z_WINDOW)
        res['z_trajectory'] = traj
    res['logger'] = logger
    res['cfg']    = cfg
    results[label] = res
    print(f'  → Phase 2: {res["n_switches"]} switches in last 1/3')

    # ── Phase-3a analysis (novel rotations) ────────────────────────────────
    nov = analyze_novel_rotation_test(logger, cfg, n_miniblocks_to_track=N_MINIBLOCKS_TRACK)
    novel_results[label] = nov
    if nov is not None:
        print(f'  → Phase 3a: {nov["n_switches"]} switches on novel rotations')
    else:
        print('  → Phase 3a: no data (run_test_phase may have been False)')

    # ── Phase-3a rotation sweep (all angles, trials 2-5 vs analytical target mean) ─
    sweep_results[label] = analyze_rotation_sweep(logger, cfg)


# ── Training behavior panels (per model) ──────────────────────────────────────
#%%
panel_order = ['rotating_targets_behavior', 'loss', 'latent_2d']
for label, res in results.items():
    log   = res['logger']
    cfg_m = res['cfg']
    fig = plot_logger_panels(log, cfg_m, panel_order, x2=None, annotate_phases='latent_2d', width= 6)
    fig.suptitle(f'{label} — training behavior', fontsize=9, y=1.02)
    plt.show()


# ── P1–P4: Main comparison figure (trained rotations) ─────────────────────────
#%%
cfg_ref = list(results.values())[-1]['cfg']

fig = plot_adaptation_comparison(
    results, cfg_ref,
    title= None, #(f'Adaptation speed — trained rotations (0°/90°, '
           #f'blocked_phase={BLOCKED_PHASE_LENGTH}'),
)
plt.show()
print(f'Export path: {cfg_ref.export_path}')
fig_name = f'adaptation_comparison_trained_rotations_seed{SEED}.png'
fig.savefig(os.path.join(cfg_ref.export_path, fig_name), dpi=300, bbox_inches='tight')
print(f'Figure saved as: {os.path.join(cfg_ref.export_path, fig_name)}')



# ── P3b: Normalized error learning curve (trained rotations) ──────────────────
#%%
fig_norm = plot_miniblock_norm_comparison(
    results, cfg_ref,
    title='Normalized error — mini-block learning curve (trained rotations)',
)
plt.show()


# ── P5/P6: Novel rotation comparison ──────────────────────────────────────────
#%%
fig_novel = plot_novel_rotation_comparison(
    {k: v for k, v in novel_results.items() if v is not None},
    cfg_ref,
    title=(f'Generalization to novel rotations ({NOVEL_ROTATIONS}°, '
           f'WU+LU active in Phase 3a)'),
)
plt.show()


# ── P7: Arena plots — last two blocks per model (trained rotations) ────────────
#%%
fig_arena, axes = plt.subplots(3, 2, figsize=(5, 6))
for row, (label, res) in enumerate(results.items()):
    log   = res['logger']
    cfg_m = res['cfg']
    phase2_end = log.others['timestep_learning_ended']
    t_a = max(0, phase2_end - 2 * cfg_m.block_size)
    t_b = t_a + cfg_m.block_size
    plot_arena_trials(log, cfg_m, t_start=t_a, ax=axes[row, 0])
    plot_arena_trials(log, cfg_m, t_start=t_b, ax=axes[row, 1])
    axes[row, 0].set_title(f'{label} — penultimate block', fontsize=8)
    axes[row, 1].set_title(f'{label} — last block',        fontsize=8)
fig_arena.suptitle('P7: Arena predictions vs observations (trained rotations)', fontsize=9, y=1.01)
plt.tight_layout()
plt.show()
fig_arena.savefig(os.path.join(cfg_ref.export_path, f'arena_plots_trained_rotations_seed{SEED}.png'), dpi=300, bbox_inches='tight')
print(f'Figure saved as: {os.path.join(cfg_ref.export_path, f"arena_plots_trained_rotations_seed{SEED}.png")}')

# ── P8: Arena plots — novel rotations (Phase 3a) ──────────────────────────────
#%%
n_models = len(results)
fig_arena_novel, axes_novel = plt.subplots(n_models, 2, figsize=(5., 2. * n_models))
if n_models == 1:
    axes_novel = axes_novel[np.newaxis, :]
for row, (label, res) in enumerate(results.items()):
    log   = res['logger']
    cfg_m = res['cfg']
    p3a_start, p3a_end = get_phase3a_range(log)
    if p3a_start is None:
        axes_novel[row, 0].set_visible(False)
        axes_novel[row, 1].set_visible(False)
        continue
    # First and second novel-rotation blocks
    t_a = p3a_start
    t_b = p3a_start + cfg_m.block_size
    plot_arena_trials(log, cfg_m, t_start=t_a, ax=axes_novel[row, 0], phase='novel')
    plot_arena_trials(log, cfg_m, t_start=t_b, ax=axes_novel[row, 1], phase='novel')
    axes_novel[row, 0].set_title(f'{label} — 1st novel block', fontsize=8)
    axes_novel[row, 1].set_title(f'{label} — 2nd novel block', fontsize=8)
fig_arena_novel.suptitle(
    f'(novel rotations {NOVEL_ROTATIONS}°)',
    fontsize=9, y=1.01,
)
plt.tight_layout()
# plt.show()
fig_arena_novel.savefig(os.path.join(cfg_ref.export_path, f'arena_plots_novel_rotations_seed{SEED}.png'), dpi=300, bbox_inches='tight')
print(f'Figure saved as: {os.path.join(cfg_ref.export_path, f"arena_plots_novel_rotations_seed{SEED}.png")}')



# ── P9: Rotation sweep — per-angle error across all 24 test angles ────────────
#%%
fig_sweep = plot_rotation_angle_sweep(sweep_results, cfg_ref)
plt.show()
fig_sweep.savefig(os.path.join(cfg_ref.export_path, f'rotation_sweep_seed{SEED}.png'), dpi=300, bbox_inches='tight')
print(f'Figure saved as: {os.path.join(cfg_ref.export_path, f"rotation_sweep_seed{SEED}.png")}')


# ── Z-space: how NeuraGEM represents each rotation ────────────────────────────
#%%
neuragm_logger = results['NeuraGEM']['logger']
neuragm_cfg    = results['NeuraGEM']['cfg']

fig_z = plot_z_by_rotation(neuragm_logger, neuragm_cfg)
plt.show()


# ── Summary statistics ─────────────────────────────────────────────────────────
#%%
nc = cfg_ref.n_colors
header = f'{"Label":<14}' + ''.join(f'  trial {p}' for p in range(1, nc + 1))

print('\n── Phase 2: Euclidean error per trial position ──')
print(header)
for label, res in results.items():
    m = np.nanmean(res['trial_errors'], axis=0)
    print(f'{label:<14}' + ''.join(f'  {v:6.4f} ' for v in m))

print('\n── Phase 2: normalized state error per trial position ──')
print(header)
for label, res in results.items():
    m = np.nanmean(res['trial_norm_errors'], axis=0)
    print(f'{label:<14}' + ''.join(f'  {v:6.4f} ' for v in m))

print(f'\n── Phase 3a ({len(NOVEL_ROTATIONS)} novel angles, 0–345° in 15° steps): normalized state error per trial position ──')
print(header)
for label, nov in novel_results.items():
    if nov is None:
        print(f'{label:<14}  (no data)')
        continue
    m = np.nanmean(nov['trial_norm_errors'], axis=0)
    print(f'{label:<14}' + ''.join(f'  {v:6.4f} ' for v in m))

# %%
