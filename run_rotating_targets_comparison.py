"""
run_rotating_targets_comparison.py — Adaptation speed comparison on the rotating-targets task.

Trains three model conditions and compares how quickly each adapts after an
unsignalled block switch in the last 1/3 of Phase 2 training:

  NeuraGEM  : self-inferred Z updated each batch via LU
  RNN       : Z frozen at initialisation — weights adapt but context does not
  Oracle Z  : true rotation angle injected as task ID (upper-bound ceiling)

After Phase 2, Phase 3a immediately runs with novel rotations and weights frozen
(test_no_of_steps_in_weight_space=0), so any adaptation there is context inference
alone: it measures how quickly each model's Z generalizes to unseen angles.

Produces:
  P1 — Euclidean error per trial position in first mini-block after switch
  P2 — Normalized state error per trial position  (0=correct, 0.5=chance, 1=wrong)
  P3 — Per-mini-block learning curve after switch (trained rotations)
  P4 — Z-trajectory (NeuraGEM only): displacement from old → new Z centroid
  P5 — Novel rotation normalized error per trial position
  P6 — Novel rotation mini-block learning curve
  P7 — Arena plots: qualitative predictions vs observations (last 2 trained blocks)
  P10 — Rotation decoding: latent Z vs hidden activity, held-out novel angles
  P11 — How distributed the hidden rotation code is: error vs #components retained
  P12 — What KIND of rotation code it is: RSA against circular / blend / categorical / …
"""

import os
# Must precede the torch import (via plot_style → functions_and_utils) to take effect.
# batch_size=1 with a 64-unit LSTM is far too small to benefit from intra-op threading,
# and on a busy shared node the threads contend and stall for seconds at a time. Measured
# on a 4900-batch Phase 2: 2 threads = 35-52s with dips to 30 it/s; 1 thread = 31.2s flat
# at 157 it/s. Remove these if you ever move to large batches or a dedicated node.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

import plot_style
plot_style.set_plot_style()
# FigSize presets are paper-panel sized (1.5-2in) and hard to read on screen. dev() doubles
# every preset without changing layout; comment it out before exporting paper figures.
plot_style.FigSize.paper()      # 1x, for paper-ready export

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
from rotation_decoding_analysis import (
    compare_z_lags,
    component_sweep_across_models,
    decode_rotation_across_models,
    plot_component_sweep,
    plot_decoded_vs_true,
    plot_decoding_comparison,
    print_lag_table,
    summarize_component_sweep,
    summarize_decoding,
    verify_hidden_state_alignment,
)
from rotation_geometry_analysis import (
    geometry_across_models,
    plot_candidate_rdms,
    plot_geometry_embedding,
    plot_rdm_fits,
    print_confusability,
    summarize_geometry,
)

# ── Experiment parameters ──────────────────────────────────────────────────────

BLOCKED_PHASE_LENGTH = 6000   # longer → more block switches in last 1/3
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
    # cfg.train_rotations = [0.0, 70.0, 115.0]  
    cfg.train_rotations = [0.0, 120.0, 210.0]  # if more than two, need ot increase blocksize.
    # cfg.train_rotations = [0.0, 120.0, 115.0, 215.0]  # if more than two, need ot increase blocksize.
    # cfg.train_rotations = [0.0, 90.0]  
    cfg.run_no_updates = False # the phase where no weights or latent updates are applied 
    cfg.log_hidden_states = True # for later decoding analysis 
    
    # Oracle Z one-hots each trained rotation into its own Z slot, so Z_dim must cover them all.
    # Kept identical across conditions so NeuraGEM/RNN/Oracle have the same latent capacity.
    cfg.latent_dims = [max(2, len(cfg.train_rotations))]

    cfg.add_passive_learning_phase = True
    cfg.passive_phase_length  = PASSIVE_PHASE_LENGTH
    cfg.blocked_phase_length  = BLOCKED_PHASE_LENGTH

    cfg.n_miniblocks_per_state_block = 14 #7  # USING DOUBLE BECAUSE RNN UNABLE TO SWITCH FULLY
    # The paper uses 6-8 miniblocks per state block
    # total timesteps= 7*n_colors*2 (for cue and outcome) = 60 per block
    cfg.block_size = cfg.n_miniblocks_per_state_block * cfg.n_colors * 2 # 120

    cfg.WU_lr    = 0.001
    cfg.pass_previous_latent = True
    
    cfg.use_add_gating = False
    cfg.use_mul_gating = not cfg.use_add_gating
    if cfg.use_mul_gating:
        cfg.pre_gating  = False # even oracles do not work with pre_gating. Strange. I wonder if additive input might work better.
        cfg.post_gating = not cfg.pre_gating

    if cfg.pre_gating:
        cfg.Z_lr = 0.1
        cfg.Z_decay = 0.000_0011
    else:
        cfg.Z_lr    = 0.5
        cfg.Z_decay  = 0.000_1
    if cfg.use_add_gating:
        cfg.Z_lr = 0.3
        cfg.Z_decay = 0.000_0011

    # Novel rotation test (Phase 3a) — run immediately after Phase 2
    # reconfigure_for_prediction uses these to set up Phase 3:
    cfg.test_rotations               = NOVEL_ROTATIONS
    cfg.test_no_of_steps_in_weight_space = 0   # weights frozen — Phase 3a adaptation is Z inference only
    cfg.test_no_of_blocks            = NOVEL_N_BLOCKS

    return cfg


# ── Model conditions ───────────────────────────────────────────────────────────
#
#  During Phase 2, both oracles read the true rotation from llcid (radians) but encode it
#  differently — the comparison is identity-only context vs. context with a metric:
#    Oracle Z (one-hot) : nearest entry of cfg.oracle_context_values (= deg2rad(train_rotations))
#                         picks a slot. Z_dim must cover every trained rotation. Distinguishes the
#                         trained contexts but has no notion of angular distance between them.
#    Oracle Z (cont.)   : rotation space is preserved, so unseen angles land between the trained
#                         ones. Two encodings, both bypassing latent_activation:
#                         'circular'   Z = [(1+cos θ)/2, (1+sin θ)/2] — no wraparound seam, and
#                                      never all-zero, so the multiplicative gate stays alive.
#                         'normalized' Z = θ/2π, one dim, directly readable as an angle. Caveat:
#                                      θ=0 → Z=0 zeroes the gate for the whole 0° block, and
#                                      359° sits maximally far from 0° despite being the same
#                                      context. Kept commented below.
#
#  During Phase 3 (novel rotations):
#    reconfigure_for_prediction forces what_latent_to_use='self' for ALL models, and applies
#    cfg.test_no_of_steps_in_latent_space (=1 for this task) so the oracles — which train with
#    LU off — actually infer Z here. Without that they would run Phase 3 with Z pinned at its
#    initial zeros: no inference at all, and for a 'none'-activation model a zeroed gate.
#    So both oracles use self-inferred Z; their advantage is solely in their weights
#    (trained with ground-truth context labels).

COMPARISONS = [
    # Note for NG Z_LU works best. 0.4 0.2 not so much
    # But it is the least circular... but best angle retrieval and gen
    # improved interpolation but hurt extrapolation..
    # Also lowering WU_lr to 0.0005 had similar effects. 
    ('NeuraGEM',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self')),
    # ('NeuraGEM_WU_0.0005',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self', WU_lr=0.0005)),
    # ('NeuraGEM_WU_0.0001',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self', WU_lr=0.0001, blocked_phase_length=10000)),
    # ('NeuraGEM_ZU_0.3',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self', Z_lr = 0.3)),
    # ('NeuraGEM_ZU_0.2',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self', Z_lr = 0.2)),
    # ('NeuraGEM_ZU_0.1',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self', Z_lr = 0.1)),
    # ('NeuraGEM_dec_-3',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self', Z_decay = 0.000_1)),
    # ('NeuraGEM_dec_-4',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self', Z_decay = 0.000_01)),
    # ('NeuraGEM_dec_-5',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self', Z_decay = 0.000_001)),

    # ('NeuraGEM_10',  dict(no_of_steps_in_latent_space=1, what_latent_to_use='self', latent_dims = [10])),
    # RNN keeps Z frozen in Phase 3 too — that is the point of the control, so it opts out of the
    # task default with test_no_of_steps_in_latent_space=0.

    # ('RNN',       dict(no_of_steps_in_latent_space=0, what_latent_to_use='self', test_no_of_steps_in_latent_space=0)), # WU_lr=0.01)), Can increase lr to give it more realisdtic adaptation time.
# ]
# f = [ #only run NeuraGEM 
    ('Oracle Z (one-hot)', dict(no_of_steps_in_latent_space=0, what_latent_to_use='context_ids',
                                oracle_context_encoding='one_hot')),
    # latent_activation='none' because softmax over a 1-dim Z is a constant 1.0 — it would leave
    # this model's Phase-3 self-inferred Z inert. 'none' also matches the raw scalar its weights
    # were trained on in Phase 2.
    ('Oracle Z (Norm)',   dict(no_of_steps_in_latent_space=0, what_latent_to_use='context_ids',
                                oracle_context_encoding='normalized', latent_dims=[1],
                                latent_activation='none')),
    # Phase 3a starts from Z=0 here (LU is off in Phases 1-2, so this model's own Z parameter is
    # untouched until then), which with latent_activation='none' is a momentarily dead gate.
    # Measured cost: trial-1 error after the first novel switch only, identical from trial 2 on.

    ('Oracle Z (cont.)',   dict(no_of_steps_in_latent_space=0, what_latent_to_use='context_ids',
                                oracle_context_encoding='circular', latent_dims=[2],
                                latent_activation='none')),
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
    # Stored as scalars today; the decoding analysis also accepts 'loggers'/'cfgs' lists,
    # so moving to many seeds is a change here rather than in the analysis module.
    res['logger'] = logger
    res['cfg']    = cfg
    res['model']  = model
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
panel_order = ['rotating_targets_behavior', 'latent_2d']
for label, res in results.items():
    log   = res['logger']
    cfg_m = res['cfg']
    fig = plot_logger_panels(log, cfg_m, panel_order,  annotate_phases='latent_2d', width= 6, dpi = 200, x2=None)#6000)
    fig = plot_logger_panels(log, cfg_m, ['rotating_targets_behavior', 'latent_2d'],  annotate_phases='latent_2d', width= 6, dpi = 200, x2=6000)
    fig.suptitle(f'{label} — training behavior', fontsize=9, y=1.02)
    plt.show()

#%%
# ── P1–P4: Main comparison figure (trained rotations) ─────────────────────────

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
fig_norm = plot_miniblock_norm_comparison(
    results, cfg_ref,
    title='Normalized error — mini-block learning curve (trained rotations)',
)
plt.show()


# ── P5/P6: Novel rotation comparison ──────────────────────────────────────────
fig_novel = plot_novel_rotation_comparison(
    {k: v for k, v in novel_results.items() if v is not None},
    cfg_ref,
    title=(f'Generalization to novel rotations ({NOVEL_ROTATIONS}°, '
           f'LU only in Phase 3a — weights frozen)'),
)
plt.show()


# ── P7: Arena plots — last two blocks per model (trained rotations) ────────────
arena_plots = False
if arena_plots:
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
if 'NeuraGEM' in results:
    neuragm_logger = results['NeuraGEM']['logger'] 
    neuragm_cfg    = results['NeuraGEM']['cfg']

    fig_z = plot_z_by_rotation(neuragm_logger, neuragm_cfg, )
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


# ── P10: Rotation decoding — latent Z vs hidden activity ──────────────────────
#
# How much of the novel rotation angle is recoverable from Z, versus from the RNN's
# hidden state?  Cross-validation holds out whole *angles*, so a low error means the
# representation interpolates to rotations the decoder never saw — not that it memorised
# a lookup of the trained contexts.  Each bar's own block-permuted null is the chance
# reference; a null at chance is what shows the 64-dim hidden state is not just overfitting.
#
# Z_dim (2-3) vs hidden_size (64) is NOT corrected by shrinking the hidden state to Z_dim:
# a linear decoder onto the 2-D target [cos θ, sin θ] has a rank-2 readout however many
# inputs it gets, so the extra width buys no readout power — only overfitting freedom,
# which the held-out-angle CV and the null already control.  The dimensionality question
# is instead asked properly by P11 below.
#%%

# The whole pipeline assumes logger.hidden_states[t] is the state that produced the
# logged prediction at t.  Check it once rather than assume it.
for label, res in results.items():
    ok, max_diff = verify_hidden_state_alignment(res['logger'], res['cfg'], res['model'])
    print(f'  {label}: hidden-state alignment {"ok" if ok else "FAILED"} '
          f'(max |Wh+b − logged prediction| = {max_diff:.2e})')

print('\nDecoding rotation from Phase 3a (novel angles, held-out-angle CV):')
decode_results = decode_rotation_across_models(results, phase='phase3a', frames='all')

summarize_decoding(decode_results)

fig_dec = plot_decoding_comparison(decode_results, cfg_ref)
plt.show()
fig_dec.savefig(os.path.join(cfg_ref.export_path, f'rotation_decoding_seed{SEED}.png'),
                dpi=300, bbox_inches='tight')

fig_dec_scatter = plot_decoded_vs_true(decode_results)
plt.show()
fig_dec_scatter.savefig(
    os.path.join(cfg_ref.export_path, f'rotation_decoding_scatter_seed{SEED}.png'),
    dpi=300, bbox_inches='tight')

# Z is logged after its latent update while the hidden state comes from the weight-update
# forward pass that ran with the previous batch's Z, so Z sits one LU step ahead at the
# same index.  lag -1 pairs each hidden state with the Z that actually drove it.
print_lag_table(compare_z_lags(results))


# ── P11: How distributed is the rotation code? ────────────────────────────────
#
# The dimensionality question, asked properly: how many dimensions of hidden activity do
# you need to recover the rotation, given that Z does it in Z_dim?  Two reduction methods,
# both fit inside each fold:
#   PLS — supervised, components chosen to predict rotation.  The honest answer.
#   PCA — unsupervised, components ordered by variance.  A large PLS/PCA gap means the
#         rotation code lives in low-variance directions of the hidden state, which is
#         exactly why a PCA-matched bar understates it so badly (the top PCs carry only
#         one of the two circular coordinates, leaving cos θ mirror-ambiguous).
#%%

sweep_pls = component_sweep_across_models(results, method='pls')
sweep_pca = component_sweep_across_models(results, method='pca', verbose=False)
summarize_component_sweep(sweep_pls)

fig_sweep_dim = plot_component_sweep(sweep_pls, sweep_pca)
plt.show()
fig_sweep_dim.savefig(
    os.path.join(cfg_ref.export_path, f'rotation_decoding_component_sweep_seed{SEED}.png'),
    dpi=300, bbox_inches='tight')


# ── P12: What KIND of rotation code is it? (RSA) ──────────────────────────────
#
# Decoding says how much rotation information is present; this says what shape it has.
# Representational similarity: average the state within each rotation, take pairwise
# distances between those states, and ask which hypothesis about the code predicts that
# pattern of distances. The same comparison applies unchanged to neural data.
#
# Read plot_candidate_rdms FIRST — it shows what each hypothesis predicts before any data
# is involved. Then the fit bars, then the 2-D map (ring? triangle? clusters?).
#
# The oracles do NOT calibrate this here: reconfigure_for_prediction forces
# what_latent_to_use='self' for every condition, so in Phase 3a no oracle uses its designed
# encoding — all conditions self-infer Z and differ only in their weights. ('Oracle Z
# (one-hot)' duly returns a circular fit, not a categorical one.) So the noise ceiling is
# the upper reference; for a lower one, re-run once with candidates=ALL_CANDIDATES — the
# 'constellation' hypothesis should sit at ~0, which is what shows the method can say "no".
#%%

test_angles = np.array(sorted(cfg_ref.test_rotations or cfg_ref.train_rotations), dtype=float)

# The hypotheses are not orthogonal. Check how confusable they are under THIS design
# before reading any winner — and note that train_rotations is a lever on that.
print_confusability(test_angles, cfg_ref.train_rotations, cfg_ref.n_colors)

fig_cand = plot_candidate_rdms(test_angles, cfg_ref.train_rotations, cfg_ref.n_colors)
plt.show()
fig_cand.savefig(os.path.join(cfg_ref.export_path, 'rotation_code_hypotheses.png'),
                 dpi=300, bbox_inches='tight')

geometry_results = geometry_across_models(results, keys=('Z', 'H'),
                                          phase='phase3a', frames='all')
summarize_geometry(geometry_results, key='Z')
summarize_geometry(geometry_results, key='H')

fig_geo = plot_rdm_fits(geometry_results, key='Z')
plt.show()
fig_geo.savefig(os.path.join(cfg_ref.export_path, f'rotation_code_fits_seed{SEED}.png'),
                dpi=300, bbox_inches='tight')

fig_embed = plot_geometry_embedding(geometry_results, key='Z')
plt.show()
fig_embed.savefig(os.path.join(cfg_ref.export_path, f'rotation_code_map_seed{SEED}.png'),
                  dpi=300, bbox_inches='tight')

# %%
