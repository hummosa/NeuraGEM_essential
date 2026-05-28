"""
run_flanker.py — Flanker task pretraining.

Task
────
5-slot arrow display: [far_left, near_left, center, near_right, far_right].
Each trial presents a noisy arrow at the target slot and one companion slot.
The companion direction is correlated with the target with probability that
decreases with slot distance (p_corr_by_distance). The RNN must infer the
target direction from the noisy observations.

Speed pressure
──────────────
Temporal discount loss weights concentrate the gradient signal at early timesteps
(response_start_timestep=1 onward). The model is pushed to be correct as early as
possible while still integrating evidence over arrows_duration steps.

Context structure
─────────────────
Within each context block the target slot is fixed. Z (latent_dims=[5], softmax)
learns a one-hot-like encoding of which slot is the target for the current block.
"""

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

#%%
import plot_style
plot_style.set_plot_style()

from plot_style import FigSize
from functions_and_utils import *
from configs import FlankerTaskConfig
from train_and_infer_functions import *


# ── Configure ─────────────────────────────────────────────────────────────────

config = FlankerTaskConfig(experiment_to_run='default')
config.run_name = 'flanker_pretrain_v1'
config.env_seed = 45
# Tweak any parameters here before training, e.g.:
# config.temporal_decay_factor = 0.0   # disable speed pressure (equal weights)
# config.p_corr_by_distance    = [1.0, 0.9, 0.7, 0.5, 0.3]  # stronger correlation
# config.arrow_noise_std       = 2.0   # harder observations
# config.n_training_contexts   = 500  # WRong cannot be changed here more training
config.arrow_noise_std       = 0.1   # harder observations
if config.arrow_noise_std > 0.5:
    config.blocked_phase_length = 10000
else:
    config.blocked_phase_length = 4000 # for low noise
# config.P_gates_bernoulli_prob = 0.5

# config.block_size = 5 # not a huge diff.
config.hidden_size = 64
# config.p_corr_by_distance = [1.0, 0.0, 0.0, 0.0, 0.0]  # All incongruent. 
        
# config.use_add_gating = True
# config.use_mul_gating = False

# config.Z_lr                  = 0.5
gating = 'post' # 'post'
config.pre_gating = gating == 'pre'
config.post_gating = gating == 'post'

# ── Train ─────────────────────────────────────────────────────────────────────

print(f'Training flanker task pretraining | seed={config.env_seed}')
print(f'  arrows_duration={config.arrows_duration}  '
      f'trials_per_context_block={config.trials_per_context_block}  '
      f'n_training_contexts={config.n_training_contexts}')
print(f'  temporal_loss_weights={[round(w, 3) for w in config.temporal_loss_weights]}')
print(f'  p_corr_by_distance={config.p_corr_by_distance}')

logger, model, config, figs = train_model(
    config, seed=config.env_seed, save_models=False, load_models=False,
)

# ── Plot ──────────────────────────────────────────────────────────────────────

panel_order = ['behavior', 'latent_2d', 'corrects']
fig = plot_logger_panels(logger, config, panel_order, x2=None,
                         annotate_phases='behavior', width=5, legends=False, dpi=140)
fig.savefig(config.export_path + 'flanker_pretrain_results.pdf', bbox_inches='tight')
print(f'Exported to: {config.export_path + "flanker_pretrain_results.pdf"}')

#%%
# ── Per-trial accuracy by timestep ────────────────────────────────────────────
#
# Left:   accuracy by training third (early/mid/late).
# Middle: accuracy split by congruency.
# Right:  RT proxy CDF — fraction of trials decided by each timestep (late third),
#         where "decided" = |output[t, -1]| > rt_threshold.

rt_threshold = 0.5   # adjust if output magnitudes differ; output is linear, target = ±1

ii_all   = np.concatenate(logger.inputs,            axis=0).reshape(-1, config.input_size)
oi_all   = np.concatenate(logger.predicted_outputs,  axis=0).reshape(-1, config.output_size)
cong_all = np.concatenate(logger.hlcids,             axis=0).reshape(-1)

ad       = config.arrows_duration
n_trials = len(ii_all) // ad

correct_all  = (ii_all[:, -1] * oi_all[:, -1] > 0).astype(float)
correct      = correct_all[:n_trials * ad].reshape(n_trials, ad)
output_traj  = oi_all[:n_trials * ad, -1].reshape(n_trials, ad)
# congruency is constant within a trial; take first timestep per trial
cong_trial   = cong_all[:n_trials * ad].reshape(n_trials, ad)[:, 0]

third  = n_trials // 3
splits = [
    (slice(0,         third),    'early'),
    (slice(third,     2 * third), 'mid'),
    (slice(2 * third, n_trials),  'late'),
]
colors_thirds = ['#a8c8e8', '#4393c3', '#084594']   # light → dark blue

fig_acc, (ax_thirds, ax_cong, ax_rt) = plt.subplots(1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1]))

for (slc, label), color in zip(splits, colors_thirds):
    ax_thirds.plot(correct[slc].mean(axis=0), color=color, label=label)
ax_thirds.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax_thirds.axvspan(-0.5, config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax_thirds.set_xticks(range(ad))
ax_thirds.set_xlabel('Timestep within trial')
ax_thirds.set_ylabel('Accuracy')
ax_thirds.set_ylim(0.3, 1.05)
ax_thirds.legend(fontsize=5)

congruent_mask   = cong_trial == 1.0
incongruent_mask = cong_trial == 0.0
for mask, label, color in [
    (congruent_mask,   'congruent',   '#4393c3'),
    (incongruent_mask, 'incongruent', '#d6604d'),
]:
    if mask.any():
        ax_cong.plot(correct[mask].mean(axis=0), color=color, label=f'{label} (n={mask.sum()})')
ax_cong.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax_cong.axvspan(-0.5, config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax_cong.set_xticks(range(ad))
ax_cong.set_xlabel('Timestep within trial')
ax_cong.set_ylabel('Accuracy')
ax_cong.set_ylim(0.3, 1.05)
ax_cong.legend(fontsize=5)

# RT proxy: first timestep where |output| > rt_threshold; undecided = arrows_duration
decided = np.abs(output_traj) > rt_threshold          # (n_trials, ad) bool
rt      = np.where(decided.any(axis=1),
                   decided.argmax(axis=1).astype(float),
                   float(ad))                          # argmax returns first True; else censored

# CDF over late-training trials, split by congruency
late_mask = np.zeros(n_trials, dtype=bool)
late_mask[2 * third:] = True
for mask, label, color in [
    (congruent_mask   & late_mask, 'congruent',   '#4393c3'),
    (incongruent_mask & late_mask, 'incongruent', '#d6604d'),
]:
    if mask.any():
        cdf = np.array([(rt[mask] <= t).mean() for t in range(ad)])
        ax_rt.plot(range(ad), cdf, color=color, label=f'{label} (n={mask.sum()})')
ax_rt.axvspan(-0.5, config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax_rt.set_xticks(range(ad))
ax_rt.set_xlabel('Timestep within trial')
ax_rt.set_ylabel(f'Fraction decided (|out| > {rt_threshold})')
ax_rt.set_ylim(0.0, 1.05)
ax_rt.legend(fontsize=5)

fig_acc.tight_layout()
fig_acc.savefig(config.export_path + 'flanker_accuracy_by_timestep.pdf', bbox_inches='tight')
print(f'Exported to: {config.export_path + "flanker_accuracy_by_timestep.pdf"}')

#%%
# ── Stage 2: frozen weights, self-learned Z on full-flanker blocks ─────────────
#
# Weights from Stage 1 are frozen (no_of_steps_in_weight_space=0).
# Z is reset and re-optimized by LU across alternating congruent / incongruent blocks.
# Expected: Z flat in congruent blocks, peaked on center dim in incongruent blocks.
#
# !! IMPORTANT — model.config vs stage config !!
# The model stores its own copy of the config (model.config) set at construction time.
# Updating stage2_config / stage3_config does NOT automatically update model.config.
# Any parameter that the model reads at forward/LU time (e.g. Z_lr, latent_dims,
# no_of_steps_in_latent_space) must be patched on model.config explicitly:
#
#     model.config.Z_lr = stage2_config.Z_lr
#
# Parameters consumed only by train_model's outer loop (e.g. no_of_steps_in_weight_space,
# blocked_phase_length, dataset_name) are read from the stage config and do not need
# to be mirrored. When in doubt, patch both.

from configs import FlankerTaskStage2Config


stage2_config = FlankerTaskStage2Config(experiment_to_run='default')

stage2_config.block_size = 300  # longer blocks to give LU more time to optimize Z
stage2_config.blocked_phase_length = stage2_config.block_size * stage2_config.n_training_contexts

stage2_config.run_name = 'flanker_stage2_v1'
stage2_config.pre_gating = config.pre_gating
stage2_config.post_gating = config.post_gating
stage2_config.use_add_gating = config.use_add_gating
stage2_config.use_mul_gating = config.use_mul_gating

stage2_config.Z_lr = 0.4
model.config.Z_lr = 0.4   # mirror on model.config — model reads Z_lr from model.config at LU time
stage2_config.no_of_steps_in_latent_space = 1

print(f'Stage 2: frozen weights, learned Z | '
      f'blocks={stage2_config.n_training_contexts}  '
      f'trials_per_block={stage2_config.trials_per_context_block}')

# Reset Z to random — Stage 1 Z was oracle-guided, not self-organized
# Use set_Z() so _rebuild_Z_optimizer() is called and Z_optimizer tracks the new parameter
model.set_Z(torch.randn_like(model.Z) * 0.2)

logger2, model2, stage2_config, figs2 = train_model(
    stage2_config, seed=stage2_config.env_seed,
    save_models=False, load_models=False,
    pretrained_model=model,
)

panel_order = ['latent_2d', 'corrects']
fig2 = plot_logger_panels(logger2, stage2_config, panel_order, x2=None,
                           width=5, legends=False, dpi=140,)
# fig2 = plot_logger_panels(logger2, stage2_config, ['corrects'], x2=None,
#                           annotate_phases='corrects', width=5, legends=False, dpi=140,)
fig2.savefig(stage2_config.export_path + 'flanker_stage2_results.pdf', bbox_inches='tight')
print(f'Exported to: {stage2_config.export_path + "flanker_stage2_results.pdf"}')

#%%
# ── Stage 2: per-trial accuracy by timestep ───────────────────────────────────
#
# Left:   accuracy by training phase (from logger2.phases).
# Middle: accuracy split by congruency for a chosen phase.
# Right:  RT proxy CDF — congruent vs incongruent for the same phase.
#
# Change accuracy_phase_idx to select which phase to use for subplots 2 & 3:
#   0 = "Learning and inference"
#   1 = "Inference only"
#   2 = "No inference nor learning"

rt_threshold2     = 0.5
accuracy_phase_idx = 0

ii2_all   = np.concatenate(logger2.inputs,            axis=0).reshape(-1, stage2_config.input_size)
oi2_all   = np.concatenate(logger2.predicted_outputs,  axis=0).reshape(-1, stage2_config.output_size)
cong2_all = np.concatenate(logger2.hlcids,             axis=0).reshape(-1)

ad2       = stage2_config.arrows_duration
n_trials2 = len(ii2_all) // ad2

correct2_all = (ii2_all[:, -1] * oi2_all[:, -1] > 0).astype(float)
correct2     = correct2_all[:n_trials2 * ad2].reshape(n_trials2, ad2)
output2_traj = oi2_all[:n_trials2 * ad2, -1].reshape(n_trials2, ad2)
cong2_trial  = cong2_all[:n_trials2 * ad2].reshape(n_trials2, ad2)[:, 0]

# phase boundaries in trial indices
phase_names  = [p[0] for p in logger2.phases]
phase_starts = [p[1] // ad2 for p in logger2.phases]
phase_ends   = phase_starts[1:] + [n_trials2]
phase_slices = [slice(s, e) for s, e in zip(phase_starts, phase_ends)]

colors_phases = ['#a8c8e8', '#4393c3', '#084594']
colors_phases = (colors_phases * ((len(phase_slices) // len(colors_phases)) + 1))[:len(phase_slices)]

fig2_acc, (ax2_phases, ax2_cong, ax2_rt) = plt.subplots(
    1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1])
)

# ── subplot 1: accuracy by phase ──────────────────────────────────────────────
for slc, name, color in zip(phase_slices, phase_names, colors_phases):
    if correct2[slc].shape[0] > 0:
        ax2_phases.plot(correct2[slc].mean(axis=0), color=color, label=name)
ax2_phases.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax2_phases.axvspan(-0.5, stage2_config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax2_phases.set_xticks(range(ad2))
ax2_phases.set_xlabel('Timestep within trial')
ax2_phases.set_ylabel('Accuracy')
ax2_phases.set_ylim(0.3, 1.05)
ax2_phases.legend(fontsize=5)

# ── subplot 2: congruent vs incongruent accuracy (selected phase) ─────────────
phase_mask2 = np.zeros(n_trials2, dtype=bool)
phase_mask2[phase_slices[accuracy_phase_idx]] = True

congruent2_mask   = cong2_trial == 1.0
incongruent2_mask = cong2_trial == 0.0
for mask, label, color in [
    (congruent2_mask   & phase_mask2, 'congruent',   '#4393c3'),
    (incongruent2_mask & phase_mask2, 'incongruent', '#d6604d'),
]:
    if mask.any():
        ax2_cong.plot(correct2[mask].mean(axis=0), color=color,
                      label=f'{label} (n={mask.sum()})')
ax2_cong.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax2_cong.axvspan(-0.5, stage2_config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax2_cong.set_xticks(range(ad2))
ax2_cong.set_xlabel('Timestep within trial')
ax2_cong.set_ylabel('Accuracy')
ax2_cong.set_ylim(0.3, 1.05)
# ax2_cong.set_title(f'Phase: {phase_names[accuracy_phase_idx]}', fontsize=7)
print(f'Phase "{phase_names[accuracy_phase_idx]}" | '
      f'congruent n={congruent2_mask.sum()}  incongruent n={incongruent2_mask.sum()}')
ax2_cong.legend(fontsize=5)

# ── subplot 3: RT proxy CDF, congruent vs incongruent (selected phase) ────────
decided2 = np.abs(output2_traj) > rt_threshold2
rt2 = np.where(decided2.any(axis=1),
               decided2.argmax(axis=1).astype(float),
               float(ad2))

for mask, label, color in [
    (congruent2_mask   & phase_mask2, 'congruent',   '#4393c3'),
    (incongruent2_mask & phase_mask2, 'incongruent', '#d6604d'),
]:
    if mask.any():
        cdf2 = np.array([(rt2[mask] <= t).mean() for t in range(ad2)])
        ax2_rt.plot(range(ad2), cdf2, color=color, label=f'{label} (n={mask.sum()})')
ax2_rt.axvspan(-0.5, stage2_config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax2_rt.set_xticks(range(ad2))
ax2_rt.set_xlabel('Timestep within trial')
ax2_rt.set_ylabel(f'Fraction decided (|out| > {rt_threshold2})')
ax2_rt.set_ylim(0.0, 1.05)
# ax2_rt.set_title(f'Phase: {phase_names[accuracy_phase_idx]}', fontsize=7)
ax2_rt.legend(fontsize=5)

fig2_acc.tight_layout()
fig2_acc.savefig(stage2_config.export_path + 'flanker_stage2_accuracy_by_timestep.pdf',
                 bbox_inches='tight')
print(f'Exported to: {stage2_config.export_path + "flanker_stage2_accuracy_by_timestep.pdf"}')

#%%
# ── Stage 2: RT exploratory — correct vs wrong, split by congruency ───────────
#
# Two subplots: left = congruent trials, right = incongruent trials.
# Each subplot has two lines: correct trials and wrong trials.
# "Correct" = model output at the final trial timestep has the right sign.
# RT = first timestep where |output| > rt_threshold2; undecided = arrows_duration.
# Trials from the phase selected by accuracy_phase_idx above.

trial_correct2 = correct2[:, -1].astype(bool)   # correct at last timestep

fig2_rt2, (ax_rt_cong, ax_rt_incong) = plt.subplots(
    1, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1]), sharey=True
)

for ax, cong_mask, cong_label in [
    (ax_rt_cong,   congruent2_mask,   'Congruent'),
    (ax_rt_incong, incongruent2_mask, 'Incongruent'),
]:
    for correct_flag, label, color, ls in [
        (True,  'correct', '#4393c3', '-'),
        (False, 'wrong',   '#d6604d', '--'),
    ]:
        mask = cong_mask & phase_mask2 & (trial_correct2 == correct_flag)
        if mask.any():
            cdf_rt = np.array([(rt2[mask] <= t).mean() for t in range(ad2)])
            ax.plot(range(ad2), cdf_rt, color=color, linestyle=ls,
                    label=f'{label} (n={mask.sum()})')
    ax.axvspan(-0.5, stage2_config.response_start_timestep - 0.5, alpha=0.08, color='k')
    ax.set_xticks(range(ad2))
    ax.set_xlabel('Timestep within trial')
    ax.set_ylabel(f'Fraction decided (|out| > {rt_threshold2})')
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f'{cong_label}', fontsize=7)
    ax.legend(fontsize=5)

fig2_rt2.tight_layout()
fig2_rt2.savefig(stage2_config.export_path + 'flanker_stage2_rt_correct_vs_wrong.pdf',
                 bbox_inches='tight')
print(f'Exported to: {stage2_config.export_path + "flanker_stage2_rt_correct_vs_wrong.pdf"}')

#%%
# ── Stage 3: distance × congruency flanker effect ─────────────────────────────
#
# 4 block types cross near/far × congruent/incongruent.
# Weights frozen; Z updates via LU. run_test_phase=False — single blocked phase only.

from configs import FlankerTaskStage3Config

stage3_config = FlankerTaskStage3Config(experiment_to_run='default')
stage3_config.run_name = 'flanker_stage3_v1'
stage3_config.pre_gating     = config.pre_gating
stage3_config.post_gating    = config.post_gating
stage3_config.use_add_gating = config.use_add_gating
stage3_config.use_mul_gating = config.use_mul_gating

stage3_config.Z_lr = stage2_config.Z_lr
stage3_config.no_of_steps_in_latent_space = 1
model.config.Z_lr = stage3_config.Z_lr
model.config.no_of_steps_in_latent_space = 1

stage3_config.block_size = 300
stage3_config.blocked_phase_length = stage3_config.block_size * stage3_config.n_training_contexts

print(f'Stage 3: distance × congruency | '
      f'blocks={stage3_config.n_training_contexts}  block_size={stage3_config.block_size}')

model.set_Z(torch.randn_like(model.Z) * 0.2)

logger3, model3, stage3_config, figs3 = train_model(
    stage3_config, seed=stage3_config.env_seed,
    save_models=False, load_models=False,
    pretrained_model=model2,
    run_test_phase=False,
)

panel_order3 = ['behavior', 'latent_2d', 'corrects']
fig3_panels = plot_logger_panels(logger3, stage3_config, panel_order3, x2=None,
                                  annotate_phases='corrects', width=5, legends=False, dpi=140)

# Label first full cycle of block types on latent_2d subplot
ax_z3 = fig3_panels.axes[1]
_block_labels = ['Near\nCong', 'Near\nIncong', 'Far\nCong', 'Far\nIncong']
bs3 = stage3_config.block_size
for i, lbl in enumerate(_block_labels):
    ax_z3.text(bs3 * i + bs3 / 2, 1.0, lbl,
               transform=ax_z3.get_xaxis_transform(),
               ha='center', va='top', fontsize=5)

fig3_panels.savefig(stage3_config.export_path + 'flanker_stage3_panels.pdf', bbox_inches='tight')
print(f'Exported to: {stage3_config.export_path + "flanker_stage3_panels.pdf"}')

#%%
# ── Stage 3 analysis: 2×2 distance × congruency ──────────────────────────────
#
# Left:  accuracy by timestep — 4 conditions (near/far × cong/incong).
# Right: RT proxy CDF         — same 4 conditions.
#
# Key prediction: near-incongruent accuracy lowest / RT slowest;
# near flanker congruency effect > far flanker congruency effect (interaction).
#
# Block type stored in logger3.hlcids: 0=near-cong, 1=near-incong, 2=far-cong, 3=far-incong.

rt_threshold3 = 0.5

ii3_all = np.concatenate(logger3.inputs,           axis=0).reshape(-1, stage3_config.input_size)
oi3_all = np.concatenate(logger3.predicted_outputs, axis=0).reshape(-1, stage3_config.output_size)
bt3_all = np.concatenate(logger3.hlcids,            axis=0).reshape(-1)

ad3       = stage3_config.arrows_duration
n_trials3 = len(ii3_all) // ad3

correct3     = (ii3_all[:n_trials3 * ad3, -1] * oi3_all[:n_trials3 * ad3, -1] > 0
               ).astype(float).reshape(n_trials3, ad3)
output3_traj = oi3_all[:n_trials3 * ad3, -1].reshape(n_trials3, ad3)
bt3_trial    = bt3_all[:n_trials3 * ad3].reshape(n_trials3, ad3)[:, 0]

decided3 = np.abs(output3_traj) > rt_threshold3
rt3 = np.where(decided3.any(axis=1),
               decided3.argmax(axis=1).astype(float),
               float(ad3))

# 4 conditions: (block_type_id, label, color, linestyle)
conditions3 = [
    (0.0, 'near-cong',   '#4393c3', '-'),
    (1.0, 'near-incong', '#4393c3', '--'),
    (2.0, 'far-cong',    '#d6604d', '-'),
    (3.0, 'far-incong',  '#d6604d', '--'),
]

fig3, (ax3_acc, ax3_rt) = plt.subplots(1, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1]))

for bt, label, color, ls in conditions3:
    mask = bt3_trial == bt
    if mask.any():
        ax3_acc.plot(correct3[mask].mean(axis=0), color=color, linestyle=ls,
                     label=f'{label} (n={mask.sum()})')
ax3_acc.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax3_acc.axvspan(-0.5, stage3_config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax3_acc.set_xticks(range(ad3))
ax3_acc.set_xlabel('Timestep within trial')
ax3_acc.set_ylabel('Accuracy')
ax3_acc.set_ylim(0.3, 1.05)
ax3_acc.legend(fontsize=5)

for bt, label, color, ls in conditions3:
    mask = bt3_trial == bt
    if mask.any():
        cdf = np.array([(rt3[mask] <= t).mean() for t in range(ad3)])
        ax3_rt.plot(range(ad3), cdf, color=color, linestyle=ls,
                    label=f'{label} (n={mask.sum()})')
ax3_rt.axvspan(-0.5, stage3_config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax3_rt.set_xticks(range(ad3))
ax3_rt.set_xlabel('Timestep within trial')
ax3_rt.set_ylabel(f'Fraction decided (|out| > {rt_threshold3})')
ax3_rt.set_ylim(0.0, 1.05)
ax3_rt.legend(fontsize=5)

fig3.tight_layout()
fig3.savefig(stage3_config.export_path + 'flanker_stage3_results.pdf', bbox_inches='tight')
print(f'Exported to: {stage3_config.export_path + "flanker_stage3_results.pdf"}')

#%%
# ── Stage 4: randomized trials, sequential history analysis ───────────────────
#
# Weights frozen, Z updates via LU every trial. Trials are drawn uniformly from
# the 4 types (near/far × cong/incong) with no block structure. The analysis
# groups incongruent trials by the congruency of the preceding 2 trials to test
# the sequential congruency effect (conflict adaptation).
#
# History key: (cong_t-2, cong_t-1) → current incong
#   CC→I  (cong,  cong)  → least adapted, worst expected performance
#   IC→I  (incong, cong) → moderate
#   CI→I  (cong, incong) → moderate
#   II→I  (incong, incong) → most adapted, best expected performance

from configs import FlankerTaskStage4Config

stage4_config = FlankerTaskStage4Config(experiment_to_run='default')
stage4_config.run_name     = 'flanker_stage4_v1'
stage4_config.pre_gating   = config.pre_gating
stage4_config.post_gating  = config.post_gating
stage4_config.use_add_gating = config.use_add_gating
stage4_config.use_mul_gating = config.use_mul_gating

stage4_config.Z_lr = stage2_config.Z_lr
stage4_config.no_of_steps_in_latent_space = 1
model2.config.Z_lr = stage4_config.Z_lr
model2.config.no_of_steps_in_latent_space = 1

print(f'Stage 4: randomized trials | total_trials={stage4_config.n_training_contexts}  '
      f'Z_lr={stage4_config.Z_lr}')

model2.set_Z(torch.randn_like(model2.Z) * 0.2)   # reset Z

logger4, model4, stage4_config, figs4 = train_model(
    stage4_config, seed=stage4_config.env_seed,
    save_models=False, load_models=False,
    pretrained_model=model2,
    run_test_phase=False,
)

#%%
# ── Stage 4 analysis: sequential congruency effect ────────────────────────────
#
# For each incongruent trial, classify the 2 preceding trials as cong/incong.
# Compare accuracy by timestep, RT proxy CDF, and Z center-dim by timestep
# across the 4 history groups.

rt_threshold4 = 0.5
ad4           = stage4_config.arrows_duration

ii4       = np.concatenate(logger4.inputs,            axis=0).reshape(-1, stage4_config.input_size)
oi4       = np.concatenate(logger4.predicted_outputs, axis=0).reshape(-1, stage4_config.output_size)
tt4       = np.concatenate(logger4.hlcids,            axis=0).reshape(-1)
z4_raw    = np.concatenate(logger4.latent_values,     axis=0)   # shape varies — see below

n_trials4    = len(ii4) // ad4
correct4     = (ii4[:n_trials4*ad4, -1] * oi4[:n_trials4*ad4, -1] > 0
               ).astype(float).reshape(n_trials4, ad4)
output4_traj = oi4[:n_trials4*ad4, -1].reshape(n_trials4, ad4)
tt4_trial    = tt4[:n_trials4*ad4].reshape(n_trials4, ad4)[:, 0]

# Z: latent_values is (n_trials, ad4, Z_dim) if stride=ad4; flatten then reshape
z4_flat  = z4_raw.reshape(-1, z4_raw.shape[-1])                        # (total_ts, Z_dim)
z4_trial = z4_flat[:n_trials4 * ad4].reshape(n_trials4, ad4, -1)       # (n_trials, ad4, Z_dim)

is_cong4 = np.isin(tt4_trial, [0, 2])   # True = congruent (types 0 & 2)

# Build history groups: skip first 2 trials (no 2-back available)
history_groups = {(True, True): [], (False, True): [], (True, False): [], (False, False): []}
for t in range(2, n_trials4):
    if not is_cong4[t]:
        key = (bool(is_cong4[t - 2]), bool(is_cong4[t - 1]))
        history_groups[key].append(t)
history_groups = {k: np.array(v) for k, v in history_groups.items()}

decided4 = np.abs(output4_traj) > rt_threshold4
rt4 = np.where(decided4.any(axis=1), decided4.argmax(axis=1).astype(float), float(ad4))

center_dim = 2   # Z index for center slot

history_specs = [
    ((True,  True),  'CC→I', '#084594'),   # both previous cong: least adapted
    ((False, True),  'IC→I', '#4393c3'),
    ((True,  False), 'CI→I', '#d6604d'),
    ((False, False), 'II→I', '#8b0000'),   # both previous incong: most adapted
]

fig4, (ax4_acc, ax4_rt, ax4_z) = plt.subplots(
    1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1])
)

for key, label, color in history_specs:
    idx = history_groups[key]
    if len(idx) == 0:
        continue
    n = len(idx)
    ax4_acc.plot(correct4[idx].mean(axis=0), color=color, label=f'{label} (n={n})')
    cdf = np.array([(rt4[idx] <= t).mean() for t in range(ad4)])
    ax4_rt.plot(range(ad4), cdf,             color=color, label=f'{label} (n={n})')
    ax4_z.plot(z4_trial[idx, :, center_dim].mean(axis=0), color=color, label=label)

for ax in (ax4_acc, ax4_rt, ax4_z):
    ax.axvspan(-0.5, stage4_config.response_start_timestep - 0.5, alpha=0.08, color='k')
    ax.set_xticks(range(ad4))
    ax.set_xlabel('Timestep within trial')
    ax.legend(fontsize=5)

ax4_acc.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax4_acc.set_ylabel('Accuracy')
ax4_acc.set_ylim(0.3, 1.05)
ax4_rt.set_ylabel(f'Fraction decided (|out| > {rt_threshold4})')
ax4_rt.set_ylim(0.0, 1.05)
ax4_z.set_ylabel(f'Z dim {center_dim} (center slot)')

fig4.suptitle('Sequential congruency effect — incongruent trials only', fontsize=7)
fig4.tight_layout()
fig4.savefig(stage4_config.export_path + 'flanker_stage4_history_effects.pdf', bbox_inches='tight')
print(f'Exported to: {stage4_config.export_path + "flanker_stage4_history_effects.pdf"}')

# %%
