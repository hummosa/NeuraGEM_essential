"""
run_flanker.py — Flanker task: staged training and analysis.

Stage 1 — Oracle Z (slot identity passed in): trains weights + LU with rotating target slot.
Stage 2 — Frozen weights, self-learned Z on alternating congruent/incongruent blocks.
Stage 3 — Frozen weights, self-learned Z, 4 block types: near/far × cong/incong.
Stage 4 — Frozen weights, randomized trials; sequential trial history analysis.

Analysis utilities live in flanker_analyses.py.
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
from flanker_analyses import (
    extract_trials, select_trials, build_history_groups,
    plot_accuracy_by_timestep, plot_rt_distribution, plot_z_by_timestep,
    plot_scalar_bars, sync_gating, mirror_to_model,
)
fig_gaussian = False   # whether to fit Gaussian to RT distribution in some analyses

# ── Stage 1: configure ────────────────────────────────────────────────────────

config = FlankerTaskConfig(experiment_to_run='default')
config.run_name = 'flanker_pretrain_v1'
config.env_seed = 42

config.blocked_phase_length = 4000 if config.arrow_noise_std <= 0.5 else 10000

gating = 'post'   # 'pre' or 'post'
config.pre_gating  = (gating == 'pre')
config.post_gating = (gating == 'post')

# ── Stage 1: train ────────────────────────────────────────────────────────────

print(f'Stage 1 | seed={config.env_seed}  '
      f'arrows_duration={config.arrows_duration}  '
      f'n_training_contexts={config.n_training_contexts}')
print(f'  temporal_loss_weights={[round(w, 3) for w in config.temporal_loss_weights]}')
print(f'  p_corr_by_distance={config.p_corr_by_distance}')

logger, model, config, figs = train_model(
    config, seed=config.env_seed, save_models=False, load_models=False,
)

panel_order = ['behavior', 'latent_2d', 'corrects']
fig = plot_logger_panels(logger, config, panel_order, x2=None,
                         annotate_phases='behavior', width=5, legends=False, dpi=140)
fig.savefig(config.export_path + 'flanker_pretrain_results.pdf', bbox_inches='tight')
print(f'Exported: {config.export_path + "flanker_pretrain_results.pdf"}')

#%%
# ── Stage 1 analysis: accuracy + RT by training progression and congruency ────
#
# Left:   accuracy by training third (early/mid/late).
# Middle: accuracy split by congruency (Stage 1 hlcids = congruency flag).
# Right:  RT distribution (late trials only) — empirical PMF + Gaussian fit.

rt_threshold = 0.5
trials1  = extract_trials(logger, config, rt_threshold=rt_threshold)
n_trials = trials1['n_trials']
ad       = trials1['ad']
third    = n_trials // 3
late_mask = np.zeros(n_trials, dtype=bool)
late_mask[2 * third:] = True

fig_acc, (ax_thirds, ax_cong, ax_rt) = plt.subplots(
    1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1])
)

# subplot 1: accuracy by training third
colors_thirds = ['#a8c8e8', '#4393c3', '#084594']
for (slc, label), color in zip(
    [(slice(0, third), 'early'), (slice(third, 2*third), 'mid'), (slice(2*third, n_trials), 'late')],
    colors_thirds,
):
    ax_thirds.plot(trials1['correct'][slc].mean(axis=0), color=color, label=label)
ax_thirds.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax_thirds.axvspan(-0.5, config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax_thirds.set_xticks(range(ad))
ax_thirds.set_xlabel('Timestep within trial')
ax_thirds.set_ylabel('Accuracy')
ax_thirds.set_ylim(0.3, 1.05)
ax_thirds.legend(fontsize=5)

# Stage 1: trial_type = hlcids = congruency flag (1.0 = cong, 0.0 = incong)
cong_specs = [
    (trials1['trial_type'] == 1.0, 'congruent',   '#4393c3'),
    (trials1['trial_type'] == 0.0, 'incongruent', '#d6604d'),
]
cong_late_specs = [(m & late_mask, lbl, c) for m, lbl, c in cong_specs]

plot_accuracy_by_timestep(ax_cong, trials1, cong_specs,      config)
plot_rt_distribution(ax_rt,       trials1, cong_late_specs,  config, fit_gaussian=fig_gaussian)

fig_acc.tight_layout()
fig_acc.savefig(config.export_path + 'flanker_accuracy_by_timestep.pdf', bbox_inches='tight')
print(f'Exported: {config.export_path + "flanker_accuracy_by_timestep.pdf"}')

#%%
# ── Stage 2: frozen weights, self-learned Z ────────────────────────────────────
#
# Weights frozen; Z re-optimized via LU on alternating congruent/incongruent blocks.
# Expected: Z flat for congruent, peaked on center dim for incongruent.
#
# model.config vs stage config: model reads Z_lr etc. from model.config at LU time.
# mirror_to_model() patches those attributes so the stage config takes effect.

from configs import FlankerTaskStage2Config

stage2_config = FlankerTaskStage2Config(experiment_to_run='default')
stage2_config.run_name    = 'flanker_stage2_v1'
stage2_config.block_size  = 300
stage2_config.blocked_phase_length = stage2_config.block_size * stage2_config.n_training_contexts
stage2_config.Z_lr                    = 0.4
stage2_config.no_of_steps_in_latent_space = 1
sync_gating(stage2_config, config)
mirror_to_model(model, stage2_config)

print(f'Stage 2 | blocks={stage2_config.n_training_contexts}  '
      f'block_size={stage2_config.block_size}  Z_lr={stage2_config.Z_lr}')

model.set_Z(torch.randn_like(model.Z) * 0.2)

logger2, model2, stage2_config, figs2 = train_model(
    stage2_config, seed=stage2_config.env_seed,
    save_models=False, load_models=False,
    pretrained_model=model,
)

fig2 = plot_logger_panels(logger2, stage2_config, ['latent_2d', 'corrects'], x2=None,
                           width=5, legends=False, dpi=140)
fig2.savefig(stage2_config.export_path + 'flanker_stage2_results.pdf', bbox_inches='tight')
print(f'Exported: {stage2_config.export_path + "flanker_stage2_results.pdf"}')

#%%
# ── Stage 2 analysis: accuracy + RT by congruency ─────────────────────────────

rt_threshold2      = 0.5
accuracy_phase_idx = 0   # 0 = 'Learning and inference' phase

trials2   = extract_trials(logger2, stage2_config, rt_threshold=rt_threshold2)
ad2       = trials2['ad']
n_trials2 = trials2['n_trials']

# Phase boundaries (trial indices)
phase_names  = [p[0] for p in logger2.phases]
phase_starts = [p[1] // ad2 for p in logger2.phases]
phase_ends   = phase_starts[1:] + [n_trials2]
phase_slices = [slice(s, e) for s, e in zip(phase_starts, phase_ends)]

fig2_acc, (ax2_phases, ax2_cong, ax2_rt) = plt.subplots(
    1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1])
)

# subplot 1: accuracy by phase (time-slice, not trial-type)
colors_phases = ['#a8c8e8', '#4393c3', '#084594']
colors_phases = (colors_phases * ((len(phase_slices) // len(colors_phases)) + 1))[:len(phase_slices)]
for slc, name, color in zip(phase_slices, phase_names, colors_phases):
    if trials2['correct'][slc].shape[0] > 0:
        ax2_phases.plot(trials2['correct'][slc].mean(axis=0), color=color, label=name)
ax2_phases.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax2_phases.axvspan(-0.5, stage2_config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax2_phases.set_xticks(range(ad2))
ax2_phases.set_xlabel('Timestep within trial')
ax2_phases.set_ylabel('Accuracy')
ax2_phases.set_ylim(0.3, 1.05)
ax2_phases.legend(fontsize=5)

# subplots 2 & 3: congruency × selected phase
phase_mask2 = np.zeros(n_trials2, dtype=bool)
phase_mask2[phase_slices[accuracy_phase_idx]] = True

cong2_specs = [
    (trials2['trial_type'] == 1.0, 'congruent',   '#4393c3'),
    (trials2['trial_type'] == 0.0, 'incongruent', '#d6604d'),
]
cong2_phase_specs = [(m & phase_mask2, lbl, c) for m, lbl, c in cong2_specs]
print(f'Phase "{phase_names[accuracy_phase_idx]}" | '
      f'congruent n={(trials2["trial_type"] == 1.0).sum()}  '
      f'incongruent n={(trials2["trial_type"] == 0.0).sum()}')

plot_accuracy_by_timestep(ax2_cong, trials2, cong2_phase_specs, stage2_config)
plot_rt_distribution(ax2_rt,       trials2, cong2_phase_specs, stage2_config, fit_gaussian=fig_gaussian)

fig2_acc.tight_layout()
fig2_acc.savefig(stage2_config.export_path + 'flanker_stage2_accuracy_by_timestep.pdf',
                 bbox_inches='tight')
print(f'Exported: {stage2_config.export_path + "flanker_stage2_accuracy_by_timestep.pdf"}')

#%%
# ── Stage 2: RT — correct vs error, split by congruency ──────────────────────

fig2_rt2, (ax_rt_cong, ax_rt_incong) = plt.subplots(
    1, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1]), sharey=True
)
for ax, cong_val, cong_label in [
    (ax_rt_cong,   1.0, 'Congruent'),
    (ax_rt_incong, 0.0, 'Incongruent'),
]:
    base = (trials2['trial_type'] == cong_val) & phase_mask2
    rt_specs_raw = [
        (base & trials2['is_correct'],  'correct', '#4393c3', '-'),
        (base & ~trials2['is_correct'], 'wrong',   '#d6604d', '--'),
    ]
    plot_rt_distribution(ax, trials2,
                         [(m, lbl, c) for m, lbl, c, _ in rt_specs_raw],
                         stage2_config, fit_gaussian=fig_gaussian,
                         linestyles=[ls for _, _, _, ls in rt_specs_raw])
    ax.set_title(cong_label, fontsize=7)

fig2_rt2.tight_layout()
fig2_rt2.savefig(stage2_config.export_path + 'flanker_stage2_rt_correct_vs_wrong.pdf',
                 bbox_inches='tight')
print(f'Exported: {stage2_config.export_path + "flanker_stage2_rt_correct_vs_wrong.pdf"}')

#%%
# ── Stage 3: distance × congruency flanker effect ─────────────────────────────
#
# 4 block types: near/far × cong/incong. Weights frozen; Z self-organized via LU.
# Key prediction: near flanker effect > far flanker effect (interaction).

from configs import FlankerTaskStage3Config

stage3_config = FlankerTaskStage3Config(experiment_to_run='default')
stage3_config.run_name    = 'flanker_stage3_v1'
stage3_config.block_size  = 300
stage3_config.blocked_phase_length = stage3_config.block_size * stage3_config.n_training_contexts
stage3_config.Z_lr                    = stage2_config.Z_lr
stage3_config.no_of_steps_in_latent_space = 1
sync_gating(stage3_config, config)
mirror_to_model(model2, stage3_config)

print(f'Stage 3 | blocks={stage3_config.n_training_contexts}  block_size={stage3_config.block_size}')

model2.set_Z(torch.randn_like(model2.Z) * 0.2)

logger3, model3, stage3_config, figs3 = train_model(
    stage3_config, seed=stage3_config.env_seed,
    save_models=False, load_models=False,
    pretrained_model=model2,
    run_test_phase=False,
)

fig3_panels = plot_logger_panels(logger3, stage3_config,
                                  ['behavior', 'latent_2d', 'corrects'], x2=None,
                                  annotate_phases='corrects', width=5, legends=False, dpi=140)
# Annotate first full cycle of block types on the latent_2d subplot
ax_z3 = fig3_panels.axes[1]
bs3 = stage3_config.block_size
for i, lbl in enumerate(['Near\nCong', 'Near\nIncong', 'Far\nCong', 'Far\nIncong']):
    ax_z3.text(bs3 * i + bs3 / 2, 1.0, lbl,
               transform=ax_z3.get_xaxis_transform(), ha='center', va='top', fontsize=5)

fig3_panels.savefig(stage3_config.export_path + 'flanker_stage3_panels.pdf', bbox_inches='tight')
print(f'Exported: {stage3_config.export_path + "flanker_stage3_panels.pdf"}')

#%%
# ── Stage 3 analysis: 2×2 accuracy + RT ──────────────────────────────────────

trials3 = extract_trials(logger3, stage3_config)

conditions3 = [
    (0.0, 'near-cong',   '#4393c3', '-'),
    (1.0, 'near-incong', '#4393c3', '--'),
    (2.0, 'far-cong',    '#d6604d', '-'),
    (3.0, 'far-incong',  '#d6604d', '--'),
]
specs3 = [(select_trials(trials3, trial_type=bt), lbl, c) for bt, lbl, c, _ in conditions3]
ls3    = [ls for _, _, _, ls in conditions3]

fig3, (ax3_acc, ax3_rt) = plt.subplots(1, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1]))
plot_accuracy_by_timestep(ax3_acc, trials3, specs3, stage3_config, linestyles=ls3)
plot_rt_distribution(ax3_rt,      trials3, specs3, stage3_config,
                     fit_gaussian=fig_gaussian, linestyles=ls3)

fig3.tight_layout()
fig3.savefig(stage3_config.export_path + 'flanker_stage3_results.pdf', bbox_inches='tight')
print(f'Exported: {stage3_config.export_path + "flanker_stage3_results.pdf"}')

#%%
# ── Stage 4: config.p_congruent fraction with randomized trials, sequential history analysis ───────────────────
#
# Weights frozen, Z updates via LU every trial. Trials drawn uniformly from 4 types.
# History key (cong_t-2, cong_t-1) → current incong:
#   CC→I  least adapted | II→I  most adapted (conflict adaptation prediction)

from configs import FlankerTaskStage4Config

stage4_config = FlankerTaskStage4Config(experiment_to_run='default')
stage4_config.run_name     = 'flanker_stage4_v1'
stage4_config.p_congruent  = 0.2   # fraction of congruent trials (0.5 = equal; try 0.7 for more congruent)
# stage4_config.Z_lr                    = 0.2
stage4_config.no_of_steps_in_latent_space = 1
sync_gating(stage4_config, config)
mirror_to_model(model2, stage4_config)

print(f'Stage 4 | total_trials={stage4_config.n_training_contexts}  '
      f'p_congruent={stage4_config.p_congruent}  Z_lr={stage4_config.Z_lr}')

model2.set_Z(torch.randn_like(model2.Z) * 0.2)

logger4, model4, stage4_config, figs4 = train_model(
    stage4_config, seed=stage4_config.env_seed,
    save_models=False, load_models=False,
    pretrained_model=model2,
    run_test_phase=False,
)

plot_logger_panels(logger4, stage4_config,
                   ['latent_2d', 'corrects'], x2=None,
                   annotate_phases='corrects', width=5, legends=False, dpi=300,)


# ── Stage 4 analysis: sequential congruency effect ────────────────────────────

trials4     = extract_trials(logger4, stage4_config)
incong_mask = select_trials(trials4, trial_type=[1, 3])          # near-incong or far-incong
groups4     = build_history_groups(trials4, n_back=2,
                                   current_mask=incong_mask,
                                   congruent_types=(0, 2))

center_dim = 2   # Z dimension for center slot
history_specs4 = [
    ((True,  True),  'CC→I', '#084594'),   # both prev cong  — least adapted
    ((True,  False), 'CI→I', '#d6604d'),
    ((False, False), 'II→I', '#8b0000'),   # both prev incong — most adapted
]
specs4 = [(groups4[key], lbl, c) for key, lbl, c in history_specs4 if groups4[key].any()]

fig4, (ax4_acc, ax4_rt, ax4_z) = plt.subplots(
    1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1])
)
plot_accuracy_by_timestep(ax4_acc, trials4, specs4, stage4_config)
plot_rt_distribution(ax4_rt,       trials4, specs4, stage4_config, fit_gaussian=fig_gaussian)
plot_z_by_timestep(ax4_z,          trials4, specs4, z_dim=center_dim, config=stage4_config)

fig4.tight_layout()
fig4.savefig(stage4_config.export_path + 'flanker_stage4_history_effects.pdf', bbox_inches='tight')
print(f'Exported: {stage4_config.export_path + "flanker_stage4_history_effects.pdf"}')

# ── Stage 4: sequential congruency effect — bar summary (mean ± SEM) ─────────

fig4b, (ax4b_acc, ax4b_rt, ax4b_z) = plt.subplots(
    1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1])
)
plot_scalar_bars(ax4b_acc, trials4, specs4, measure='accuracy')
plot_scalar_bars(ax4b_rt,  trials4, specs4, measure='rt')
plot_scalar_bars(ax4b_z,   trials4, specs4, measure=('z_start', center_dim))

ax4b_acc.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)

fig4b.suptitle('Sequential congruency effect — mean ± SEM', fontsize=7)
fig4b.tight_layout()
fig4b.savefig(stage4_config.export_path + 'flanker_stage4_history_effects_bars.pdf', bbox_inches='tight')
print(f'Exported: {stage4_config.export_path + "flanker_stage4_history_effects_bars.pdf"}')


# ── Stage 4 analysis: evidence accumulation dynamics ─────────────────────────
#
# Output is sign-normalised by the model's own final-timestep decision.
# Positive = accumulating toward whatever the model eventually chose;
# negative = initially going against the eventual decision.
# This reveals the internal dynamics of commitment independent of left/right.
#
# 4 conditions: correct-cong, error-cong, correct-incong, error-incong.
# Key patterns to look for:
#   - Incongruent trials should show an early dip (flanker pulls away from final choice)
#   - Error trials may show a slow or non-monotonic rise (weak commitment)

cong_mask4   = np.isin(trials4['trial_type'], [0, 2])
incong_mask4 = np.isin(trials4['trial_type'], [1, 3])
corr_mask4   = trials4['is_correct']
err_mask4    = ~trials4['is_correct']

accum_specs = [
    (corr_mask4 & cong_mask4,   'Correct, cong',   '#084594', '-'),
    (err_mask4  & cong_mask4,   'Error, cong',     '#4393c3', '--'),
    (corr_mask4 & incong_mask4, 'Correct, incong', '#8b0000', '-'),
    (err_mask4  & incong_mask4, 'Error, incong',   '#d6604d', '--'),
]

fig4c, ax4c = plt.subplots(figsize=(FigSize.large[0], FigSize.large[1]))

for mask, label, color, ls in accum_specs:
    if mask.any():
        mean_traj = trials4['signed_output'][mask].mean(axis=0)
        ax4c.plot(mean_traj, color=color, linestyle=ls,
                  label=f'{label} (n={mask.sum()})')

ax4c.axhline(0,  color='k', linewidth=0.8, linestyle='-',  alpha=0.25)
ax4c.axhline( 1, color='k', linewidth=0.5, linestyle='--', alpha=0.2)
ax4c.axhline(-1, color='k', linewidth=0.5, linestyle='--', alpha=0.2)
ax4c.axvspan(-0.5, stage4_config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax4c.set_xticks(range(trials4['ad']))
ax4c.set_xlabel('Timestep within trial')
ax4c.set_ylabel('Output toward final decision\n(sign-normalised)')
ax4c.set_ylim(-1.1, 1.1)
ax4c.legend(fontsize=5)
ax4c.spines['top'].set_visible(False)
ax4c.spines['right'].set_visible(False)

fig4c.tight_layout()
fig4c.savefig(stage4_config.export_path + 'flanker_stage4_accumulation.pdf', bbox_inches='tight')
print(f'Exported: {stage4_config.export_path + "flanker_stage4_accumulation.pdf"}')


# ── Stage 4 analysis: post-error effects ──────────────────────────────────────
#
# Trial A: the preceding trial. Trial B: the trial whose performance we measure.
# 3 conditions defined by A, trial B pooled across congruency:
#   A = error on near-flanker trial  (types 0 or 1)
#   A = error on far-flanker trial   (types 2 or 3)
#   A = correct (any type)           — comparison baseline
#
# Hypothesis: post-near-error > post-far-error effect, because near flankers
# carry more prediction weight so errors there cause a larger Z correction.

# One-trial lookback via np.roll; index 0 zeroed to prevent end-wraparound
prev_is_error   = np.roll(~trials4['is_correct'], 1);  prev_is_error[0]   = False
prev_trial_type = np.roll(trials4['trial_type'],  1);  prev_trial_type[0] = -1
prev_correct    = np.roll(trials4['is_correct'],  1);  prev_correct[0]    = False

post_error_specs = [
    (prev_is_error & np.isin(prev_trial_type, [0, 1]), 'Post near-flanker error', '#d6604d', '-'),
    (prev_is_error & np.isin(prev_trial_type, [2, 3]), 'Post far-flanker error',  '#f4a582', '-'),
    (prev_correct,                                      'Post correct trial',      '#4393c3', '-'),
]

specs5 = [(m, lbl, c) for m, lbl, c, _ in post_error_specs]
ls5    = [ls for _, _, _, ls in post_error_specs]

# ── Line plots: temporal dynamics within trial B ──────────────────────────────
fig5, (ax5_acc, ax5_rt, ax5_z) = plt.subplots(
    1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1])
)
plot_accuracy_by_timestep(ax5_acc, trials4, specs5, stage4_config)
plot_rt_distribution(ax5_rt,       trials4, specs5, stage4_config, fit_gaussian=fig_gaussian)
plot_z_by_timestep(ax5_z,          trials4, specs5, z_dim=center_dim, config=stage4_config)

fig5.suptitle('Post-error effects: trial B dynamics by trial A type', fontsize=7)
fig5.tight_layout()
fig5.savefig(stage4_config.export_path + 'flanker_stage4_post_error.pdf', bbox_inches='tight')
print(f'Exported: {stage4_config.export_path + "flanker_stage4_post_error.pdf"}')


# ── Stage 4: post-error bar summary (mean ± SEM) ──────────────────────────────
# Accuracy at RT-crossing, RT, and Z at end of trial B — 3 conditions.

fig5b, (ax5b_acc, ax5b_rt, ax5b_z) = plt.subplots(
    1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1]), dpi=200
)
plot_scalar_bars(ax5b_acc, trials4, specs5, measure='accuracy')
plot_scalar_bars(ax5b_rt,  trials4, specs5, measure='rt')
plot_scalar_bars(ax5b_z,   trials4, specs5, measure=('z_start', center_dim))

ax5b_acc.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)

fig5b.suptitle('Post-error effects: mean ± SEM', fontsize=7)
fig5b.tight_layout()
fig5b.savefig(stage4_config.export_path + 'flanker_stage4_post_error_bars.pdf', bbox_inches='tight')
print(f'Exported: {stage4_config.export_path + "flanker_stage4_post_error_bars.pdf"}')

#%%
# ── Diagnostic: post-error split by trial B congruency ────────────────────────
#
# The pooled-B analysis above conflates two opposite effects:
#   - center-slot Z focus (from error LU update) HELPS incongruent B
#   - center-slot Z focus HURTS congruent B (useful flanker info ignored)
#
# Expected if the model shows conflict adaptation:
#   near-err → incong B  >  correct → incong B   (better, conflict-adapted)
#   near-err → cong B    <  correct → cong B      (worse, over-focused)
# If this crossover exists, the pooled result is misleading.
#
# 'prev_correct' baseline is also dominated by post-congruent-correct trials
# (most correct trials are congruent), so the comparison is really:
#   post-near-incong-error  vs.  post-congruent-correct
# Splitting B by congruency makes this explicit.

b_cong   = np.isin(trials4['trial_type'], [0, 2])
b_incong = np.isin(trials4['trial_type'], [1, 3])
near_err = prev_is_error & np.isin(prev_trial_type, [0, 1])
far_err  = prev_is_error & np.isin(prev_trial_type, [2, 3])

diag_specs = [
    (near_err  & b_incong, 'near-err, incong B', '#8b0000'),
    (near_err  & b_cong,   'near-err, cong B',   '#f4a582'),
    (far_err   & b_incong, 'far-err, incong B',  '#74add1'),
    (far_err   & b_cong,   'far-err, cong B',    '#abd9e9'),
    (prev_correct & b_incong, 'correct, incong B','#d6604d', ),
    (prev_correct & b_cong,   'correct, cong B',  '#4393c3', ),
]
diag_specs3 = [(m, lbl, c) for m, lbl, c in diag_specs]

# Print counts
for m, lbl, c in diag_specs3:
    print(f'  {lbl}: n={m.sum()}')

fig5c, (ax5c_acc, ax5c_rt, ax5c_z) = plt.subplots(
    1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1]), dpi=200
)
plot_scalar_bars(ax5c_acc, trials4, diag_specs3, measure='accuracy',  group_spacing=[2, 4])
plot_scalar_bars(ax5c_rt,  trials4, diag_specs3, measure='rt',        group_spacing=[2, 4])
plot_scalar_bars(ax5c_z,   trials4, diag_specs3, measure=('z_start', center_dim), group_spacing=[2, 4])

ax5c_acc.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)

fig5c.suptitle('Diagnostic: post-error × trial-B congruency (near-err | far-err | baseline)', fontsize=7)
fig5c.tight_layout()
fig5c.savefig(stage4_config.export_path + 'flanker_stage4_post_error_diagnostic.pdf', bbox_inches='tight')
print(f'Exported: {stage4_config.export_path + "flanker_stage4_post_error_diagnostic.pdf"}')

#%%
# ── Diagnostic v2: condition on trial-A congruency ────────────────────────────
#
# The v1 diagnostic pools congruent and incongruent errors into "near-err" /
# "far-err", making the correct baseline inhomogeneous (dominated by congruent
# correct trials). Here we produce two separate figures:
#
#   Figure 1 — congruent A:   near-cong-err | far-cong-err | cong-correct
#   Figure 2 — incongruent A: near-incong-err | far-incong-err | incong-correct
#
# Each figure keeps the trial-B split (cong B / incong B) so the 2×3 structure
# is preserved, but now every column is a homogeneous A type.
#
# Reads: b_cong, b_incong, prev_is_error, prev_trial_type, prev_correct (from above)

prev_cong_A   = np.isin(prev_trial_type, [0, 2])
prev_incong_A = np.isin(prev_trial_type, [1, 3])

for a_label, near_types, far_types in [
    ('congruent_A',   [0], [2]),
    ('incongruent_A', [1], [3]),
]:
    near_err_A = prev_is_error & np.isin(prev_trial_type, near_types)
    far_err_A  = prev_is_error & np.isin(prev_trial_type, far_types)
    correct_A  = prev_correct  & np.isin(prev_trial_type, near_types + far_types)

    specsV2 = [
        (near_err_A & b_incong, 'near-err → incong B', '#8b0000'),
        (near_err_A & b_cong,   'near-err → cong B',   '#f4a582'),
        (far_err_A  & b_incong, 'far-err → incong B',  '#74add1'),
        (far_err_A  & b_cong,   'far-err → cong B',    '#abd9e9'),
        (correct_A  & b_incong, 'correct → incong B',  '#d6604d'),
        (correct_A  & b_cong,   'correct → cong B',    '#4393c3'),
    ]

    print(f'\n{a_label}:')
    for m, lbl, c in specsV2:
        print(f'  {lbl}: n={m.sum()}')

    figV2, (axV2_acc, axV2_rt, axV2_z) = plt.subplots(
        1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1]), dpi=200
    )
    plot_scalar_bars(axV2_acc, trials4, specsV2, measure='accuracy',              group_spacing=[2, 4])
    plot_scalar_bars(axV2_rt,  trials4, specsV2, measure='rt',                    group_spacing=[2, 4])
    plot_scalar_bars(axV2_z,   trials4, specsV2, measure=('z_start', center_dim), group_spacing=[2, 4])

    axV2_acc.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
    title = a_label.replace('_', ' ').title()
    figV2.suptitle(f'Post-error × B congruency | {title}  (near-err | far-err | correct)', fontsize=7)
    figV2.tight_layout()
    fname = f'flanker_stage4_post_error_diag_{a_label}.pdf'
    figV2.savefig(stage4_config.export_path + fname, bbox_inches='tight')
    print(f'Exported: {stage4_config.export_path + fname}')

# %%
