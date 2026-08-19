"""
run_flanker_blocked.py — ARCHIVED blocked-design flanker stages.

These were Stage 2 (alternating congruent/incongruent blocks) and Stage 3
(near/far x cong/incong blocks) of the original four-stage script. They are kept so
the figures in exports/flanker_stage2/ and exports/flanker_stage3/ can be
regenerated, but they are no longer part of the pipeline.

Why they were retired
─────────────────────
Blocked presentation confounds condition with time-in-block and with
adaptation-to-switch, so effects measured this way are not comparable to the human
data, which is randomly interleaved. Every effect these stages measured is now
obtained by masking the single random session produced by run_flanker.py Stage 2.
If a blocked result is ever needed, read asymptotic within-block behaviour only —
not the trials right after a switch.

How to run
──────────
Execute the Stage 1 cell of run_flanker.py first, so `config` and `model` exist,
then run the cells below in order.

Two behavioural notes vs. the original script:
  - mirror_to_model now also pushes Z_lr into the live optimizer, so `Z_lr = 0.4`
    below actually takes effect here. In the original runs it did not: the Adam
    optimizer keeps the learning rate it was built with unless Z's shape changes,
    so these stages really ran at Stage 1's Z_lr = 0.3.
  - `('z_start', dim)` is a legacy measure name. Z has no within-trial dynamics, so
    it resolves to the trial's Z; prefer ('z_in', dim) for the inherited state.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

import plot_style
plot_style.set_plot_style()

from plot_style import FigSize
from functions_and_utils import *
from train_and_infer_functions import *
from flanker_analyses import (
    extract_trials, select_trials,
    plot_accuracy_by_timestep, plot_rt_distribution,
    plot_scalar_bars, sync_gating, mirror_to_model,
)

fig_gaussian = False

#%%
# ── Stage 2: frozen weights, self-learned Z ────────────────────────────────────
#
# Weights frozen; Z re-optimized via LU on alternating congruent/incongruent blocks.
# Expected: Z flat for congruent, peaked on center dim for incongruent.

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

# %%
