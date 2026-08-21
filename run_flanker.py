"""
run_flanker.py — Flanker task: training, then one randomly interleaved test session.

Stage 1 — Training. Oracle Z: the identity of the target slot is supplied to the
          model as a one-hot latent, and the weights learn to use it. The target
          slot rotates across short blocks so a control setting is learned for
          every slot. Nothing here is analysed.
Stage 2 — Test. Weights frozen. Z is now inferred online by gradient descent on
          prediction error, one update per trial, and is the only thing that
          adapts. Trials are drawn i.i.d. from the four near/far x cong/incong
          types — the randomly interleaved design human participants perform.

Every result comes from Stage 2, by masking one random session. Blocked designs
confound condition with time-in-block and with adaptation-to-switch, so they are no
longer used; the old blocked stages live in archive/run_flanker_blocked.py.

Three analysis conventions, all enforced in flanker_analyses.py:
  - RT is the threshold crossing: interpolated inside the response window, and
    extrapolated (least-squares projection, capped at 10 timesteps) for trials that
    never cross. No trial is dropped, because failing to cross is ~4x more common on
    incongruent trials and censoring them biased incongruent RT downward. About half
    the extrapolated trials are not rising at all and land on the cap, so RT partly
    encodes failure-to-decide — `report_extraction` prints the extrapolated fraction.
  - Z is reported activated (softmax across slots) via the scalar `focus` index.
  - `focus_in` is the state a trial *inherited*; `delta_focus` is the update it
    produced. Z is logged after the LU step, so a trial's own Z already contains
    its own update and must not be read as the state it started from.

**This is one session, i.e. one synthetic subject.** Several effects that looked clear
in a single 5000-trial session did not survive ten seeds, and one reversed sign. Use
these figures to see mechanism and shape; take the statistics from the group scripts:

    python flanker_sweep_analysis.py                                 # across-seed tables
    python flanker_model_figures.py --variant spatial_steep --p 0.5  # model card

Analysis utilities live in flanker_analyses.py; per-seed measures in flanker_metrics.py.
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

from plot_style import FigSize
from functions_and_utils import *
from configs import FlankerTaskConfig, FlankerRandomTrialsConfig
from train_and_infer_functions import *
from flanker_analyses import (
    extract_trials, select_trials, lagged_factors, decode_trial_types,
    report_extraction, print_cell_counts,
    plot_accuracy_by_timestep, plot_rt_distribution, plot_rt_continuous,
    plot_scalar_bars, sync_gating, mirror_to_model, reset_Z_uniform,
)

fig_gaussian = False   # whether to overlay a Gaussian fit on RT PMFs

# Result 6 (trial-history regression). False prints the ~6 rows that are results — the
# human signatures and the mediation. True adds every model's full coefficient table and
# the collinearity check, including the nuisance regressors that are in the design to be
# controlled for rather than read. Turn it on when auditing a fit or comparing against
# the human coefficients in code_shared/STA_BH.
regression_detail = False

# Congruency drives colour, distance drives line style, throughout.
COL = dict(
    cong    = '#4393c3',
    incong  = '#d6604d',
    correct = '#4393c3',
    error   = '#8b0000',
    neutral = '#777777',
)


def learning_curve(acc, mask=None, n_bins=20):
    """Accuracy in equal-width bins of trial position. Returns (x, y)."""
    n = len(acc)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    x, y = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = np.zeros(n, dtype=bool)
        sel[a:b] = True
        if mask is not None:
            sel = sel & mask
        if sel.any():
            x.append(0.5 * (a + b))
            y.append(acc[sel].mean())
    return np.array(x), np.array(y)

# ── Stage 1: configure ────────────────────────────────────────────────────────

config = FlankerTaskConfig(experiment_to_run='default')
config.run_name = 'flanker_pretrain_v1'
config.env_seed = 42

config.Z_optimizer = 'SGD'
# Session length is specified in trials, not timesteps, so it stays correct if
# arrows_duration ever changes.
n_pretrain_trials = 2000
config.blocked_phase_length = n_pretrain_trials * config.arrows_duration

gating = 'post'   # 'pre' or 'post'
config.pre_gating  = (gating == 'pre')
config.post_gating = (gating == 'post')

# ── Stage 1: train ────────────────────────────────────────────────────────────

print(f'Stage 1 | seed={config.env_seed}  trials={n_pretrain_trials}  '
      f'arrows_duration={config.arrows_duration}')
print(f'  temporal_loss_weights={[round(w, 3) for w in config.temporal_loss_weights]}')
print(f'  p_corr_by_distance={config.p_corr_by_distance}')

logger, model, config, figs = train_model(
    config, seed=config.env_seed,
    save_models=True, load_models=False,
    run_test_phase=False,
)

panel_order = [ 'latent_2d', 'corrects']
_=fig = plot_logger_panels(logger, config, panel_order, x2=None,
                         annotate_phases='corrects', width=5, legends=False, dpi=140)
fig.savefig(config.export_path + 'flanker_pretrain_results.pdf', bbox_inches='tight')
print(f'Exported: {config.export_path + "flanker_pretrain_results.pdf"}')

#%%
# ── Stage 1 sanity check: did training work? ──────────────────────────────────
#
# 1: learning curve — accuracy against training trial, split by congruency. Watch
#    whether the incongruent curve has actually plateaued before the weights are
#    frozen; if it is still climbing, the spatial weighting is not yet settled.
# 2: accuracy by training third (early/mid/late), across timesteps within a trial.
# 3: accuracy split by congruency (Stage 1 hlcids = congruency flag).
# 4: RT distribution, late trials only.
#
# This is a training diagnostic, not a result — Stage 1 is blocked and oracle-driven.

trials1  = extract_trials(logger, config, rt_threshold=0.5)
report_extraction(trials1, label='Stage 1 | ')

n_trials = trials1['n_trials']
third    = n_trials // 3
late_mask = np.zeros(n_trials, dtype=bool)
late_mask[2 * third:] = True

fig_acc, (ax_curve, ax_thirds, ax_cong, ax_rt) = plt.subplots(
    1, 4, figsize=(FigSize.large[0] * 4, FigSize.large[1])
)

acc1 = trials1['correct_at_decision'].astype(float)
for mask1, label1, color1 in [(trials1['trial_type'] == 1.0, 'congruent',   COL['cong']),
                              (trials1['trial_type'] == 0.0, 'incongruent', COL['incong']),
                              (None,                          'all trials',  'k')]:
    xs, ys = learning_curve(acc1, mask1)
    ax_curve.plot(xs, ys, color=color1, label=label1)
ax_curve.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax_curve.set_xlabel('Training trial')
ax_curve.set_ylabel('Accuracy')
ax_curve.legend(fontsize=5)
ax_curve.set_title('Learning', fontsize=6)

colors_thirds = ['#a8c8e8', '#4393c3', '#084594']
for (slc, label), color in zip(
    [(slice(0, third), 'early'), (slice(third, 2*third), 'mid'), (slice(2*third, n_trials), 'late')],
    colors_thirds,
):
    ax_thirds.plot(trials1['correct'][slc].mean(axis=0), color=color, label=label)
ax_thirds.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax_thirds.axvspan(-0.5, config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax_thirds.set_xticks(range(trials1['ad']))
ax_thirds.set_xlabel('Timestep within trial')
ax_thirds.set_ylabel('Accuracy')
ax_thirds.set_ylim(0.3, 1.05)
ax_thirds.legend(fontsize=5)

# Stage 1: trial_type = hlcids = congruency flag (1.0 = cong, 0.0 = incong)
cong_specs = [
    (trials1['trial_type'] == 1.0, 'congruent',   COL['cong']),
    (trials1['trial_type'] == 0.0, 'incongruent', COL['incong']),
]
cong_late_specs = [(m & late_mask, lbl, c) for m, lbl, c in cong_specs]

plot_accuracy_by_timestep(ax_cong, trials1, cong_specs,     config)
plot_rt_distribution(ax_rt,       trials1, cong_late_specs, config, fit_gaussian=fig_gaussian)

fig_acc.tight_layout()
fig_acc.savefig(config.export_path + 'flanker_pretrain_sanity.pdf', bbox_inches='tight')
print(f'Exported: {config.export_path + "flanker_pretrain_sanity.pdf"}')

#%%
# ── Stage 2: frozen weights, self-inferred Z, randomly interleaved trials ─────
#
# model.config vs stage config: the model reads Z_lr and no_of_steps_in_latent_space
# from model.config at LU time, and Z_lr is additionally baked into the Adam
# optimizer at construction. mirror_to_model() patches both. sync_gating() carries
# over the runtime gating choice, which a freshly-constructed config would revert.

test_config = FlankerRandomTrialsConfig(experiment_to_run='default')
test_config.run_name = 'flanker_random_v1'
test_config.env_seed = config.env_seed
test_config.set_n_trials(5000)
test_config.p_congruent = 0.5          # matches the human task
test_config.no_of_steps_in_latent_space = 1

sync_gating(test_config, config)
mirror_to_model(model, test_config)
reset_Z_uniform(model, scale=0.2, seed=test_config.env_seed)

print(f'Stage 2 | trials={test_config.n_trials}  p_congruent={test_config.p_congruent}  '
      f'p_near={test_config.p_near}  Z_lr={test_config.Z_lr}')

logger_t, model_t, test_config, figs_t = train_model(
    test_config, seed=test_config.env_seed,
    save_models=False, load_models=False,
    pretrained_model=model,
    run_test_phase=False,
)

plot_logger_panels(logger_t, test_config, ['latent_2d', 'corrects'], x2=None,
                   annotate_phases='corrects', width=5, legends=False, dpi=140)

#%%
# ── Stage 2: extract trials and build the condition masks ─────────────────────

trials = extract_trials(logger_t, test_config, rt_threshold=0.5)
report_extraction(trials, label='Stage 2 | ')

f = lagged_factors(trials, n_back=2)

cong,   incong = f['cong'] == 1,   f['cong'] == 0
near,   far    = f['near'] == 1,   f['near'] == 0
valid          = f['valid']

prev_cong,   prev_incong  = f['cong_1'] == 1,    f['cong_1'] == 0
prev_near,   prev_far     = f['near_1'] == 1,    f['near_1'] == 0
prev_correct, prev_error  = f['correct_1'] == 1, f['correct_1'] == 0
prev2_cong,  prev2_incong = f['cong_2'] == 1,    f['cong_2'] == 0
prev2_correct             = f['correct_2'] == 1
repeated, switched        = f['resp_rep'] == 1,  f['resp_rep'] == 0

# Response actually emitted, i.e. evaluated at the threshold crossing
correct_dec = trials['correct_at_decision']
error_dec   = ~correct_dec

print(f'  congruent n={cong.sum()}  incongruent n={incong.sum()}  '
      f'near n={near.sum()}  far n={far.sum()}')
print(f'  errors: congruent {(~trials["correct_at_decision"] & cong).sum()}  '
      f'incongruent {(~trials["correct_at_decision"] & incong).sum()}')

#%%
# ── Result 1: congruency × distance ───────────────────────────────────────────
#
# The headline behavioural fingerprint, now from interleaved trials rather than
# blocks. Prediction: incongruent is slower and less accurate, and the cost is
# larger for near flankers than far ones.

conditions = [
    (near & cong,   'near-cong',   COL['cong'],   '-'),
    (near & incong, 'near-incong', COL['incong'], '-'),
    (far  & cong,   'far-cong',    COL['cong'],   '--'),
    (far  & incong, 'far-incong',  COL['incong'], '--'),
]
specs_cd = [(m, lbl, c) for m, lbl, c, _ in conditions]
ls_cd    = [ls for _, _, _, ls in conditions]
print_cell_counts(specs_cd, label='Congruency × distance:')

fig1, (ax1a, ax1b, ax1c) = plt.subplots(1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1]))
plot_accuracy_by_timestep(ax1a, trials, specs_cd, test_config, linestyles=ls_cd)
plot_rt_continuous(ax1b,        trials, specs_cd, test_config, linestyles=ls_cd)
plot_rt_distribution(ax1c,      trials, specs_cd, test_config,
                     fit_gaussian=fig_gaussian, linestyles=ls_cd)
fig1.tight_layout()
fig1.savefig(test_config.export_path + 'flanker_congruency_distance.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_congruency_distance.pdf"}')

# Z panel uses delta_focus, not focus. Trials are i.i.d., so what a trial *inherits*
# (focus_in) cannot differ by its own condition — that panel is flat by construction.
# The trial's own `focus` is logged after its update, so grouping it by condition shows
# the update mixed into the mean level; delta_focus shows the update on its own.
fig1b, (ax1d, ax1e, ax1f) = plt.subplots(1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1]))
plot_scalar_bars(ax1d, trials, specs_cd, measure='accuracy',  baseline=0.5)
plot_scalar_bars(ax1e, trials, specs_cd, measure='rt_interp')
plot_scalar_bars(ax1f, trials, specs_cd, measure='delta_focus', baseline=0.0)
fig1b.suptitle('Congruency × distance — mean ± SEM', fontsize=7)
fig1b.tight_layout()
fig1b.savefig(test_config.export_path + 'flanker_congruency_distance_bars.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_congruency_distance_bars.pdf"}')

#%%
# ── Result 1b: session stability and RT by outcome ────────────────────────────
#
# 1: accuracy across the test session. Weights are frozen, so any drift here is Z
#    settling into the statistics of the list — it should be flat after the first
#    few trials.
# 2 & 3: RT distributions for correct vs error responses, within each congruency.
#    Errors should be fast on incongruent trials if they are flanker-driven.

fig1c, (ax1c_curve, ax1c_cong, ax1c_incong) = plt.subplots(
    1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1])
)

acc2 = correct_dec.astype(float)
for mask2, label2, color2 in [(cong, 'congruent', COL['cong']),
                              (incong, 'incongruent', COL['incong']),
                              (None, 'all trials', 'k')]:
    xs, ys = learning_curve(acc2, mask2)
    ax1c_curve.plot(xs, ys, color=color2, label=label2)
ax1c_curve.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
ax1c_curve.set_xlabel('Test trial')
ax1c_curve.set_ylabel('Accuracy')
ax1c_curve.legend(fontsize=5)
ax1c_curve.set_title('Frozen weights: session stability', fontsize=6)

for ax, base, title in [(ax1c_cong, cong, 'Congruent'), (ax1c_incong, incong, 'Incongruent')]:
    specs_out = [(base & correct_dec, 'correct', COL['correct']),
                 (base & error_dec,   'error',   COL['error'])]
    plot_rt_continuous(ax, trials, specs_out, test_config, linestyles=['-', '--'])
    ax.set_title(title, fontsize=6)

fig1c.tight_layout()
fig1c.savefig(test_config.export_path + 'flanker_session_and_rt_by_outcome.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_session_and_rt_by_outcome.pdf"}')

#%%
# ── Result 2: within-trial evidence accumulation ──────────────────────────────
#
# Output is sign-normalised by the model's own final decision, so left and right
# decisions align on one axis. Positive = accumulating toward whatever the model
# eventually chose. This is the model's analogue of the human beta-power
# lateralisation signal: incongruent trials should dip early (flankers pull away
# from the eventual choice) and reverse late on correct trials.

# Colour = congruency, line style = outcome, as everywhere else in this script.
accum_specs = [
    (correct_dec  & cong,   'Correct, cong',   COL['cong'],   '-'),
    (~correct_dec & cong,   'Error, cong',     COL['cong'],   '--'),
    (correct_dec  & incong, 'Correct, incong', COL['incong'], '-'),
    (~correct_dec & incong, 'Error, incong',   COL['incong'], '--'),
]

fig2, ax2 = plt.subplots(figsize=(FigSize.large[0], FigSize.large[1]))
for mask, label, color, ls in accum_specs:
    if mask.any():
        ax2.plot(trials['signed_output'][mask].mean(axis=0), color=color, linestyle=ls,
                 label=f'{label} (n={mask.sum()})')
ax2.axhline(0,  color='k', linewidth=0.8, linestyle='-',  alpha=0.25)
ax2.axhline( 1, color='k', linewidth=0.5, linestyle='--', alpha=0.2)
ax2.axhline(-1, color='k', linewidth=0.5, linestyle='--', alpha=0.2)
ax2.axvspan(-0.5, test_config.response_start_timestep - 0.5, alpha=0.08, color='k')
ax2.set_xticks(range(trials['ad']))
ax2.set_xlabel('Timestep within trial')
ax2.set_ylabel('Output toward final decision\n(sign-normalised)')
ax2.set_ylim(-1.1, 1.1)
ax2.legend(fontsize=5)
fig2.tight_layout()
fig2.savefig(test_config.export_path + 'flanker_accumulation.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_accumulation.pdf"}')

#%%
# ── Result 3: sequential congruency, all four history cells ───────────────────
#
# Two fixes over the previous version:
#   - CI and IC are separated. They were conflated before, but "congruent then
#     incongruent" and "incongruent then congruent" are opposite trajectories.
#   - Restricted to trials where BOTH preceding trials were correct. Incongruent
#     trials fail more often, so an unrestricted history cell is partly measuring
#     post-error adaptation rather than conflict adaptation.
#
# Both → I and → C targets are shown. II→C vs CC→C is where the predicted
# speed-accuracy dissociation lives (faster but more error-prone), so read the
# accuracy and RT panels together.

hist_ok = valid & prev_correct & prev2_correct

def history_cell(c2, c1, current):
    """Mask for (t-2, t-1) -> t congruency pattern; True = congruent."""
    m = hist_ok.copy()
    m &= (prev2_cong if c2 else prev2_incong)
    m &= (prev_cong  if c1 else prev_incong)
    m &= (cong       if current else incong)
    return m

hist_cells = [(True, True, 'CC'), (True, False, 'CI'),
              (False, True, 'IC'), (False, False, 'II')]

specs_hist = (
    [(history_cell(c2, c1, False), f'{lbl}→I', COL['incong']) for c2, c1, lbl in hist_cells]
    + [(history_cell(c2, c1, True), f'{lbl}→C', COL['cong']) for c2, c1, lbl in hist_cells]
)
print_cell_counts(specs_hist, label='Sequential congruency (post-correct only):')

fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(FigSize.large[0] * 3.4, FigSize.large[1]))
plot_scalar_bars(ax3a, trials, specs_hist, measure='accuracy',  group_spacing=[4], baseline=0.5)
plot_scalar_bars(ax3b, trials, specs_hist, measure='rt_interp', group_spacing=[4])
plot_scalar_bars(ax3c, trials, specs_hist, measure='focus_in',  group_spacing=[4])
fig3.suptitle('Sequential congruency — history (t-2, t-1) → t, post-correct only', fontsize=7)
fig3.tight_layout()
fig3.savefig(test_config.export_path + 'flanker_sequential_congruency.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_sequential_congruency.pdf"}')

# Same cells as time courses rather than summary bars — accuracy builds within the
# trial, and history should show up as a shift in that build-up, not only in its endpoint.
hist_shades = ['#c6dbef', '#6baed6', '#3182bd', '#08519c']
specs_hist_I = [(history_cell(c2, c1, False), f'{lbl}→I', sh)
                for (c2, c1, lbl), sh in zip(hist_cells, hist_shades)]

fig3b, (ax3b_acc, ax3b_rt) = plt.subplots(1, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1]))
plot_accuracy_by_timestep(ax3b_acc, trials, specs_hist_I, test_config)
plot_rt_continuous(ax3b_rt,        trials, specs_hist_I, test_config)
fig3b.suptitle('Sequential congruency — within-trial time course, → incongruent', fontsize=7)
fig3b.tight_layout()
fig3b.savefig(test_config.export_path + 'flanker_sequential_timecourse.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_sequential_timecourse.pdf"}')

#%%
# ── Result 3b: does the sequential effect survive response repetition? ────────
#
# The classic confound (Mayr, Awh & Laurey 2003): congruency sequences are
# correlated with stimulus/response repetitions, and repetition priming produces a
# Gratton-shaped pattern without any control adjustment. If the interaction only
# appears when repetitions are pooled in, it is priming, not control.

def prev_cell(c1, current, extra):
    m = valid & prev_correct & extra
    m &= (prev_cong if c1 else prev_incong)
    m &= (cong      if current else incong)
    return m

fig4, axes4 = plt.subplots(2, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1] * 2))
for col, (rep_mask, rep_label) in enumerate([(switched, 'response switched'),
                                             (repeated, 'response repeated')]):
    specs_rep = [
        (prev_cell(True,  False, rep_mask), 'C→I', COL['incong']),
        (prev_cell(False, False, rep_mask), 'I→I', COL['error']),
        (prev_cell(True,  True,  rep_mask), 'C→C', COL['cong']),
        (prev_cell(False, True,  rep_mask), 'I→C', '#084594'),
    ]
    print_cell_counts(specs_rep, label=f'Gratton 2×2 | {rep_label}:')
    plot_scalar_bars(axes4[0, col], trials, specs_rep, measure='accuracy', baseline=0.5)
    plot_scalar_bars(axes4[1, col], trials, specs_rep, measure='rt_interp')
    axes4[0, col].set_title(rep_label, fontsize=6)

fig4.suptitle('Sequential congruency split by response repetition (post-correct)', fontsize=7)
fig4.tight_layout()
fig4.savefig(test_config.export_path + 'flanker_sequential_by_repetition.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_sequential_by_repetition.pdf"}')

#%%
# ── Result 4: post-error effects ──────────────────────────────────────────────
#
# Trial A = the preceding trial, trial B = the trial being measured.
#
# A is restricted to INCONGRUENT trials only, and split by outcome. Pooling A
# across congruency made the "post-correct" baseline a mixture dominated by
# congruent-correct trials, so the old comparison was really
# post-incongruent-error vs post-congruent-correct — two things changing at once.
#
# B is split by congruency, because a more target-focused control state helps
# incongruent B and hurts congruent B. Pooling B averages an effect with its own
# opposite.
#
# Z panel uses focus_in: the state B inherited, before B's own update.

post_specs = [
    (valid & prev_incong & prev_error   & incong, 'A err → I',  COL['error']),
    (valid & prev_incong & prev_error   & cong,   'A err → C',  '#f4a582'),
    (valid & prev_incong & prev_correct & incong, 'A corr → I', COL['incong']),
    (valid & prev_incong & prev_correct & cong,   'A corr → C', COL['cong']),
]
print_cell_counts(post_specs, label='Post-error (incongruent A only):')

fig5, (ax5a, ax5b, ax5c) = plt.subplots(1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1]))
plot_scalar_bars(ax5a, trials, post_specs, measure='accuracy',  group_spacing=[2], baseline=0.5)
plot_scalar_bars(ax5b, trials, post_specs, measure='rt_interp', group_spacing=[2])
plot_scalar_bars(ax5c, trials, post_specs, measure='focus_in',  group_spacing=[2])
fig5.suptitle('Post-error effects — incongruent trial A, trial B split by congruency', fontsize=7)
fig5.tight_layout()
fig5.savefig(test_config.export_path + 'flanker_post_error.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_post_error.pdf"}')

# Time courses for the same four cells
fig5b, (ax5b_acc, ax5b_rt) = plt.subplots(1, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1]))
plot_accuracy_by_timestep(ax5b_acc, trials, post_specs, test_config,
                          linestyles=['-', '--', '-', '--'])
plot_rt_continuous(ax5b_rt,        trials, post_specs, test_config,
                   linestyles=['-', '--', '-', '--'])
fig5b.suptitle('Post-error effects — within-trial time course', fontsize=7)
fig5b.tight_layout()
fig5b.savefig(test_config.export_path + 'flanker_post_error_timecourse.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_post_error_timecourse.pdf"}')

#%%
# ── Result 4b: near vs far errors ─────────────────────────────────────────────
#
# The hypothesis is that near-flanker errors drive a larger control correction than
# far-flanker errors, because near slots carry more prediction weight. That is only
# meaningful within incongruent trials — with congruent flankers there is nothing
# to be misled by — so A is incongruent throughout.

nearfar_specs = [
    (valid & prev_incong & prev_near & prev_error   & incong, 'near err → I',  COL['error']),
    (valid & prev_incong & prev_near & prev_error   & cong,   'near err → C',  '#f4a582'),
    (valid & prev_incong & prev_far  & prev_error   & incong, 'far err → I',   '#74add1'),
    (valid & prev_incong & prev_far  & prev_error   & cong,   'far err → C',   '#abd9e9'),
    (valid & prev_incong & prev_near & prev_correct & incong, 'near corr → I', COL['incong']),
    (valid & prev_incong & prev_far  & prev_correct & incong, 'far corr → I',  COL['neutral']),
]
print_cell_counts(nearfar_specs, label='Near vs far errors (incongruent A only):')

fig6, (ax6a, ax6b, ax6c) = plt.subplots(1, 3, figsize=(FigSize.large[0] * 3.4, FigSize.large[1]))
plot_scalar_bars(ax6a, trials, nearfar_specs, measure='accuracy',  group_spacing=[2, 4], baseline=0.5)
plot_scalar_bars(ax6b, trials, nearfar_specs, measure='rt_interp', group_spacing=[2, 4])
plot_scalar_bars(ax6c, trials, nearfar_specs, measure='focus_in',  group_spacing=[2, 4])
fig6.suptitle('Near vs far flanker errors — incongruent trial A', fontsize=7)
fig6.tight_layout()
fig6.savefig(test_config.export_path + 'flanker_post_error_near_far.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_post_error_near_far.pdf"}')

#%%
# ── Result 5: what drives the control update ──────────────────────────────────
#
# The panels above ask what a control state *does*. This one asks what *produces*
# it, and the difference matters: reading a Z level after conditioning on the trial
# outcome is circular, because a well-focused Z is what made the trial correct in
# the first place. `delta_focus` is the update the trial generated, which is a
# description of the learning rule rather than of its consequence, so grouping it
# by the trial's own properties is legitimate.
#
# Masks here index trial A itself, not the trial after it.

drive_specs = [
    (near & incong & ~correct_dec, 'near-incong err',  COL['error']),
    (far  & incong & ~correct_dec, 'far-incong err',   '#74add1'),
    (near & incong &  correct_dec, 'near-incong corr', COL['incong']),
    (far  & incong &  correct_dec, 'far-incong corr',  COL['neutral']),
]
print_cell_counts(drive_specs, label='Z update drivers:')

fig7, axes7 = plt.subplots(1, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1]))
plot_scalar_bars(axes7[0], trials, drive_specs, measure='delta_focus', group_spacing=[2], baseline=0.0)
if trials['pe'] is not None:
    plot_scalar_bars(axes7[1], trials, drive_specs, measure='pe', group_spacing=[2])
else:
    axes7[1].set_visible(False)
fig7.suptitle('What drives the Z update (measured on the trial itself)', fontsize=7)
fig7.tight_layout()
fig7.savefig(test_config.export_path + 'flanker_z_update_drivers.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_z_update_drivers.pdf"}')

#%%
# ── Result 6: trial-history regression ────────────────────────────────────────
#
# The cell contrasts above each condition on one factor at a time, so they cannot
# separate effects that travel together: an error is followed by a slow trial, but
# errors also happen mostly on incongruent trials, after slow trials, and later in the
# session. This estimates everything at once, mirroring the regression run on the human
# data (code_shared/STA_BH/Regression_on_Behavior.m, Fischer et al. 2018).
#
#   M1  the human design, minus RSI (no analogue here)
#   M2  + the lab's extended sequential terms, lag-2 history, target repetition
#   M3  + focus_in — does the control state absorb the history effects?
#
# Accuracy DV is ACC (1 = correct), so positive = more accurate. The RT model keeps
# error trials and controls for them with is_error, exactly as the human code does.

from flanker_regression import (SUMMARY_TERMS, build_design, fit_session,
                                plot_coefficients, report_session)

# M1 is only needed to compare against the human coefficients, so it is fitted only in
# detail mode; M2 carries the signatures and M3 adds the control state.
specs = ('M1', 'M2', 'M3') if regression_detail else ('M2', 'M3')
design = build_design(trials)
reg = {spec: fit_session(trials, spec=spec, design=design) for spec in specs}

report_session(reg, detail=regression_detail, design=design)

# Figure: M2 against M3, so the mediation is visible rather than tabulated. Plotted as
# t/z values (what STA_BH plots) because accuracy is in log-odds and RT in log timesteps,
# and the congruency effect is an order of magnitude larger than everything else.
terms_to_plot = ([t for t in reg['M3']['rt']['term'] if t != 'Intercept']
                 if regression_detail else SUMMARY_TERMS)
fig8, axes8 = plt.subplots(1, 2, figsize=FigSize.custom(6.6, 3.4 if regression_detail else 2.2))
for ax, dv, name in [(axes8[0], 'acc', 'Accuracy (1 = correct)'),
                     (axes8[1], 'rt', 'log RT')]:
    plot_coefficients(ax, reg['M2'][dv], title=name, order=terms_to_plot,
                      color=COL['incong'], y_offset=+0.16)
    plot_coefficients(ax, reg['M3'][dv], title=name, order=terms_to_plot,
                      color=COL['neutral'], y_offset=-0.16)
    ax.set_xlabel('t / z value  (dotted = ±1.96)')
axes8[0].legend(handles=[
    plt.Line2D([], [], color=COL['incong'], marker='o', markersize=3, linewidth=0.9,
               label='M2 (behaviour only)'),
    plt.Line2D([], [], color=COL['neutral'], marker='o', markersize=3, linewidth=0.9,
               label='M3 (+ inherited Z focus)')], fontsize=5, loc='lower left')
fig8.suptitle('Trial-history regression — one session', fontsize=7)
fig8.tight_layout()
fig8.savefig(test_config.export_path + 'flanker_regression_coefficients.pdf', bbox_inches='tight')
print(f'Exported: {test_config.export_path + "flanker_regression_coefficients.pdf"}')

# %%
