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
  - RT is the threshold crossing, interpolated inside the response window. A trial
    that never crosses did not respond: it is given rt = arrows_duration, so those
    trials pile into the last bin instead of being dropped (which censored incongruent
    trials, ~4x more likely to fail) or projected to an invented value. The failure
    rate is its own measure — `report_extraction` prints it, and `dec_*` carries it
    per cell.
  - Z is reported activated (softmax across slots) via the scalar `focus` index.
  - `focus_in` is the state a trial *inherited*; `delta_focus` is the update it
    produced. Z is logged after the LU step, so a trial's own Z already contains
    its own update and must not be read as the state it started from.

**This is one session, i.e. one synthetic subject.** Several effects that looked clear
in a single 5000-trial session did not survive ten seeds, and one reversed sign. Use
these figures to see mechanism and shape; take the statistics from the group scripts:

    python flanker_sweep_analysis.py            # across-seed tables
    python flanker_sweep_figures.py            # the group figure set

Analysis utilities live in flanker_analyses.py; per-seed measures in flanker_metrics.py.
"""

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

#%%
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot_style
plot_style.set_plot_style()

from plot_style import FLANKER_COLORS as COL, FigSize
from functions_and_utils import *
from configs import FlankerTaskConfig, FlankerRandomTrialsConfig
from train_and_infer_functions import *
from flanker_analyses import (
    extract_trials, select_trials, lagged_factors, decode_trial_types,
    report_extraction, print_cell_counts,
    plot_accuracy_by_timestep, plot_rt_distribution, plot_rt_continuous,
    plot_scalar_bars, plot_trial, example_trial_indices, plot_correlation_structure,
    sync_gating, mirror_to_model, reset_Z_uniform,
    mark_target_onset, _timestep_ticks,
)
# Panels shared with the group figures. Results 1c, 3c and 5b below draw from the same
# `spec_*` builders flanker_sweep_figures.py uses, so the workbench view of a measure and
# the across-seed view cannot drift apart. The cost is that a one-session panel has no
# error bar — `bars_with_seeds` needs replicates and here there is one — so where the
# trial-level spread is the point, use plot_scalar_bars with masks instead, as the older
# Results do.
from flanker_metrics import session_effects
from flanker_figure_utils import (bar_grid, bar_row, share_ylim, spec_post_conflict,
                                  spec_rt_by_outcome, spec_z_slot_update, _has)

fig_gaussian = False   # whether to overlay a Gaussian fit on RT PMFs

# ── Target-onset delay ("flankers first") ─────────────────────────────────────
#
# Timesteps by which the TARGET arrow's onset is delayed in Stage 2. The flankers are on
# from frame 0 throughout; for the first `target_delay` frames the centre slot carries
# background noise only. 0 reproduces simultaneous onset, i.e. every result before this.
#
# It is set on the TEST config only (below). Stage 1 trains on simultaneous onset and
# FlankerTaskDataset asserts the delay is 0, so this is a probe of what a read-out
# calibrated on simultaneous onset does when the target is late — which also means a null
# result is ambiguous between "no delay effect" and "never calibrated for this".
#
# Nothing is compensated for the delay. `response_start_timestep` stays 1 and the temporal
# loss weights are unchanged, so the model is asked for the target direction from t=1
# whether or not the target has arrived, and RT is measured from trial start. That is the
# point: the question is whether the response is *delayed* when the target is late, and
# whether the flankers alone are enough to answer early. Zeroing the loss over the
# pre-target window, or re-referencing RT to onset, would define the effect away.
#
# Prediction: RT rises with the delay, but congruent trials are held up less than
# incongruent ones — during the delay the flankers already point at the answer, so a
# congruent trial can commit before the target exists and an incongruent one commits
# wrongly. Read Result 2 first.
target_delay = 1       # 0 = simultaneous onset; 9 response steps, so 4 leaves 5 post-onset

# Result 6 (trial-history regression). False prints the ~6 rows that are results — the
# human signatures and the mediation. True adds every model's full coefficient table and
# the collinearity check, including the nuisance regressors that are in the design to be
# controlled for rather than read. Turn it on when auditing a fit or comparing against
# the human coefficients in code_shared/STA_BH.
regression_detail = False

# The palette is defined once in plot_style.FLANKER_COLORS and used by every flanker
# figure, single-session and group alike. Three orthogonal channels:
#
#     hue    congruency   blue = congruent, red = incongruent
#     shade  distance     dark = near flankers, light = far flankers
#     fill   outcome      filled bar / solid line = correct,
#                         hollow bar / dashed line = error
#
# Outcome deliberately rides on fill rather than on a third hue, so it reads the same way
# on a bar as on a line and costs no colour. Where a panel puts a *different* factor on
# shade — trial-history panels use it for the previous trial — that is called out locally.


# Every figure exported below is also queued here, together with a short caption drawn
# from the section comment above it, so a single combined PDF — one figure page followed
# by one caption page, in script order — can be written at the end. This is additive: it
# does not change how any individual figure is saved or displayed.
report_entries = []


def export_fig(fig, filename, cfg, caption=None):
    """Save fig to <cfg.export_path>/filename (unchanged behaviour) and, if a caption is
    given, queue (fig, filename, caption) for the combined report PDF."""
    path = cfg.export_path + filename
    fig.savefig(path, bbox_inches='tight')
    print(f'Exported: {path}')
    if caption is not None:
        report_entries.append((fig, filename, caption))


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

# 1.35 = 0.9 x 1.5, the old working point carried across the retiming to a 10-timestep
# trial: evidence accumulates over the response window, so SNR grows as
# sqrt(n_response_steps) and 4 -> 9 steps is a factor of 1.5. A first-order estimate, not a
# calibration — check accuracy is off both ceiling and floor before reading anything into a
# result, and let the sweep's noise ladder locate the real working point.
training_noise_std = 1.35
testing_noise_std = 1.35
config.arrow_noise_std = training_noise_std

# ── Stage-1 oracle gate jitter ────────────────────────────────────────────────
#
# True redraws the sharpness of the oracle gate on every training trial, using
# models.DEFAULT_ORACLE_GATE_JITTER = (0.5, 3.0); pass an explicit (lo, hi) pair to choose
# the range yourself; None or False reproduces the original fixed oracle. (0.5, 3.0) spans
# gate peaks of about 0.29 to 0.83 with Z_dim = 5 at softmax_temp = 1.
#
# Why it is here. Stage 1 hands the model the target slot as a one-hot, which
# `latent_activation_function` softmaxes into the gate the RNN actually applies. At
# softmax_temp = 1 that is the SAME vector on every training trial — peak 0.405 on the
# target, 0.149 elsewhere — so the weights only ever learn to emit +-1 at that one gate
# sharpness. Stage 2 then infers Z freely and runs sharper than 0.405 on about three
# quarters of trials, where the output overshoots to 2-3 against a target of 1. Because
# the latent update descends squared error, on a congruent trial (whose sign is already
# right at every gate) the only gradient left is 'make the output smaller', and the update
# obeys it by flattening the gate. The controller is taught to stop attending by exactly
# the trials on which attention does not matter, the gate drifts onto slots that carry no
# arrow, and near-flanker trials — which leave the outer slots empty — pay for it.
#
# Jitter separates the two things sharpness was doing. Trained across a range of gate
# sharpness, the read-out emits the same magnitude at all of them, so sharpness no longer
# doubles as a gain control and the only gradient left on Z is 'which slot'.
#
# Measured effect (3 seeds, 5000 test trials, jitter 0.5-3 against None):
#   arrow_noise_std 0.4  gate peaks off target 0.297 -> 0.000; near-cong minus far-cong
#                        accuracy -0.028 -> +0.001; non-responses 12.8% -> 2.2%. Overall
#                        accuracy goes to 0.98, so the accuracy congruency effect collapses
#                        to 0.037 on ceiling while the RT one grows, 0.361 -> 0.473
#                        timesteps among trials that responded. At that ceiling Result 6's
#                        accuracy GLM has ~40 errors in 5000 trials and cannot be
#                        identified; it is reported as not fitted and the RT model still
#                        runs. Another reason to prefer arrow_noise_std = 1.0.
#   arrow_noise_std 1.0  gate peaks off target 0.087 -> 0.002; congruency effect
#                        0.227 -> 0.252 in accuracy and 0.428 -> 0.557 timesteps in RT;
#                        distance effect on incongruent trials -0.138 -> -0.199; distance
#                        interaction +0.151 -> +0.215. Nothing is lost and the human
#                        signatures all grow, which is why 1.0 is the level to run at.
# Only three seeds; confirm with flanker_sweep before treating any of it as a result.
# The full diagnosis is in flanker_near_cong_diagnostic.py.
#
# CONFIRMED, AND IT DOES NOT SAY WHAT THE NUMBERS ABOVE SAY. Those 3-seed measurements were
# taken against bg_noise_std = 0.1 (see update_config below, which sets it to 0). A 2x2 over
# jitter x p_corr_by_distance[2] at bg_noise_std = 0 — 20 seeds per cell, 5 noise levels,
# exports/flanker_random/factorial_corr_jitter — says jitter is NOT as critical as the
# paragraph above implies, and on balance costs more than it buys at arrow_noise_std 0.9:
#
#   near-cong minus far-cong   +0.011 already WITHOUT jitter, so the artifact jitter was
#                              adopted to fix is fixed by bg_noise_std = 0 on its own
#   PERI                       +0.073 (human direction) -> -0.188 REVERSED by jitter
#   distance effect on RT      +0.137 (significant) -> +0.097 (n.s.)
#   mean focus (Z)             0.342 -> 0.391, i.e. jitter drives Z HIGHER; suspect this
#                              as the mechanism behind both losses
#   post-error slowing pes_BI  0.108 (n.s.) -> 0.375 (significant) — the one real gain
#   human signatures matched   9 of 11 without jitter, 8 of 11 with. Across the whole
#                              ladder jitter never matches more than the baseline at any
#                              level; the PERI reversal is mid-ladder only, and at
#                              arrow_noise_std 0.4 jitter doubles PERI instead
#
# Set it to None to run what the factorial found best; it is left on here so this script
# keeps reproducing the sessions the current figures came from.
config.oracle_gate_jitter = (0.5, 1.5)      # None/False = off; True = (0.5, 3.0); or a (lo, hi) pair

def update_config(cfg):
    # Use this to apply to both training and testing configs.
    # bg_noise_std = 0 is a real departure from the class default (0.1) and is mirrored in
    # flanker_sweep_config.PRETRAIN_OVERRIDES / TEST_OVERRIDES so the sweep runs the same
    # simulation as this workbench. Change it in one place and change it in the other.
    cfg.bg_noise_std = 0

update_config(config)
# n_pretrain_trials (4000) and post-gating are FlankerTaskConfig defaults; override here
# only if this run needs to deviate, e.g. config.set_n_pretrain_trials(2000).



# ── Stage 1: train ────────────────────────────────────────────────────────────

print(f'Stage 1 | seed={config.env_seed}  trials={config.n_pretrain_trials}  '
      f'arrows_duration={config.arrows_duration}')
print(f'  temporal_loss_weights={[round(w, 3) for w in config.temporal_loss_weights]}')
print(f'  p_corr_by_distance={config.p_corr_by_distance}')
print(f'  oracle_gate_jitter={config.oracle_gate_jitter}  softmax_temp={config.softmax_temp}')

logger, model, config, figs = train_model(
    config, seed=config.env_seed,
    save_models=True, load_models=False,
    run_test_phase=False,
)

panel_order = [ 'latent_2d', 'corrects']
_=fig = plot_logger_panels(logger, config, panel_order, x2=None,
                         annotate_phases='corrects', width=5, legends=False, dpi=140)
export_fig(fig, 'flanker_pretrain_results.pdf', config, caption=(
    "Stage 1 training. Left: 2D projection of the latent Z trajectory across training. "
    "Right: P(target) accuracy across trials, with block boundaries annotated. Oracle Z "
    "(the true target slot, handed to the model as a one-hot latent) drives training so "
    "the weights learn to use it. This is a training diagnostic, not a result — nothing "
    "here is analysed."))

#%%
# ── Stage 1 sanity check: did training work? ──────────────────────────────────
#
# 1: learning curve — accuracy against training trial, split by congruency. Watch
#    whether the incongruent curve has actually plateaued before the weights are
#    frozen; if it is still climbing, the spatial weighting is not yet settled.
# 2: P(target) by training third (early/mid/late), across timesteps within a trial.
# 3: P(target) split by congruency (Stage 1 hlcids = congruency flag).
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
ax_thirds.set_xticks(_timestep_ticks(trials1['ad']))
ax_thirds.set_xlabel('Timestep within trial')
ax_thirds.set_ylabel('P(target)')
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
export_fig(fig_acc, 'flanker_pretrain_sanity.pdf', config, caption=(
    "Stage 1 sanity check: did training work? (1) Learning curve, accuracy against "
    "training trial, split by congruency — watch whether the incongruent curve has "
    "plateaued before the weights are frozen. (2) P(target) by training third "
    "(early/mid/late) across timesteps within a trial. (3) P(target) split by congruency "
    "(Stage 1 congruency flag). (4) RT distribution, late trials only. This is a training "
    "diagnostic, not a result — Stage 1 is blocked and oracle-driven."))

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
test_config.no_of_steps_in_latent_space = 1
update_config(test_config)   # was update_config(config): Stage 2 kept the class default
                             # bg_noise_std = 0.1 while Stage 1 trained at 0
test_config.arrow_noise_std = testing_noise_std
test_config.target_delay   = target_delay      # Stage 2 only; Stage 1 asserts it is 0
test_config._validate()                        # bounds-checks target_delay against arrows_duration
sync_gating(test_config, config)
mirror_to_model(model, test_config)
reset_Z_uniform(model, scale=0.2, seed=test_config.env_seed)

print(f'Stage 2 | trials={test_config.n_trials}  p_congruent={test_config.p_congruent}  '
      f'p_near={test_config.p_near}  Z_lr={test_config.Z_lr}  '
      f'arrows_duration={test_config.arrows_duration}  target_delay={test_config.target_delay}')

logger_t, model_t, test_config, figs_t = train_model(
    test_config, seed=test_config.env_seed,
    save_models=False, load_models=False,
    pretrained_model=model,
    run_test_phase=False,
)

_=plot_logger_panels(logger_t, test_config, ['latent_2d', 'corrects'], x2=None,
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
# ── What a trial looks like ───────────────────────────────────────────────────
#
# One example trial per condition: the five slots' noisy observations, the attention
# gate the trial inherited, and the decision variable that came out. This is the task
# illustration — it says what "near-incongruent" actually means as a stimulus, and it is
# the panel to reach for when a cell's summary statistic looks wrong and the question is
# what the model was shown.
#
# It reads from `trials`, i.e. from the logger, and needs no live model: `_log_batch`
# records the raw UNMASKED input batch, so both the arrow observations and the hidden
# true direction survive into `logger.inputs`. (This used to be done by breakpointing
# inside `predictive_learning` and running a script against its locals.)

for _label, _idx in example_trial_indices(trials, test_config, seed=test_config.env_seed):
    _fig = plot_trial(trials, test_config, trial=_idx)
    export_fig(_fig, f'flanker_trial_{_label}.pdf', test_config, caption=(
        f"Example {_label} trial (#{_idx}) from the Stage-2 session. Rows top to bottom: "
        "each slot's observed value at every timestep (up = rightward), the true target "
        "direction, the temporal loss weights that impose speed pressure, and the "
        "decision variable. Colour is role — black target, congruency hue and distance "
        "shade for the flankers, grey for slots holding no arrow. The dashed line in each "
        "active row is the noiseless arrow level, so a trace on the wrong side of zero is "
        "a slot that misled the model. The gate column is the softmaxed attention weight "
        "the trial inherited (z_in, the gate these outputs were produced under), against "
        "a dotted line at the uniform value 1/5."))

# The Stage-1 correlation profile that teaches the model near != far. Not a per-trial
# quantity — it is the config knob whose index-1-vs-2 gap sets the ceiling on any
# distance effect, shown against the power-law family it was chosen from.
fig_corr = plot_correlation_structure(config)
export_fig(fig_corr, 'flanker_correlation_structure.pdf', test_config, caption=(
    "Stage-1 companion correlation against slot distance: the config's own "
    "p_corr_by_distance alongside the power-law family p(d) = 1 / (1 + d)^alpha it was "
    "chosen from. The gap between distance 1 and 2 is the only thing that teaches the "
    "model near != far, so it sets the ceiling on any distance effect; the profile must "
    "stay above 0.5 everywhere or the model learns to ignore the companion entirely."))

#%%
# ── Result 1: congruency × distance ───────────────────────────────────────────
#
# The headline behavioural fingerprint, now from interleaved trials rather than
# blocks. Prediction: incongruent is slower and less accurate, and the cost is
# larger for near flankers than far ones.

# Ordered congruent pair first, then incongruent, near before far inside each — so the
# congruency effect the figure is about sits side by side and distance is the within-pair
# step. Shade carries distance; the line style repeats it for the curve panels, where two
# shades of one hue are harder to separate than two dash patterns.
conditions = [
    (near & cong,   'near-cong',   COL['near_cong'],   '-'),
    (far  & cong,   'far-cong',    COL['far_cong'],    '--'),
    (near & incong, 'near-incong', COL['near_incong'], '-'),
    (far  & incong, 'far-incong',  COL['far_incong'],  '--'),
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
export_fig(fig1, 'flanker_congruency_distance.pdf', test_config, caption=(
    "Result 1: congruency x distance — the headline behavioural fingerprint, from "
    "Stage-2 randomly interleaved trials. Accuracy by timestep, continuous RT, and the "
    "RT distribution for the four near/far x congruent/incongruent cells. Prediction: "
    "incongruent trials are slower and less accurate, and the cost is larger for near "
    "flankers than far ones."))

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
export_fig(fig1b, 'flanker_congruency_distance_bars.pdf', test_config, caption=(
    "Result 1, bar summary: mean +/- SEM accuracy, RT, and delta_focus for the four "
    "congruency x distance cells. delta_focus (the update a trial produced), not focus, "
    "is plotted — trials are i.i.d., so the state a trial inherits (focus_in) cannot "
    "differ by its own condition; delta_focus isolates the update itself."))

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

for ax, base, title, col in [(ax1c_cong, cong, 'Congruent', COL['cong']),
                             (ax1c_incong, incong, 'Incongruent', COL['incong'])]:
    specs_out = [(base & correct_dec, 'correct', col),
                 (base & error_dec,   'error',   col)]
    plot_rt_continuous(ax, trials, specs_out, test_config, linestyles=['-', '--'])
    ax.set_title(title, fontsize=6)

fig1c.tight_layout()
export_fig(fig1c, 'flanker_session_and_rt_by_outcome.pdf', test_config, caption=(
    "Result 1b: session stability and RT by outcome. (1) Accuracy across the Stage-2 "
    "test session — weights are frozen, so drift here is Z settling into the list "
    "statistics, not learning. (2-3) RT distributions for correct vs. error responses, "
    "within congruent and within incongruent trials — errors should be fast on "
    "incongruent trials if they are flanker-driven."))

#%%
# ── Result 1c: RT for correct vs error, in each cell ──────────────────────────
#
# Result 1b splits RT by outcome pooled over distance. This adds the four cells and the
# contrast itself, which is the quantity the scorecard scores.
#
# Human flanker errors are FAST: on an incongruent trial the flankers reach threshold
# before the target does, so an error beats a correct response to the boundary. Positive
# `fasterr` is that signature.
#
# Read the decided-only version. `rt_interp` gives a trial that never crossed the trial
# end by the house convention, and errors are exactly the trials that fail to cross — so
# on the uncensored measure a non-response reads as a very slow error rather than as no
# response at all, which points the contrast the wrong way.

sess_eff = session_effects(trials)

fig1d, axes1d = bar_row(spec_rt_by_outcome(sess_eff, decided=True))
share_ylim(axes1d[0], axes1d[1])        # the two level panels are the same quantity
fig1d.suptitle('RT by outcome, per cell (decided trials only)', fontsize=7)
fig1d.tight_layout()
export_fig(fig1d, 'flanker_rt_by_outcome.pdf', test_config, caption=(
    "Result 1c: RT for correct vs. error responses in each congruency x distance cell, "
    "on trials that crossed threshold inside the window. Left two panels are the RT "
    "levels (hollow = errors, the house fill convention for outcome); the right panel is "
    "the contrast, positive where errors are faster than correct responses — the human "
    "flanker signature, since on an incongruent trial the flankers reach threshold "
    "first. Decided-only because errors fail to cross far more often than correct "
    "responses, so the uncensored measure reports non-responses as slow errors. One "
    "session, so the bars carry no error bar; the across-seed version is "
    "flanker_sweep_figures group_3."))

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
mark_target_onset(ax2, test_config, label=True)
ax2.set_xticks(_timestep_ticks(trials['ad']))
ax2.set_xlabel('Timestep within trial')
ax2.set_ylabel('Output toward final decision\n(sign-normalised)')
ax2.set_ylim(-1.1, 1.1)
ax2.legend(fontsize=5)
fig2.tight_layout()
export_fig(fig2, 'flanker_accumulation.pdf', test_config, caption=(
    "Result 2: within-trial evidence accumulation. Output is sign-normalised by the "
    "model's own final decision, so left and right decisions align on one axis; positive "
    "= accumulating toward whatever the model eventually chose. The model's analogue of "
    "the human beta-power lateralisation signal — incongruent trials should dip early "
    "(flankers pull away from the eventual choice) and reverse late on correct trials. "
    "With a target-onset delay the dash-dotted line marks the first timestep at which the "
    "target can reach the output; everything left of it is driven by the flankers alone, "
    "so congruent traces climbing and incongruent traces heading the wrong way before that "
    "line is the delay effect itself, not an artefact."))

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

hist_cells = [(True, True, 'CC'), (False, True, 'IC'),
              (True, False, 'CI'), (False, False, 'II')]

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
export_fig(fig3, 'flanker_sequential_congruency.pdf', test_config, caption=(
    "Result 3: sequential congruency, all four (t-2, t-1) -> t history cells (CC, CI, "
    "IC, II), restricted to trials where both preceding trials were correct (so the "
    "effect isn't confounded with post-error adaptation). CI and IC are kept separate "
    "rather than pooled, since they are opposite trajectories. II->C vs CC->C is where "
    "the predicted speed-accuracy dissociation lives (faster but more error-prone)."))

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
    # Hue = the current trial's congruency; shade = the previous trial's (dark = the
    # conflict trial). Distance is not shown in this panel, so shade is free to carry it.
    specs_rep = [
        (prev_cell(True,  False, rep_mask), 'C→I', COL['far_incong']),
        (prev_cell(False, False, rep_mask), 'I→I', COL['near_incong']),
        (prev_cell(True,  True,  rep_mask), 'C→C', COL['far_cong']),
        (prev_cell(False, True,  rep_mask), 'I→C', COL['near_cong']),
    ]
    print_cell_counts(specs_rep, label=f'Gratton 2×2 | {rep_label}:')
    plot_scalar_bars(axes4[0, col], trials, specs_rep, measure='accuracy', baseline=0.5)
    plot_scalar_bars(axes4[1, col], trials, specs_rep, measure='rt_interp')
    axes4[0, col].set_title(rep_label, fontsize=6)

fig4.suptitle('Sequential congruency split by response repetition (post-correct)', fontsize=7)
fig4.tight_layout()
export_fig(fig4, 'flanker_sequential_by_repetition.pdf', test_config, caption=(
    "Result 3b: does the sequential effect survive response repetition? Splits the "
    "classic Gratton 2x2 (C->I, I->I, C->C, I->C accuracy and RT) by whether the "
    "response repeated or switched, to rule out the repetition-priming confound (Mayr, "
    "Awh & Laurey 2003) — if the interaction only appears when repetitions are pooled "
    "in, it is priming, not control."))

#%%
# ── Result 3c: post-incongruent slowing and accuracy ──────────────────────────
#
# The conflict-triggered twin of Result 4 below: same shape, one factor changed. There
# trial A is an error; here trial A is CORRECT and incongruent.
#
# That restriction is the whole measure. Incongruent trials fail more often, so an
# unrestricted "after an incongruent trial" contrast is partly post-error slowing wearing
# conflict's name — the same reason Result 3 restricts its history cells to post-correct.
#
# Positive slowing AND positive accuracy is the control-recruitment reading: conflict
# recruits control, and the next trial is more deliberate. Trial B is split by congruency
# because a target-focused state helps incongruent B and hurts congruent B.

sess_eff = session_effects(trials)

fig3c, axes3c = bar_row(spec_post_conflict(sess_eff))
share_ylim(axes3c[0], axes3c[1])        # PCS against its decided-only companion
fig3c.suptitle('Post-incongruent adaptation — post-correct trial A', fontsize=7)
fig3c.tight_layout()
export_fig(fig3c, 'flanker_post_conflict.pdf', test_config, caption=(
    "Result 3c: post-incongruent slowing (PCS) and accuracy (PCA), trial A restricted to "
    "correct responses so this is conflict adaptation rather than post-error adaptation. "
    "Trial B split by congruency, since a target-focused state helps incongruent B and "
    "hurts congruent B. The third bar in the PCS and PCA panels is the lag-2 cell "
    "contrast II->I against CC->I, in the same unit as the lag-1 measure beside it. "
    "Panel 2 is the decided-only RT companion: a large gap from panel 1 means the "
    "contrast is carrying non-responses rather than speed. Panel 4 is the inherited "
    "control state behind the behaviour. One session; the across-seed version is "
    "flanker_sweep_figures group_10."))

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

# Hue = trial B's congruency, which is what the bar is measuring. Trial A's outcome is
# the fill: hollow after an error, filled after a correct response.
post_specs = [
    (valid & prev_incong & prev_error   & incong, 'A err → I',  COL['incong']),
    (valid & prev_incong & prev_error   & cong,   'A err → C',  COL['cong']),
    (valid & prev_incong & prev_correct & incong, 'A corr → I', COL['incong']),
    (valid & prev_incong & prev_correct & cong,   'A corr → C', COL['cong']),
]
post_hollow = [True, True, False, False]
print_cell_counts(post_specs, label='Post-error (incongruent A only):')

fig5, (ax5a, ax5b, ax5c) = plt.subplots(1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1]))
plot_scalar_bars(ax5a, trials, post_specs, measure='accuracy',  group_spacing=[2],
                 baseline=0.5, hollow=post_hollow)
plot_scalar_bars(ax5b, trials, post_specs, measure='rt_interp', group_spacing=[2],
                 hollow=post_hollow)
plot_scalar_bars(ax5c, trials, post_specs, measure='focus_in',  group_spacing=[2],
                 hollow=post_hollow)
fig5.suptitle('Post-error effects — incongruent trial A, trial B split by congruency', fontsize=7)
fig5.tight_layout()
export_fig(fig5, 'flanker_post_error.pdf', test_config, caption=(
    "Result 4: post-error effects. Trial A (the preceding trial, incongruent only) "
    "split by outcome; trial B split by congruency. Accuracy, RT, and focus_in (the "
    "state B inherited, before B's own update) for the four A-outcome x B-congruency "
    "cells."))

# Time courses for the same four cells
fig5b, (ax5b_acc, ax5b_rt) = plt.subplots(1, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1]))
# Dashed = trial A was an error, the line-plot counterpart of the hollow bars above.
post_ls = ['--', '--', '-', '-']
plot_accuracy_by_timestep(ax5b_acc, trials, post_specs, test_config, linestyles=post_ls)
plot_rt_continuous(ax5b_rt,        trials, post_specs, test_config, linestyles=post_ls)
fig5b.suptitle('Post-error effects — within-trial time course', fontsize=7)
fig5b.tight_layout()
export_fig(fig5b, 'flanker_post_error_timecourse.pdf', test_config, caption=(
    "Result 4, time course: the same four post-error cells as within-trial accuracy and "
    "RT time courses. Dashed = trial A was an error, the line-plot counterpart of the "
    "hollow bars in the previous figure."))

#%%
# ── Result 4b: near vs far errors ─────────────────────────────────────────────
#
# The hypothesis is that near-flanker errors drive a larger control correction than
# far-flanker errors, because near slots carry more prediction weight. That is only
# meaningful within incongruent trials — with congruent flankers there is nothing
# to be misled by — so A is incongruent throughout.

# Hue = trial B's congruency, shade = trial A's flanker distance (the factor under test
# here, so it takes the shade channel), fill = trial A's outcome.
nearfar_specs = [
    (valid & prev_incong & prev_near & prev_error   & incong, 'near err → I',  COL['near_incong']),
    (valid & prev_incong & prev_near & prev_error   & cong,   'near err → C',  COL['near_cong']),
    (valid & prev_incong & prev_far  & prev_error   & incong, 'far err → I',   COL['far_incong']),
    (valid & prev_incong & prev_far  & prev_error   & cong,   'far err → C',   COL['far_cong']),
    (valid & prev_incong & prev_near & prev_correct & incong, 'near corr → I', COL['near_incong']),
    (valid & prev_incong & prev_far  & prev_correct & incong, 'far corr → I',  COL['far_incong']),
]
nearfar_hollow = [True, True, True, True, False, False]
print_cell_counts(nearfar_specs, label='Near vs far errors (incongruent A only):')

fig6, (ax6a, ax6b, ax6c) = plt.subplots(1, 3, figsize=(FigSize.large[0] * 3.4, FigSize.large[1]))
plot_scalar_bars(ax6a, trials, nearfar_specs, measure='accuracy',  group_spacing=[2, 4],
                 baseline=0.5, hollow=nearfar_hollow)
plot_scalar_bars(ax6b, trials, nearfar_specs, measure='rt_interp', group_spacing=[2, 4],
                 hollow=nearfar_hollow)
plot_scalar_bars(ax6c, trials, nearfar_specs, measure='focus_in',  group_spacing=[2, 4],
                 hollow=nearfar_hollow)
fig6.suptitle('Near vs far flanker errors — incongruent trial A', fontsize=7)
fig6.tight_layout()
export_fig(fig6, 'flanker_post_error_near_far.pdf', test_config, caption=(
    "Result 4b: near vs. far errors. Tests whether near-flanker errors on trial A "
    "(incongruent throughout) drive a larger control correction than far-flanker "
    "errors, since near slots carry more prediction weight. Accuracy, RT, and focus_in "
    "split by trial A's distance and outcome, trial B's congruency."))

#%%
# ── Result 4c: the circularity behind the missing post-error improvement ──────
#
# Result 4 asks what happens *after* an error. This asks what was already true *before*
# it, and the answer undercuts the usual reading: a trial that goes wrong already
# inherited a lower control state, so conditioning on the outcome and then reading the
# state is circular — the low state is what produced the error.
#
# The error's own update is real and much larger than a correct trial's. Whether the next
# trial still starts behind depends on whether that update covers the gap, which panel 2
# measures. Replicates here are individual error events within this one session, so the
# error bars are trial-level; the group version (one dot per seed) is
# flanker_sweep_figures.fig_circularity.

from flanker_metrics import event_locked
from flanker_figure_utils import plot_circularity

ev = event_locked(trials)
ev_plot = dict(ev)
for key in ('start_gap', 'upd_err', 'upd_corr',
            'frac_err_noisy', 'frac_err_clean',
            'dfocus_err_noisy', 'dfocus_err_clean'):    # scalars -> one replicate
    ev_plot[key] = np.array([ev[key]])
for key in ('curve_x', 'curve_y'):
    ev_plot[key] = [ev[key]]

fig4c, axes4c = plt.subplots(1, 5, figsize=(FigSize.large[0] * 5, FigSize.large[1]))
plot_circularity(axes4c, ev_plot)
fig4c.suptitle('Post-error control — the deficit precedes the error', fontsize=7)
fig4c.tight_layout()
export_fig(fig4c, 'flanker_pia_circularity.pdf', test_config, caption=(
    "Result 4c: the circularity behind the missing post-error improvement. (1) What was "
    "already true *before* an error: a trial that goes wrong already inherited a lower "
    "control state, so conditioning on the outcome and then reading the state is "
    "circular. (2-3) Why the average post-error update is unreliable: incongruent errors "
    "split (median split on centre_evidence) into noise-driven — the centre slot's own "
    "evidence misled the model — and flanker-driven — centre evidence was clean, the "
    "flankers just won. Panel 3 shows each kind's own delta_focus update against a "
    "correct trial's; the two kinds teach Z in opposite directions, which is why the "
    "pooled correction in panel (1)'s gap does not reliably close it. (4-5) The "
    "behavioural cost: accuracy after an error vs. after a correct trial, and the "
    "exchange rate between inherited control and accuracy."))

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

# Every cell is incongruent, so hue is constant and the two channels that vary are the
# ones being contrasted: shade = distance, fill = outcome.
drive_specs = [
    (near & incong & ~correct_dec, 'near-incong err',  COL['near_incong']),
    (far  & incong & ~correct_dec, 'far-incong err',   COL['far_incong']),
    (near & incong &  correct_dec, 'near-incong corr', COL['near_incong']),
    (far  & incong &  correct_dec, 'far-incong corr',  COL['far_incong']),
]
drive_hollow = [True, True, False, False]
print_cell_counts(drive_specs, label='Z update drivers:')

fig7, axes7 = plt.subplots(1, 2, figsize=(FigSize.large[0] * 2, FigSize.large[1]))
plot_scalar_bars(axes7[0], trials, drive_specs, measure='delta_focus', group_spacing=[2],
                 baseline=0.0, hollow=drive_hollow)
if trials['pe'] is not None:
    plot_scalar_bars(axes7[1], trials, drive_specs, measure='pe', group_spacing=[2],
                     hollow=drive_hollow)
else:
    axes7[1].set_visible(False)
fig7.suptitle('What drives the Z update (measured on the trial itself)', fontsize=7)
fig7.tight_layout()
export_fig(fig7, 'flanker_z_update_drivers.pdf', test_config, caption=(
    "Result 5: what drives the control update. delta_focus (the update a trial "
    "generated, not the Z level) grouped by the trial's own near/far x outcome, "
    "incongruent trials only — a description of the learning rule rather than of its "
    "consequence, so grouping it by the trial's own properties is legitimate, unlike "
    "reading a post-outcome Z level."))

#%%
# ── Result 5b: what the update does to each slot ──────────────────────────────
#
# Result 5 reads the update through `delta_focus` — centre minus the mean of the flankers
# — which can only say whether the gate moved toward the target. This says where it went.
#
# Row 1 is the fixed geometry (centre, the near pair, the far pair). Row 2 is the role
# each pair played on that trial, which is a different question: a near display leaves
# slots 0 and 4 empty and a far display leaves 1 and 3, so the fixed geometry mixes
# "distractor" with "nothing there".
#
# Row 3 appears only when the run logged gradients, and it is the one that is safe to
# read as magnitude. `delta_z` is the change in the SOFTMAXED gate, so the five per-slot
# values sum to ~0: the centre cannot rise without something else falling, and a negative
# bar in rows 1 and 2 is not on its own evidence that a slot was suppressed. The raw
# dL/dZ is under no such constraint. Note its sign is inverted relative to the update —
# the step descends the gradient, so a positive dL/dZ is a slot the trial pushed DOWN.

sess_eff = session_effects(trials)

slot_rows = [spec_z_slot_update(sess_eff, grouping='geometry'),
             spec_z_slot_update(sess_eff, grouping='role')]
if _has(sess_eff, 'zgrad_centre_incong_err'):
    slot_rows.append(spec_z_slot_update(sess_eff, grouping='geometry', measure='zgrad'))
fig7b, axes7b = bar_grid(slot_rows)
fig7b.suptitle('What the update does to each slot', fontsize=7)
fig7b.tight_layout()
export_fig(fig7b, 'flanker_z_slot_update.pdf', test_config, caption=(
    "Result 5b: the update to Z per slot rather than collapsed into the focus index, "
    "correct trials against errors (hollow). Row 1 fixed geometry, row 2 the role each "
    "slot pair played on the trial — which pair is empty swaps with flanker distance, so "
    "the two groupings are not the same decomposition. Row 3, when gradients were "
    "logged, is the raw dL/dZ: delta_z is a change in the softmaxed gate and sums to ~0 "
    "across slots, so a negative bar above is not by itself suppression, while the "
    "gradient carries no such constraint. The gradient's sign is inverted relative to "
    "the update, since the step descends it. Correct and error panels are deliberately "
    "not on a shared y scale — an error's update is several times a correct trial's. One "
    "session; the across-seed version is flanker_sweep_figures group_11."))

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
export_fig(fig8, 'flanker_regression_coefficients.pdf', test_config, caption=(
    "Result 6: trial-history regression, mirroring the human GLM "
    "(code_shared/STA_BH/Regression_on_Behavior.m, Fischer et al. 2018). M2 = "
    "behavioural history terms only; M3 = + focus_in (the inherited control state), "
    "testing whether the control state mediates the history effects. Plotted as t/z "
    "values, since accuracy is in log-odds and RT in log timesteps; dotted lines at "
    "+/-1.96."))

#%%
# ── Combined report: every figure above, one per page, with a caption page after each ──
#
# Purely additive — every figure is still saved and displayed exactly as above; this just
# collects the same figure objects (in the order they were created) into one PDF for
# skimming the whole session end to end. Written to the Stage-2 export folder, since most
# of the figures — and the run this document is really about — live there.

def _caption_page(title, caption, figsize=(8.27, 3.0)):
    """A text-only page: the export filename as a heading, the section comment as body."""
    fig = plt.figure(figsize=figsize)
    wrapped = textwrap.fill(caption, width=100)
    fig.text(0.06, 0.90, title, ha='left', va='top', fontsize=9, fontweight='bold')
    fig.text(0.06, 0.78, wrapped, ha='left', va='top', fontsize=7, family='monospace')
    return fig

report_path = test_config.export_path + 'flanker_all_figures_report.pdf'
with PdfPages(report_path) as pdf:
    for fig, filename, caption in report_entries:
        pdf.savefig(fig, bbox_inches='tight')
        cap_fig = _caption_page(filename, caption)
        pdf.savefig(cap_fig)
        plt.close(cap_fig)
print(f'Exported combined report ({len(report_entries)} figures): {report_path}')

# %%
