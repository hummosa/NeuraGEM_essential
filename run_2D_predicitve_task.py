"""
run_example.py — Minimal end-to-end NeuraGEM training and visualisation.

Algorithm summary
─────────────────
NeuraGEM runs two gradient-descent steps per batch:

  Weight Update (WU): standard BPTT.
    Minimise prediction loss by updating all RNN weights.

  Latent Update (LU): gradient descent on Z only.
    Minimise the same prediction loss by updating Z — the latent context variable —
    without touching any weights.

Z is carried across batches (pass_previous_latent=True), so context
inferred in one batch warm-starts the next. This lets the model adapt
to context shifts within a few timesteps, without requiring explicit task labels.

Training phases
───────────────
  Phase 2 (blocked): both WU and LU; context presented in long blocks.
  Phase 3a (test):   weights frozen; LU still runs → measures adaptation speed.
  Phase 3b (test):   both frozen → pure feedforward baseline.
"""
if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    

import plot_style
plot_style.set_plot_style()

from functions_and_utils import *
from configs import *
from train_and_infer_functions import *


# ── Choose and configure the task ─────────────────────────────────────────────

from configs import RotatingTargetsConfig

config = RotatingTargetsConfig()
config.train_rotations = [0.0,  90.0,]
# config.test_rotations  = [45.0, 135.0, 225.0, 315.0]
config.test_rotations  = [0.0, 90.0]
config.no_of_steps_in_latent_space = 1
# block_size = n_miniblocks * n_colors * 20 = 800 timesteps per block
# set blocked_phase_length to control number of training blocks (e.g. 10 blocks = 8000)
config.add_passive_learning_phase = True
config.passive_phase_length = 1000
config.blocked_phase_length = 1000   # ← 
config.n_miniblocks_per_state_block = 200 // config.n_colors  # 10 
config.block_size  = config.n_miniblocks_per_state_block * config.n_colors  # 80
config.test_no_of_blocks   = 4       # ← blocks in Phase 3 test
config.pass_previous_latent = True
if config.pass_previous_latent:
    config.LU_lr = 0.3
    config.l2_loss = 0.000_0011
else:
    config.LU_lr = 0.3
    config.l2_loss = 0.
    config.no_of_steps_in_latent_space = 3
    config.seq_len = 3
# config.seq_len = 4
config.WU_lr = 0.00051
config.what_latent_to_use = 'self'
# config.what_latent_to_use = 'taskID'
config.pre_gating = False
config.post_gating = True
# ── Train ─────────────────────────────────────────────────────────────────────

print(f'Training with seed {config.env_seed}')
logger, model, config, figs = train_model(
    config, seed=0, save_models=False, load_models=False,
    run_test_phase=True,
)

# ── Plot ──────────────────────────────────────────────────────────────────────
#%%
if config.dataset_name == 'seq_learn':
    fig = plot_corrects_seq_learn(logger, config)
else:
    # Full overview: task structure, raw behaviour, latent dynamics, gradient signal
    panel_order = ['task_illustration_and_hierarchies', 'behavior', 'latent_2d', 'loss' ]
    fig = plot_logger_panels(logger, config, panel_order, x2=None, annotate_phases='behavior')
    x_start = config.passive_phase_length+config.blocked_phase_length//2
    x_end   = x_start + config.passive_phase_length+config.blocked_phase_length
    fig = plot_logger_panels(logger, config, panel_order[1:], x1=x_start, x2=x_end, annotate_phases=None)
    for ax in fig.axes:
        ax.set_xlim(0, 1600)


# ── Arena plot ────────────────────────────────────────────────────────────────
#
# Alignment note (predict_first_frame=False):
#   The model receives inputs[:, :-1, :] and produces outputs aligned with
#   inputs[:, 1:, :].  After logging with stride=1:
#       oi[t]  is the model's prediction FOR ii[t]  (made when seeing ii[t-1])
#   Therefore at cue t, the prediction for the upcoming attack is oi[t+1].

def plot_arena_trials(logger, config, t_start=0, t_end=None, same_block_only=True, ax=None):
    """Plot observations and predictions on the 2-D circular arena.

    Cue timesteps (color one-hot active) are paired with the following outcome
    timestep (x, y attack position). Points are colored by shield color.
    Observations: filled circles; model predictions: crosses (same color, lower alpha).

    Parameters
    ----------
    t_start, t_end : int
        Timestep range to draw from (flat, post-concatenation indices).
    same_block_only : bool
        If True, restrict to the context block that contains t_start
        (i.e. all timesteps where llcid == llcid[t_start]).
    """
    ii = np.concatenate(logger.inputs, axis=0).reshape(-1, config.input_size)
    ll = np.concatenate(logger.llcids, axis=0).reshape(-1)

    has_preds = bool(logger.predicted_outputs)
    if has_preds:
        oi = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1, config.output_size)

    if t_end is None:
        t_end = len(ii)
    t_end = min(t_end, len(ii) - 2)  # need t+1 for outcome AND t+1 for pred

    if same_block_only:
        # llcid repeats the same value (rotation angle) across many blocks, so
        # value-equality would match all blocks at that rotation.  Instead, find
        # the single contiguous run of constant llcid that contains t_start.
        changes = np.flatnonzero(np.diff(ll)) + 1  # indices where llcid changes
        block_starts = np.concatenate([[0], changes])
        block_ends   = np.concatenate([changes, [len(ll)]])
        bi = int(np.searchsorted(block_starts, t_start, side='right')) - 1
        valid = np.arange(int(block_starts[bi]), min(int(block_ends[bi]), t_end))
        valid = valid[valid >= t_start]
    else:
        valid = np.arange(t_start, t_end)

    # keep only cue timesteps: color one-hot non-zero in first n_colors dims
    nc = config.n_colors
    cue_ts = [t for t in valid if ii[t, :nc].sum() > 0.5]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    else:
        fig = ax.get_figure()

    # draw arena circle
    theta = np.linspace(0, 2 * np.pi, 200)
    r = getattr(config, 'target_radius', 0.5)
    ax.plot(r * np.cos(theta), r * np.sin(theta), color='lightgrey', linewidth=1, zorder=0)

    colors = plt.get_cmap('tab10', nc)

    for t in cue_ts:
        color_idx = int(np.argmax(ii[t, :nc]))
        c = colors(color_idx)
        obs_xy  = ii[t + 1, -2:]   # outcome frame carries x, y
        ax.scatter(*obs_xy, color=c, s=12, alpha=0.35, marker='x', linewidths=1, zorder=1)
        if has_preds:
            pred_xy = oi[t + 1, -2:]   # oi[t+1] is the prediction FOR the outcome frame
            ax.scatter(*pred_xy, color=c, s=15, alpha=0.6, zorder=2)

    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    rotation_deg = np.degrees(ll[t_start]) if cue_ts else '?'
    ax.set_title(f'rotation ≈ {rotation_deg:.0f}°', fontsize=8)
    ax.axhline(0, color='lightgrey', linewidth=0.5, zorder=0)
    ax.axvline(0, color='lightgrey', linewidth=0.5, zorder=0)
    return fig

fig, ax = plt.subplots(1,2, figsize=(7, 3.5))
t_start = logger.phases[2][1] if len(logger.phases) > 1 else (len(logger.inputs) - 2*  config.block_size) 
print(f'Plotting blocks at timesteps: {t_start} and {t_start + config.block_size}')
_=plot_arena_trials(logger, config, t_start=t_start, ax=ax[0])
plot_arena_trials(logger, config, t_start=t_start+config.block_size, ax=ax[1])


# ── Debug: responses to one color over time ────────────────────────────────
def extract_color_trials(logger, config, color_idx=0):
    """Return a dict with all trials where shield color=color_idx was shown.

    Keys
    ----
    't'        : logged timestep of the cue frame
    'obs_xy'   : (N,2) observed attack positions  (from outcome frame ii[t+1])
    'pred_xy'  : (N,2) model predictions           (from oi[t+1], same alignment)
    'llcid'    : (N,)  rotation context at cue
    'trial_no' : (N,)  sequential trial index (cue occurrences of this color)
    """
    ii = np.concatenate(logger.inputs, axis=0).reshape(-1, config.input_size)
    ll = np.concatenate(logger.llcids, axis=0).reshape(-1)
    has_preds = bool(logger.predicted_outputs)
    if has_preds:
        oi = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1, config.output_size)

    nc = config.n_colors
    # cue frames for this color: one-hot at color_idx is 1, within bounds
    cue_ts = [t for t in range(len(ii) - 1)
              if ii[t, color_idx] > 0.5 and ii[t, :nc].sum() > 0.5]

    ts       = np.array(cue_ts)
    obs_xy   = ii[ts + 1, -2:]                            # outcome frame
    pred_xy  = oi[ts + 1, -2:] if has_preds else None    # prediction FOR outcome frame
    llcids   = ll[ts]
    trial_no = np.arange(len(ts))

    return dict(t=ts, obs_xy=obs_xy, pred_xy=pred_xy, llcid=llcids, trial_no=trial_no)


def plot_color_response_over_time(logger, config, color_idx=0):
    """Show how model predictions for one shield color evolve across training."""
    d = extract_color_trials(logger, config, color_idx=color_idx)
    if d['pred_xy'] is None:
        print('No predicted_outputs logged.'); return

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # ── x and y prediction vs observation over trial index ─────────────────
    ax = axes[0]
    ax.plot(d['trial_no'], d['obs_xy'][:, 0], '.', color='tab:grey',  alpha=0.5, ms=3, label='obs x')
    ax.plot(d['trial_no'], d['pred_xy'][:, 0], '.', color='tab:blue', alpha=0.7, ms=3, label='pred x')
    ax.plot(d['trial_no'], d['obs_xy'][:, 1], '.', color='tab:orange', alpha=0.5, ms=3, label='obs y')
    ax.plot(d['trial_no'], d['pred_xy'][:, 1], '.', color='tab:red',   alpha=0.7, ms=3, label='pred y')
    # mark context switches
    switches = np.where(np.diff(d['llcid']) != 0)[0]
    for s in switches:
        ax.axvline(s, color='k', linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlabel('Trial # (color appearances)')
    ax.set_ylabel('Position')
    ax.set_title(f'Color {color_idx} — x/y over time', fontsize=8)

    # ── prediction error over trial index ──────────────────────────────────
    ax = axes[1]
    err = np.linalg.norm(d['pred_xy'] - d['obs_xy'], axis=1)
    ax.plot(d['trial_no'], err, '.', color='tab:grey', alpha=0.5, ms=3)
    ax.plot(d['trial_no'], np.convolve(err, np.ones(10)/10, mode='same'), color='k', linewidth=1)
    for s in switches:
        ax.axvline(s, color='k', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('Trial # (color appearances)')
    ax.set_ylabel('||pred − obs||')
    ax.set_title(f'Color {color_idx} — prediction error', fontsize=8)

    plt.tight_layout()
    return fig, d


fig_debug, d = plot_color_response_over_time(logger, config, color_idx=0)