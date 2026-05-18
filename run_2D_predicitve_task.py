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
config.passive_phase_length = 500
config.blocked_phase_length = 1000   # ← 
config.n_miniblocks_per_state_block = 50 // config.n_colors  # 10 
config.block_size  = config.n_miniblocks_per_state_block * config.n_colors  # 80
config.test_no_of_blocks   = 4       # ← blocks in Phase 3 test
# config.what_latent_to_use = 'taskID'
config.pre_gating = False # the rotating targets task worked not with pre_gating! Such bad interference. 
config.post_gating = not config.pre_gating

if config.pre_gating:
    config.Z_lr = 0.1
    config.Z_decay = 0.000_0011
else:
    config.Z_lr = .05
    config.Z_decay = 0.000_1


# config.seq_len = 4
config.WU_lr = 0.001
config.what_latent_to_use = 'self'

over_segment = True
if over_segment: # trying out changing params to make ng oversegment to shields
    config.latent_dims = [10]    # e.g. [4] for Z_dim=4; [2, 2] for Z_dim=4 split into 2 chunks of 2
    config.latent_chunks = 5    # number of independently-activated sub-vectors within Z
    config.latent_activation = 'softmax_chunked'   # 'softmax', 'sigmoid', or 'none'; applied before Z is used
    config.Z_lr = 1.
    config.Z_decay = 0.000_8
    config.seq_len = 2
    config.add_passive_learning_phase = False
    config.blocked_phase_length = 1500   # ← 
    config.block_size = 200
    # config.LU_optimizer = 'sgd' # 'adam' or 'sgd'; Adam's momentum seems to make it harder for the model to adapt quickly within a block, even with a low Z_lr.
    # config.Z_decay = 0.0
    # config.latent_aggregation_op = 'none'
    # config.update_latent_before_weights = True # whether to run the LU step before the WU step within each batch. This seems to help a lot with oversegmenting models, maybe by giving them a chance to adjust Z before the weights have to follow it.
    config.pre_gating = False
    config.post_gating = not config.pre_gating
run_rnn = False
if run_rnn:
    config.no_of_steps_in_latent_space = 0
    # config.block_size = 100 # because RNN struggles to adapt within the reported blocksize of 6-8 miniblocks. 
    config.WU_lr = 0.01 # to give the RNN a more realistic adaptation curve. Otherwise not fast enough.

# ── Arena plot ────────────────────────────────────────────────────────────────
#
# Alignment note (predict_first_frame=False):
#   The model receives inputs[:, :-1, :] and produces outputs aligned with
#   inputs[:, 1:, :].  After logging with stride=1:
#       oi[t]  is the model's prediction FOR ii[t]  (made when seeing ii[t-1])
#   Therefore at cue t, the prediction for the upcoming attack is oi[t+1].

def plot_arena_trials(logger, config, t_start=0, t_end=None, same_block_only=True, ax=None,
                      title=None, phase='train'):
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
    title : str, optional
        Override the auto-generated title. If None, shows 'rotation ≈ X°'.
    phase : str
        Label shown in the auto-title prefix when title is None.
        Use 'novel' for Phase 3a test blocks so the annotation is clear.
    """
    ii = np.concatenate(logger.inputs, axis=0).reshape(-1, config.input_size)
    ll = np.concatenate(logger.context_ids, axis=0).reshape(-1)

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
    if title is not None:
        ax.set_title(title, fontsize=8)
    else:
        rotation_deg = f'{np.degrees(ll[t_start]):.0f}' if cue_ts else '?'
        prefix = 'novel ' if phase == 'novel' else ''
        ax.set_title(f'{prefix}rotation ≈ {rotation_deg}°', fontsize=8)
    ax.axhline(0, color='lightgrey', linewidth=0.5, zorder=0)
    ax.axvline(0, color='lightgrey', linewidth=0.5, zorder=0)
    return fig


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
    ll = np.concatenate(logger.context_ids, axis=0).reshape(-1)
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
    context_ids   = ll[ts]
    trial_no = np.arange(len(ts))

    return dict(t=ts, obs_xy=obs_xy, pred_xy=pred_xy, llcid=context_ids, trial_no=trial_no)


def __main__(config):
    # ── Train ─────────────────────────────────────────────────────────────────────

    print(f'Training with seed {config.env_seed}')
    logger, model, config, figs = train_model(
        config, seed=0, save_models=False, load_models=False,
        run_test_phase=True,
    )

    # ── Plot ──────────────────────────────────────────────────────────────────────
    if config.dataset_name == 'seq_learn':
        fig = plot_corrects_seq_learn(logger, config)
    else:
        # Full overview: task structure, raw behaviour, latent dynamics, gradient signal
        panel_order = ['rotating_targets_behavior', 'latent_2d', 'loss' ]
        fig = plot_logger_panels(logger, config, panel_order, x2=None, annotate_phases='latent_2d', width= 5)
        x_start = config.passive_phase_length+config.blocked_phase_length//2
        x_end   = x_start + config.passive_phase_length+config.blocked_phase_length
        # fig = plot_logger_panels(logger, config, panel_order, x1=x_start, x2=x_end, annotate_phases=None)

    fig, ax = plt.subplots(1,2, figsize=(7, 3.5))
    t_start = logger.phases[2][1] if len(logger.phases) > 1 else (len(logger.inputs) - 2*  config.block_size) 
    print(f'Plotting blocks at timesteps: {t_start} and {t_start + config.block_size}')
    _=plot_arena_trials(logger, config, t_start=t_start, ax=ax[0])
    plot_arena_trials(logger, config, t_start=t_start+config.block_size, ax=ax[1])

    from rotating_targets_analysis import analyze_z_color_modulation
    results, fig3 = analyze_z_color_modulation(logger, config, phase='phase3a')
    # results['rotation_r2']      → how much Z encodes rotation
    # results['within_color_r2']  → how much Z leaks color identity within a block
    return logger, model, config, figs

if __name__ == '__main__':
    logger, model, config, figs = __main__(config)