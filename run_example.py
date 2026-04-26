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
    
#%%
import plot_style
plot_style.set_plot_style()

from functions_and_utils import *
from configs import *
from train_and_infer_functions import *


# ── Choose and configure the task ─────────────────────────────────────────────

config = ContextualSwitchingTaskConfig(experiment_to_run='figure')
# config = SeqLearnConfig(experiment_to_run='few_long_blocks')

config.default_std = 0.1   # observation noise (paper uses 0.3; 0.1 is clearer for visualisation)
config.log_weights = False
config.save_model = False
config.load_saved_model = False
config.blocked_phase_length = 1000
config.no_of_steps_in_latent_space = 1
# config.l2_loss = 1e-4
# config.LU_lr = 0.3 # this is ideal
# config.l2_loss = 8e-4
# config.LU_lr = 0.7 # this is bad too fast.
# config.l2_loss = 1e-5
# config.LU_lr = 0.05 # this is bad too slow.
config.update_latent_before_weights = False
config.latent_aggregation_op = 'none'

# o τ=1 ≈ white noise, τ=20 gives ~20-timestep correlations. T
config.correlated_noise = True
config.noise_correlation_tau = 1  # longer = smoother noiseo

# ── Train ─────────────────────────────────────────────────────────────────────

print(f'Training with seed {config.env_seed}')
logger, model, config, figs = train_model(
    config, seed=config.env_seed, save_models=False, load_models=False,
)


# ── Plot ──────────────────────────────────────────────────────────────────────

if config.dataset_name == 'seq_learn':
    fig = plot_corrects_seq_learn(logger, config)
else:
    # Full overview: task structure, raw behaviour, latent dynamics, gradient signal
    panel_order = ['behavior', 'latent_2d', 'corrects']
    fig = plot_logger_panels(logger, config, panel_order, x2=None, annotate_phases='behavior')

fig = plot_logger_panels(logger, config, panel_order, x1=3500, x2=4000, annotate_phases='behavior')

print(f'Export path: {config.export_path}')
#%%
fig_analysis = plot_logger_analysis(logger, config,
    subplot_width=2, subplot_height=2,
    last_ts_in_a_block=15,
    aggregate_blocks=5,  
)
