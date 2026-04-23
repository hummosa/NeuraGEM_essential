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

import plot_style
plot_style.set_plot_style()

from functions_and_utils import *
from configs import *
from train_and_infer_functions import *


# ── Choose and configure the task ─────────────────────────────────────────────

config = ContextualSwitchingTaskConfig(experiment_to_run='figure')
# config = SeqLearnConfig(experiment_to_run='few_long_blocks')
config.dataset_name = 'contextual_switching_task_2D'
config.input_size = 2
config.output_size = 2
config.latent_dims = [2]


config.default_std = 0.1   # observation noise (paper uses 0.3; 0.1 is clearer for visualisation)
config.log_weights = False
config.save_model = False
config.load_saved_model = False

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
    panel_order = ['task_illustration_and_hierarchies', 'behavior', 'latent_2d', 'gradients']
    fig = plot_logger_panels(logger, config, panel_order, x2=None, annotate_phases='behavior')

print(f'Export path: {config.export_path}')
