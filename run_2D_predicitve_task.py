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
config.train_rotations = [0.0, 90.0]
config.test_rotations  = [45.0, 135.0, 225.0, 315.0]
logger, model, config, figs = train_model(config, seed=0)

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
    panel_order = ['task_illustration_and_hierarchies', 'behavior', 'latent_2d', 'loss' ]
    fig = plot_logger_panels(logger, config, panel_order, x2=None, annotate_phases='behavior')
