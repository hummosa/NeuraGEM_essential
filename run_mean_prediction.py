"""
run_mean_prediction.py — Mean-prediction variant of the contextual switching task.

Task
────
Same two-context Gaussian setup as the 1D contextual switching task, but the
network is trained to output the latent mean (context value) rather than predict
the next observation.

How it works
────────────
Each dataset timestep is 2D: [observation, ground_truth_mean].
  - input_feed_mask  = [1, 0]: the model only sees the observation (dim 0);
    the mean (dim 1) is zeroed out before the forward pass so the model cannot cheat.
  - output_loss_mask = [0, 1]: loss is computed only on dim 1, so the network
    learns to produce the mean in its second output dimension.

The latent variable Z still adapts via LU to help the model infer context.
"""

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

#%%
import plot_style
plot_style.set_plot_style()

from functions_and_utils import *
from configs import MeanPredictionConfig
from train_and_infer_functions import *


# ── Configure ─────────────────────────────────────────────────────────────────

config = MeanPredictionConfig(experiment_to_run='figure')

config.default_std = 0.4          # observation noise (lower = easier; higher = harder)
config.save_model  = False
config.blocked_phase_length = 400
config.test_block_size = 20
config.seq_len = 3
config.no_of_steps_in_latent_space = 1
##
config.LU_optimizer = 'sgd'
config.l2_loss = 8e-4
config.LU_lr   = 0.9
##
# config.l2_loss = 1e-4
# config.LU_lr   = 0.4
##
# config.l2_loss = 1e-5
# config.LU_lr   = 0.1

# ── Train ─────────────────────────────────────────────────────────────────────

print(f'Training mean-prediction task with seed {config.env_seed}')
logger, model, config, figs = train_model(
    config, seed=config.env_seed, save_models=False, load_models=False,
)

# ── Plot ──────────────────────────────────────────────────────────────────────

# logger.outputs dim 1 should track logger.llcids after learning converges.
# logger.outputs dim 0 is unconstrained (no loss gradient on it).

panel_order = ['behavior', 'latent_2d', 'corrects']
fig = plot_logger_panels(logger, config, panel_order, x2=None,
                         annotate_phases='behavior', width=5, legends=False)

print(f'Export path: {config.export_path}')

# ── Quick sanity checks (uncomment to run) ────────────────────────────────────
# Ablation: set input_feed_mask=[1,1] so model sees the true mean on input dim 1.
# If the mask is working, the model should then "cheat" perfectly (output ≈ input dim 1).
#
# config_cheat = MeanPredictionConfig()
# config_cheat.input_feed_mask = [1, 1]   # let the mean through
# logger_cheat, _, _, _ = train_model(config_cheat, seed=42, save_models=False, load_models=False)
