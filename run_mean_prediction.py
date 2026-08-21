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

config.default_std = 0.3          # observation noise (lower = easier; higher = harder)
config.save_model  = False
config.blocked_phase_length = 1000
config.test_block_size = 20
config.seq_len = 4
config.output_loss_mask = [0, 1] # IMPORTANT: controls whether model learns to predict the ground truth latent mean (dim 1) or the next observation (dim 0) or both.
z_steps = 0
config.no_of_steps_in_latent_space = z_steps

attempt_two_attractors_as_before = True
if attempt_two_attractors_as_before:
    config.default_std = 0.1          # observation noise (lower = easier; higher = harder)
    config.blocked_phase_length =5000
    # config.WU_lr = 3e-4
    config.block_size = 20
    z_steps = 1
    config.no_of_steps_in_latent_space = z_steps
    config.seq_len = 3
    config.output_loss_mask = [1, 0] # IMPORTANT: controls whether model learns to predict the ground truth latent mean (dim 1) or the next observation (dim 0) or both.
    ##
##
# config.Z_optimizer = 'adam'
# config.Z_decay = 8e-4
# config.Z_lr   = 0.9
# config.loss_reduction_LU = 'sum'    # how to reduce per-element loss before backward: 'sum' or 'mean'

##
_z_decay_for_lr = lambda lr: 1e-3 * lr ** 2
# config.Z_lr   = 1.
config.Z_lr   = 0.4

config.Z_decay = _z_decay_for_lr(config.Z_lr)
##
# config.Z_decay = 1e-5
# config.Z_lr   = 0.1

# ── Train ─────────────────────────────────────────────────────────────────────

print(f'Training mean-prediction task with seed {config.env_seed}')
logger, model, config, figs = train_model(
    config, seed=config.env_seed,
)

# ── Plot ──────────────────────────────────────────────────────────────────────

# logger.outputs dim 1 should track logger.context_ids after learning converges.
# logger.outputs dim 0 is unconstrained (no loss gradient on it).

panel_order = ['behavior', 'latent_2d', 'corrects']
fig = plot_logger_panels(logger, config, panel_order, x2=None,
                         annotate_phases='behavior', width=5, legends=False, dpi=200)
fig.savefig(config.export_path + 'mean_prediction_results.pdf', bbox_inches='tight')
print(f'exported to: {config.export_path + "mean_prediction_results.pdf"}')

# ── Quick sanity checks (uncomment to run) ────────────────────────────────────
# Ablation: set input_feed_mask=[1,1] so model sees the true mean on input dim 1.
# If the mask is working, the model should then "cheat" perfectly (output ≈ input dim 1).
#
# config_cheat = MeanPredictionConfig()
# config_cheat.input_feed_mask = [1, 1]   # let the mean through
# logger_cheat, _, _, _ = train_model(config_cheat, seed=42, save_models=False, load_models=False)


#%% Testing context inference dynamics on first evidence and a switch
# Gradual-transition variant: the generative mean ramps at morph_rate per
# timestep (starting from 0.5) rather than switching abruptly.
# This lets us probe how quickly the model tracks a slowly drifting context.
import copy
from configs import GradualMeanPredictionConfig

model2 = copy.deepcopy(model)
model2.reset_latent()  # start with a fresh slate for the latent variable Z, to see how it learns to track the drifting context from scratch.
print(f'latent variable Z reset to: {model2.Z}')
config2 = GradualMeanPredictionConfig(experiment_to_run='figure')
for k in config.__dict__.keys():
    if k in ['input_feed_mask', 'output_loss_mask', 'Z_lr', 'Z_decay']:
        setattr(config2, k, config.__dict__[k])
config2.morph_rate    = 0.04   # mean steps 0.05 per timestep toward the target
config2.env_seed      = 3#15
config2.default_std   = 0.01
# config2.default_std   = config.default_std

config2.first_block_mean = 0.8   # start high
# or
# config2.first_block_mean = 0.2   # start low
# config2.training_data_means = [0.2, 0.8]   # same two contexts as the main task
config2.no_of_steps_in_latent_space = z_steps
config2.no_of_steps_in_weight_space = 0
config2.no_of_blocks  = 4
# Each block needs at least ceil(0.6 / morph_rate) = 12 steps to fully ramp;
# which gives estimated timesteps gives a clear plateau after the ramp.
config2.block_size    = np.ceil(0.6 / config2.morph_rate) #25
config2.blocked_phase_length = config2.no_of_blocks * config2.block_size
# config2.test_block_size = 20
config2.seq_len       = config.seq_len
config2.log_initial_burn_in_timesteps = True
model2.config = config2

logger2, model2, config2, figs2 = train_model(
    config2, seed=config2.env_seed, pretrained_model=model2, run_test_phase=False,
)

panel_order = ['behavior', 'latent_2d', 'corrects']
fig = plot_logger_panels(logger2, config2, panel_order, x2=None,
                         annotate_phases='behavior', width=5, legends=False, dpi=200)

fig = plot_logger_panels(logger2, config2, ['behavior'], x2=70,
                         annotate_phases=None, width=5, legends=True, dpi=200)

config2.first_block_mean = 0.2   # start low
model2.reset_latent()  # start with a fresh slate for the latent variable Z, to see how it learns to track the drifting context from scratch.
logger2, model2, config2, figs2 = train_model(
    config2, seed=config2.env_seed, pretrained_model=model2, run_test_phase=False,
)
fig = plot_logger_panels(logger2, config2, ['behavior'], x2=70,
                         annotate_phases=None, width=5, legends=True, dpi=200)

# %%
