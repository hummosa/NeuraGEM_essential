import os
import torch
import numpy as np

# Abbreviations used throughout: LU = Latent Update (Z optimization)  |  WU = Weight Update (standard BPTT)

class Config:
    def __init__(self):
        self._run_name = 'default'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Core Training ──────────────────────────────────────────────────────
        self.env_seed = 1
        self.epochs = 1
        self.no_of_steps_in_weight_space = 1  # WU gradient steps per batch
        self.no_of_steps_in_latent_space = 1  # LU gradient steps per batch
        self.predict_first_frame = True  # if True, predict input at t=0; else start from t=1 and predict next frame
        self.update_latent_before_weights = False  # if True, run LU before WU each batch; else run WU first
        # ── Latent Variable (Z) ────────────────────────────────────────────────
        # Z has shape (batch, seq_len, Z_dim) where Z_dim = product(latent_dims).
        self.latent_dims = [2]    # e.g. [4] for Z_dim=4; [2, 2] for Z_dim=4 split into 2 chunks of 2
        self.latent_chunks = 1    # number of independently-activated sub-vectors within Z
        self.latent_activation = 'softmax'   # 'softmax', 'sigmoid', or 'none'; applied before Z is used
        self.softmax_temp = 1             # temperature for softmax (higher = more uniform)
        self.pass_previous_latent = True  # carry Z across batches so LU warm-starts from prior context
        self.what_latent_to_use = 'self'  # 'self' (learn Z), 'taskID' (oracle), 'uniform', or 'zeros'

        # ── Latent Optimizer (LU) ──────────────────────────────────────────────
        self.LU_lr = 0.4
        self.LU_optimizer = 'Adam'        # 'Adam', 'AdamW', or 'SGD'
        self.LU_Adam_betas = (0.6, 0.7)   # For faster dynamics (0.9, 0.999)
        self.LU_momentum = 0.0            # only used when LU_optimizer='SGD'
        self.l2_loss = 0.0001                  # L2 regularization weight on Z (weight decay)
        self.loss_reduction_LU = 'mean'    # how to reduce per-element loss before backward: 'sum' or 'mean'
        # Gradient aggregation across the time dimension of Z before each LU optimizer step.
        # Options: 'exponential_increase' (recent steps weighted more), 'average', 'last', 'none'.
        self.latent_aggregation_op = 'exponential_increase'
        # exponential_increase filter weights per chunk: steepness=0 → uniform (same as 'average');
        # steepness=2 → mild recency bias; steepness=40 → focus on last few steps only.
        self.exponential_increase_steepness = [2] * self.latent_chunks   # one value per chunk
        self.exponential_increase_multipliers = [1] * self.latent_chunks  # scale applied after normalizing

        # ── Weight Optimizer (WU) ──────────────────────────────────────────────
        self.WU_lr = 0.001
        self.WU_optimizer = 'Adam'
        self.WU_momentum = 0.0
        self.loss_reduction_WU = 'mean'

        # ── Architecture ───────────────────────────────────────────────────────
        self.rnn_type = 'lstm'    # 'lstm', 'gru', or 'rnn'. LSTM recommended (Hochreiter 1997).
        # Multiplicative gating: Z is projected through a sparse random mask and multiplied into
        # the RNN hidden state. This lets context modulate dynamics without changing inputs.
        self.use_mul_gating = True
        self.pre_gating = True    # apply gate before each RNN step (recommended)
        self.post_gating = False  # apply gate after each RNN step; can combine with pre_gating
        # Additive gating: Z is concatenated to the input instead. Alternative to mul_gating.
        self.use_add_gating = False
        self.use_input_attention = False
        self.P_gates_bernoulli_prob = 0.3  # fraction of Z→hidden connections enabled in the mask

        # ── Training Phases / Curriculum ───────────────────────────────────────
        self.add_passive_learning_phase = False   # Phase 1: WU only, no LU (pre-trains weights)
        self.passive_phase_length = 200
        self.add_interleaved_phase = True         # Phase 2: randomly mixed contexts, WU + LU
        self.interleaved_phase_length = 500
        self.latent_updates_during_shuffle = True
        self.add_blocked_phase = True             # Phase 3: blocked contexts, WU + LU
        self.blocked_phase_length = 1000
        self.shuffle_or_interleave = 'interleave' # ordering within interleaved phase
        self.random_transition_shuffle_or_interleave = 'shuffle'
        self.start_always_on_the_same_block = False  # if True, always start with context A

        # ── Loss masking ───────────────────────────────────────────────────────
        # List of 0/1 of length output_size, or None to use all dims.
        # E.g. [0,0,0,0,0,1,1] predicts only xy and ignores the color one-hot.
        self.output_loss_mask = None

        # ── Noise Injection ────────────────────────────────────────────────────
        self.add_noise_to_input = False
        self.noise_std = 0.0

        # ── Logging ────────────────────────────────────────────────────────────
        self.log_weights = False
        self.log_hidden_states = False
        self.log_end_weights = False
        self.log_initial_burn_in_timesteps = False  # include the first seq_len timesteps in logs
        self.eval_z_space_interval = 0  # freeze model and snapshot Z space every N batches; 0 to skip

        # ── I/O ────────────────────────────────────────────────────────────────
        self.save_model = False
        self.load_saved_model = False
        self.export_folder = './exports/'
        self.export_path = ''

        # ── Internal (do not modify directly) ──────────────────────────────────
        self._allow_latent_updates = True  # set False during passive phase by train_model()

        # ── Experimental / unused in standard NeuraGEM runs ────────────────────
        self.latent_type = '1d_latent'       # only '1d_latent' supported currently
        self.save_latent_updates = False
        self.rl_task = False
        self.use_COIN_channel_experiment = False
        self.add_washout_phase = False

        self.initialize_common_config()

    def initialize_common_config(self):
        """Set task-specific dimensions. Override in subclasses."""
        self.input_size = None    # set in subclass
        self.hidden_size = None   # set in subclass
        self.output_size = None   # set in subclass
        self.seq_len = None       # set in subclass
        self.stride = None        # set in subclass
        self.dataset_name = ''    # set in subclass

        self.no_of_blocks = 4
        self.batch_size = 1
        self.epochs = 1
        self.block_size = 100

    @property
    def run_name(self):
        return self._run_name

    @run_name.setter
    def run_name(self, value):
        self._run_name = value
        self.update_export_path()

    def update_export_path(self):
        self.export_path = f'{self.export_folder}{self.dataset_name}/{self._run_name}/'
        os.makedirs(self.export_path, exist_ok=True)

    def reconfigure_for_prediction(self, experiment_to_run):
        """Switch to test/inference mode: freeze weights, adjust dataset size.

        Defaults to 4 blocks × 200 timesteps. Override either dimension before calling:
            config.test_no_of_blocks = 8   # more blocks, block_size stays 200
            config.test_block_size   = 500  # longer blocks, no_of_blocks stays 4
        """
        self.what_latent_to_use = 'self'
        self.no_of_steps_in_weight_space = 0
        self.block_size  = getattr(self, 'test_block_size',   self.block_size)
        self.no_of_blocks = getattr(self, 'test_no_of_blocks', 4)
        self.update_export_path()

    def _validate(self):
        """Check that coupled parameters are consistent. Called at end of subclass __init__."""
        assert len(self.exponential_increase_steepness) == self.latent_chunks, (
            f"exponential_increase_steepness must have one value per latent_chunk. "
            f"Got {len(self.exponential_increase_steepness)} values for {self.latent_chunks} chunks. "
            f"Set exponential_increase_steepness = [value] * latent_chunks."
        )
        assert len(self.exponential_increase_multipliers) == self.latent_chunks, (
            f"exponential_increase_multipliers must have one value per latent_chunk. "
            f"Got {len(self.exponential_increase_multipliers)} values for {self.latent_chunks} chunks."
        )
        Z_dim = int(np.prod(self.latent_dims))
        assert Z_dim % self.latent_chunks == 0, (
            f"Z_dim (product of latent_dims={self.latent_dims} → {Z_dim}) "
            f"must be divisible by latent_chunks={self.latent_chunks}."
        )


class ContextualSwitchingTaskConfig(Config):
    """
    Configuration for the 1D Gaussian context-switching task.

    Two contexts A and B produce scalar observations from Gaussian(mean_A, std) and
    Gaussian(mean_B, std) respectively. The model receives no context label; it must
    infer context from prediction errors and update Z accordingly.

    Usage:
        config = ContextualSwitchingTaskConfig()           # paper figure preset
        config = ContextualSwitchingTaskConfig('tweaking') # exploration preset
        config.LU_lr = 0.5                                  # override any parameter
    """
    def __init__(self, experiment_to_run='figure'):
        super().__init__()
        self.dataset_name = 'contextual_switching_task'
        self.experiment_to_run = experiment_to_run

        # ── Task dimensions ────────────────────────────────────────────────
        self.input_size = 1       # single scalar observation per timestep
        self.output_size = 1
        self.hidden_size = 64
        self.seq_len = 4
        self.stride = 1

        # ── Task data generation ───────────────────────────────────────────
        self.training_data_means = [0.2, 0.8]  # Gaussian means for context A and B
        self.default_std = 0.1                  # observation noise std
        self.correlated_noise = False           # if True, use AR(1) temporally correlated noise
        self.noise_correlation_tau = 10         # autocorrelation time in timesteps
        self.task_length = 1
        self.block_duration_distribution = 'geometric'
        self.use_high_task_structure = False
        self.latent_change_interval = 1
        self.high_level_latent_change_interval_in_blocks = 3
        self.start_always_on_the_same_block = True  # always begin with context A

        # ── Training schedule ──────────────────────────────────────────────
        self.epochs = 1
        self.batch_size = 1
        self.no_of_blocks = 200
        self.block_size = 25 * self.task_length
        self.passive_epochs = 1
        self.blocked_phase_length = 850

        # ── Latent optimizer ──────────────────────────────────────────────
        self.latent_activation = 'softmax'
        self.latent_aggregation_op = 'exponential_increase'
        self.pass_previous_latent = True
        self.no_of_steps_in_latent_space = 1
        self.LU_lr = 0.8
        self.LU_Adam_betas = (0.6, 0.7)
        self.l2_loss = 0.0001
        self.LU_optimizer = 'Adam'
        self.exponential_increase_steepness = [2] * self.latent_chunks

        # ── Weight optimizer ──────────────────────────────────────────────
        self.WU_lr = 0.001
        self.loss_reduction_LU = 'mean'
        self.loss_reduction_WU = 'mean'

        # ── OOD challenge block (optional) ─────────────────────────────────
        # One block presents observations from an unseen mean to test generalization.
        self.out_of_distribution_challenge = {
            'use_challenge': True,
            'block_no': 15,
            'duration': 50,
            'mean': 0.5,   # midpoint between A and B — unseen during training
            'std': 0.2,
        }

        # ── Analysis windows ───────────────────────────────────────────────
        self.pre_window = 3   # timesteps before context switch for error-strip analysis
        self.post_window = 20
        self.length_of_opposite_block_sequence = 4  # for COIN-style test truncation

        if self.use_add_gating:
            self.input_size += int(np.prod(self.latent_dims))

        self.update_export_path()

        # Tweaking-specific overrides (exploration preset differs in 3 values)
        if experiment_to_run in ('tweaking', 'weight_grads_comp'):
            self.LU_lr = 0.5
            self.WU_lr = 0.005
            self.blocked_phase_length = 1200

        self._validate()

    def reconfigure_for_prediction(self, experiment_to_run):
        """Switch to inference mode for the test phase."""
        self.what_latent_to_use = 'self'
        self.batch_size = 1
        self.epochs = 1
        self.no_of_steps_in_weight_space = 0
        self.add_noise_to_input = False
        self.block_size   = getattr(self, 'test_block_size',  self.block_size)
        self.no_of_blocks = getattr(self, 'test_no_of_blocks', 4)
        self.limited_testing_samples_no = int(2000 / self.batch_size)


class SeqLearnConfig(Config):
    """
    Configuration for the Beukers et al. (2024) blocked sequence-learning task.

    Two task types each with two transition patterns over 10 one-hot states.
    Reference: https://osf.io/preprints/psyarxiv/9bptj
    """
    def __init__(self, experiment_to_run='few_long_blocks'):
        super().__init__()
        self.dataset_name = 'seq_learn'
        self.experiment_to_run = experiment_to_run

        # ── Task dimensions ────────────────────────────────────────────────
        self.input_size = 10      # one-hot state encoding
        self.output_size = 10
        self.hidden_size = 32
        self.stride = 1
        self.task_length = 6      # fixed sequence length
        self.observation_scale = 1  # scale applied to one-hot state vectors

        # ── Task-specific flags ────────────────────────────────────────────
        self.seq_learn_use_deterministic_transition_2 = False
        self.plot_diagnostic_plots = False
        self.plot_dynamic_optim_gif = False
        self.grad_model_type = 'none'

        if self.use_add_gating:
            self.input_size += int(np.prod(self.latent_dims))

        if experiment_to_run == 'few_long_blocks':
            self._apply_few_long_blocks_preset()

        self.limited_testing_samples_no = int(1000 / self.batch_size)
        self.update_export_path()
        self._validate()

    def _apply_few_long_blocks_preset(self):
        """Long blocked training with average gradient aggregation."""
        self.seq_len = 18
        self.latent_activation = 'softmax'
        self.latent_aggregation_op = 'average'
        self.pass_previous_latent = False
        self.no_of_steps_in_latent_space = 10
        self.l2_loss = 0
        self.LU_lr = 0.1
        self.LU_optimizer = 'Adam'
        self.WU_lr = 0.001
        self.loss_reduction_LU = 'mean'
        self.loss_reduction_WU = 'mean'
        self.hidden_size = 32
        self.P_gates_bernoulli_prob = 0.5
        self.epochs = 1
        self.batch_size = 1
        self.task_length = 6
        self.block_size = 20 * self.task_length
        self.blocked_phase_length = 1200
        self.interleaved_phase_length = self.blocked_phase_length
        self.add_passive_learning_phase = False
        self.add_interleaved_phase = True
        self.latent_updates_during_shuffle = False
        self.passive_phase_length = 500
        self.no_of_blocks = self.blocked_phase_length // self.block_size
        self.add_noise_to_input = False
        self.noise_std = 0.0
        self.exponential_increase_steepness = [2] * self.latent_chunks

    def reconfigure_for_prediction(self, experiment_to_run):
        """Switch to inference mode for the test phase."""
        self.what_latent_to_use = 'self'
        self.no_of_steps_in_weight_space = 0
        self.block_size   = getattr(self, 'test_block_size',   self.block_size)
        self.no_of_blocks = getattr(self, 'test_no_of_blocks', 4)
        self.limited_testing_samples_no = int(500 / self.batch_size)


class RotatingTargetsConfig(Config):
    """
    Configuration for the rotating-targets predictive-inference task (Yu et al. 2025).

    5 shield colors each have a target on a circular arena. All targets rotate together
    at unsignaled state-block boundaries. Each trial produces two timesteps:
        cue     [color_onehot, 0, 0]  →  outcome  [0...0, attack_x, attack_y]
    Frame prediction (predict_first_frame=False) means the model at the cue step must
    predict where the attack lands — the primary learning signal.

    Train on config.train_rotations; test generalization on config.test_rotations.
    """

    def __init__(self):
        super().__init__()
        self.experiment_to_run = 'default'

        # ── Task structure ─────────────────────────────────────────────────
        self.n_colors = 5
        self.n_miniblocks_per_state_block = 50
        self.noise_std = 0.04
        self.target_radius = 0.5

        # ── Rotation schedule ──────────────────────────────────────────────
        self.train_rotations = [0.0, 90.0]  # degrees; cycling across state-blocks
        self.test_rotations  = []           # novel angles for transfer test; empty = use train_rotations

        # ── Dataset keys ──────────────────────────────────────────────────
        self.dataset_name      = 'rotating_targets'
        self.test_dataset_name = 'rotating_targets_test'

        # ── Dimensions (derived) ──────────────────────────────────────────
        self.input_size  = self.n_colors + 2  # 7
        self.output_size = self.n_colors + 2  # 7
        self.hidden_size = 64
        self.output_loss_mask = [0,0,0,0,0,1,1] #predicts only xy and ignores the color one-hot.
        # ── Sequence windowing ────────────────────────────────────────────
        # self.seq_len = self.n_colors * 2  # 10: one full mini-block (cue+outcome per color)
        self.seq_len = 5
        self.stride  = 1

        # ── Block structure ───────────────────────────────────────────────
        self.block_size  = self.n_miniblocks_per_state_block * self.n_colors  # 80
        # self.no_of_blocks = 40 # does not do anything 
        self.block_duration_distribution = 'fixed'  # 'fixed' or 'geometric'; geometric adds variability to block lengths

        # ── Latent variable ───────────────────────────────────────────────
        self.latent_dims = [2]   # scalar Z = rotation angle in radians
        self.LU_lr = 0.2
        self.l2_loss = 0.0001
        self.latent_chunks = 1
        self.latent_activation = 'softmax'
        self.no_of_steps_in_latent_space = 1
        self.exponential_increase_steepness = [2]
        self.exponential_increase_multipliers = [1]

        # ── Training ──────────────────────────────────────────────────────
        self.predict_first_frame = False
        self.pass_previous_latent = True
        self.batch_size = 1
        self.epochs = 1
        self.WU_lr = 0.001
        self.loss_reduction_LU = 'mean'
        self.loss_reduction_WU = 'mean'

        # ── Test-phase overrides (used by reconfigure_for_prediction) ──────
        # Set test_no_of_steps_in_weight_space=1 to keep WU active during Phase 3.
        self.test_no_of_steps_in_weight_space = 0   # 0 = freeze weights (standard test)
        self.test_no_of_blocks = 6                 # blocks to run in Phase 3

        self.update_export_path()
        self._validate()

    def reconfigure_for_prediction(self, experiment_to_run):
        """Switch to test/inference mode: freeze weights, adjust dataset size."""
        self.what_latent_to_use = 'self'
        self.no_of_steps_in_weight_space = getattr(self, 'test_no_of_steps_in_weight_space', 0)
        # fall back to the training block_size so mini-block structure is consistent
        self.block_size   = getattr(self, 'test_block_size',   self.block_size)
        self.no_of_blocks = getattr(self, 'test_no_of_blocks', 4)
        self.update_export_path()

class MeanPredictionConfig(ContextualSwitchingTaskConfig):
    """
    Variant of the 1D contextual switching task where the network predicts the
    latent mean rather than the next observation.

    The dataset augments each timestep to 2D: [observation, ground_truth_mean].
    input_feed_mask hides the mean from the model input (no cheating).
    output_loss_mask trains only on the mean-prediction output dimension.
    """

    def __init__(self, experiment_to_run='figure'):
        super().__init__(experiment_to_run)
        self.dataset_name    = 'mean_prediction'
        self.input_size      = 2   # [observation, mean]
        self.output_size     = 2
        self.input_feed_mask  = [1, 0]  # zero out mean dim before model sees input
        self.output_loss_mask = [0, 1]  # compute loss only on mean-prediction dim
        self.update_export_path()


# Backwards-compatibility aliases
seq_learnConfig = SeqLearnConfig
CSWConfig = SeqLearnConfig
