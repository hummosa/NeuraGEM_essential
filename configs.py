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
        # LU steps during the test phase; None keeps the training value. Set to 1 for a condition
        # that trained with LU off (an oracle) but should infer Z at test, 0 to keep Z frozen.
        self.test_no_of_steps_in_latent_space = None
        self.predict_first_frame = True  # if True, predict input at t=0; else start from t=1 and predict next frame
        self.update_latent_before_weights = False  # if True, run LU before WU each batch; else run WU first
        # ── Latent Variable (Z) ────────────────────────────────────────────────
        # Z has shape (batch, seq_len, Z_dim) where Z_dim = product(latent_dims).
        self.latent_dims = [2]    # e.g. [4] for Z_dim=4; [2, 2] for Z_dim=4 split into 2 chunks of 2
        self.latent_chunks = 1    # number of independently-activated sub-vectors within Z
        self.latent_activation = 'softmax'   # 'softmax', 'sigmoid', or 'none'; applied before Z is used
        self.softmax_temp = 1             # temperature for softmax (higher = more uniform)
        self.pass_previous_latent = True  # carry Z across batches so LU warm-starts from prior context
        self.what_latent_to_use = 'self'  # 'self' (learn Z), 'context_ids' (oracle), 'uniform', or 'zeros'

        # ── Oracle latent (only read when what_latent_to_use='context_ids') ────
        # The dataset's context_ids carry the value of whichever experimental variable defines
        # the context (a mean, a rotation angle, a slot index). These three fields say how that
        # value becomes a Z vector; a task sets the one its encoding needs.
        #   'one_hot'    — context identity. Needs oracle_context_values: the ordered table of
        #                  possible values. Each id takes the slot of its nearest entry, so
        #                  Z_dim >= len(table). No metric between contexts.
        #   'normalized' — context magnitude. Needs oracle_context_range (lo, hi); Z_dim == 1.
        #                  Z = (value - lo) / (hi - lo), so lo → 0 and hi → 1.
        #   'circular'   — context magnitude on a ring. Needs oracle_context_range spanning one
        #                  full period; Z_dim == 2. Z = [(1+cos θ)/2, (1+sin θ)/2], no wraparound
        #                  discontinuity between hi and lo.
        self.oracle_context_encoding = 'one_hot'
        self.oracle_context_values = None   # list/array of context values, in slot order
        self.oracle_context_range = None    # (lo, hi) of the context variable

        # Oracle gate jitter: True (models.DEFAULT_ORACLE_GATE_JITTER), an explicit
        # (lo, hi) pair of positive scale factors, or None/False to disable.
        # 'one_hot' encodings only; anything else is ignored.
        #
        # The one-hot is scaled by a factor drawn uniformly from (lo, hi) BEFORE the softmax,
        # so the oracle gate arrives at a different sharpness on every trial. Without it the
        # oracle is the same vector on every training trial — softmax(one-hot / softmax_temp),
        # peak 0.405 with Z_dim=5 at temp 1 — and the weights are only ever calibrated at that
        # one sharpness. A later stage that infers Z freely will run sharper gates, where the
        # output overshoots its target, and a latent update descending squared error can then
        # lower its loss by flattening the gate rather than by pointing it anywhere useful.
        # Jitter removes that: the read-out is trained to emit the same magnitude at every
        # sharpness, so gate sharpness stops acting as a gain control and Z carries only
        # 'which context', which is what it is supposed to mean.
        #
        # One factor is drawn per forward pass, i.e. per trial, shared across its timesteps.
        # It is applied only while weights are plastic (no_of_steps_in_weight_space > 0) and
        # only on the oracle path, so an inference stage using what_latent_to_use='self' is
        # untouched. See docs/flanker_task.md and flanker_near_cong_diagnostic.py (T17, T21).
        #
        # It is probably NOT as critical as the paragraph above implies. Jitter was adopted
        # to fix the near-congruent-worse-than-far-congruent artifact, but that was measured
        # against runs carrying bg_noise_std = 0.1; at bg_noise_std = 0 the artifact is
        # already absent without it. A 2x2 over jitter x p_corr_by_distance[2] (20 seeds,
        # arrow_noise_std 0.9, exports/flanker_random/factorial_corr_jitter) found jitter
        # costs more than it buys: it REVERSES PERI, +0.073 (the human direction) to -0.188,
        # and weakens the incongruent RT distance effect out of significance. It also drives
        # Z HIGHER — mean focus 0.342 -> 0.391 — which is the mechanism to suspect for both.
        # What it does buy is post-error slowing, pes_BI 0.108 (n.s.) -> 0.375. Net across
        # the 11 human signatures: 9 matched without jitter, 8 with.
        #
        # Across the WHOLE ladder jitter never matches more signatures than the baseline at
        # any noise level: it ties at 1.3 and 0.7 (and is cleaner at 0.7, 0 opposite vs 1)
        # and loses at 1.0, 0.9 and 0.4. The only thing it does consistently is raise mean
        # focus, by ~0.05 at every level. The PERI reversal is mid-ladder — at
        # arrow_noise_std 0.4 jitter instead roughly doubles PERI, 0.397 -> 0.936.
        self.oracle_gate_jitter = None

        # ── Latent Optimizer (LU) ──────────────────────────────────────────────
        self.Z_lr = 0.4
        self.Z_optimizer = 'Adam'        # 'Adam', 'AdamW', or 'SGD'
        self.Z_Adam_betas = (0.6, 0.7)   # For faster dynamics (0.9, 0.999)
        self.Z_momentum = 0.0            # only used when Z_optimizer='SGD'
        self.Z_decay = 0.0001                  # L2 regularization weight on Z (weight decay)
        # Which code path applies Z_decay. There are two, and historically BOTH ran:
        #   'grad'      — added to the gradient in RNN_with_latent._apply_chunk_lr_and_decay, and
        #                 the Z optimizer is built with weight_decay=0. Honours chunk_l2_losses,
        #                 and behaves identically under Adam / AdamW / SGD. Prefer this.
        #   'optimizer' — the Z optimizer's own weight_decay, manual term skipped. Coupled for
        #                 Adam, decoupled for AdamW, so the meaning of Z_decay changes with
        #                 Z_optimizer. Offered for completeness.
        #   'both'      — DEPRECATED, and the default only so that existing runs stay
        #                 reproducible. Under Adam both paths add decay*Z to the gradient, so the
        #                 effective decay is 2 * Z_decay and every tuned Z_decay on disk means
        #                 half what it says. New experiments should set 'grad' explicitly.
        self.Z_decay_mode = 'both'
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
        self.W_l2_loss = 0.0  # L2 regularization weight on model weights (weight decay)
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

        # Carry the RNN hidden state across windows (truncated BPTT / RL2-style):
        # forward() starts from the previous window's detached end state instead of
        # re-initializing. Reset per phase by predictive_learning(). Requires
        # non-overlapping windows (stride == seq_len), else predictive_learning raises.
        # Off by default: with it off, Z is the only cross-window memory, which is what
        # the existing flanker results assume.
        self.stateful_hidden = False

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

        Switching to what_latent_to_use='self' only lets Z be inferred if LU steps are enabled.
        A condition that trained with no_of_steps_in_latent_space=0 (an oracle, or a frozen-Z
        control) keeps LU off here unless test_no_of_steps_in_latent_space says otherwise.
        """
        self.what_latent_to_use = 'self'
        self.no_of_steps_in_weight_space = 0
        test_LU = getattr(self, 'test_no_of_steps_in_latent_space', None)
        if test_LU is not None:                      # None → keep the training value
            self.no_of_steps_in_latent_space = test_LU
        self.block_size  = getattr(self, 'test_block_size',   self.block_size)
        self.no_of_blocks = getattr(self, 'test_no_of_blocks', 4)
        self.update_export_path()

    def __setattr__(self, name, value):
        # Backward-compat: old names redirect to current names
        if name == 'l2_loss' or name == 'Z_l2_loss':
            name = 'Z_decay'
        elif name == 'Z_lr':
            name = 'Z_lr'
        super().__setattr__(name, value)

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
        config.Z_lr = 0.5                                  # override any parameter
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
        self.Z_lr = 0.8
        self.Z_Adam_betas = (0.6, 0.7)
        self.Z_decay = 0.0001
        self.Z_optimizer = 'Adam'
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

        self.update_export_path()

        # Tweaking-specific overrides (exploration preset differs in 3 values)
        if experiment_to_run in ('tweaking', 'weight_grads_comp'):
            self.Z_lr = 0.5
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
        self.Z_decay = 0
        self.Z_lr = 0.1
        self.Z_optimizer = 'Adam'
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
        # Which rotation each state-block gets. 'cyclic' walks train_rotations in order (and is
        # seed-independent, so the schedule is fully predictable); 'random_no_repeat' draws one
        # per block, never repeating the previous. With only two rotations the two are the same
        # thing — the flag earns its keep from three rotations up.
        self.rotation_block_order = 'cyclic'  # 'cyclic' or 'random_no_repeat'

        # ── Context-belief output (off by default) ────────────────────────
        # See enable_context_output(): appends dims carrying the rotation, hidden from the model
        # input but supervised at the loss, so the network reports its context belief directly.
        self.context_output_encoding = None   # None, 'circular' or 'one_hot'
        self.context_output_dims     = 0

        # ── Oracle latent ─────────────────────────────────────────────────
        # Context variable = rotation angle (radians), so a full period is the range.
        # 'one_hot' needs Z_dim >= len(train_rotations); 'normalized' needs Z_dim=1;
        # 'circular' needs Z_dim=2. See the oracle block in Config.__init__.
        self.oracle_context_encoding = 'one_hot'
        self.oracle_context_range = (0.0, 2 * np.pi)

        # ── Latent variable ───────────────────────────────────────────────
        self.latent_dims = [2]
        self.Z_lr = 0.2
        self.Z_decay = 0.0001
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
        # Phase 3 is a self-inference test on novel rotations, so LU is on even for conditions
        # that trained with it off (the oracles). Set to 0 for a deliberately frozen-Z control.
        self.test_no_of_steps_in_latent_space = 1

        self.update_export_path()
        self._validate()

    @property
    def oracle_context_values(self):
        """
        Context values the 'one_hot' oracle encoding gives slots to, in slot order.

        The context variable here is the rotation angle, and the dataset emits it in radians.
        Defaults to the trained rotations rather than a fixed list, so it stays correct when a
        run script overrides train_rotations after construction. Assign to pin an explicit table.
        """
        if self._oracle_context_values is not None:
            return self._oracle_context_values
        return np.deg2rad(self.train_rotations)

    @oracle_context_values.setter
    def oracle_context_values(self, values):
        self._oracle_context_values = values

    def enable_context_output(self, encoding='circular', loss_weight=1.0):
        """Make the network report its context belief directly, via the augmented-input trick.

        Same mechanism MeanPredictionConfig uses ([1,0] / [0,1]): the rotation is appended to the
        observation, zeroed out by input_feed_mask before the forward pass so the model cannot
        read it, and re-opened by output_loss_mask so the model is trained to *emit* it.

        Observation layout becomes:
            [ color_onehot(n_colors) | context(C) | attack_x, attack_y ]
        The context sits *before* the coordinates on purpose — every rotating-targets analysis
        reads the attack as [-2:] (rotating_targets_analysis._analyze_adaptation,
        analyze_rotation_sweep, run_2D_predicitve_task.plot_arena_trials), and appending at the
        end would silently point all of them at the context dims instead.

        Encodings:
            'circular' — C=2, target_radius * [cos θ, sin θ]. Same scale as the coordinates, so
                         the two loss terms are balanced, and no wraparound seam. Decode with
                         atan2(out[nc+1], out[nc]).
            'one_hot'  — C=len(train_rotations), one slot per trained rotation. No metric between
                         contexts and no valid target for an unseen angle.

        Call this *after* train_rotations is final; 'one_hot' sizes itself from it.

        loss_weight scales the context term. output_loss_mask is applied as a plain elementwise
        multiply in train_and_infer_functions._mask_loss, so a float entry works as a weight with
        no further machinery; 0.0 keeps the dims present but unsupervised.
        """
        if encoding == 'circular':
            C = 2
        elif encoding == 'one_hot':
            C = len(self.train_rotations)
        else:
            raise ValueError(f"Unknown context_output_encoding '{encoding}'. "
                             "Choose 'circular' or 'one_hot'.")

        base = self.n_colors + 2
        self.context_output_encoding = encoding
        self.context_output_dims     = C
        self.input_size  = base + C
        self.output_size = base + C
        self.input_feed_mask  = [1] * self.n_colors + [0] * C + [1, 1]
        self.output_loss_mask = [0] * self.n_colors + [loss_weight] * C + [1, 1]

        assert len(self.input_feed_mask) == self.input_size, (
            f'input_feed_mask has {len(self.input_feed_mask)} entries for '
            f'input_size={self.input_size}.')
        assert len(self.output_loss_mask) == self.output_size, (
            f'output_loss_mask has {len(self.output_loss_mask)} entries for '
            f'output_size={self.output_size}.')

    @property
    def context_output_slice(self):
        """Column slice of the context dims in an (T, input_size) array, or None if disabled."""
        if not self.context_output_dims:
            return None
        return slice(self.n_colors, self.n_colors + self.context_output_dims)

    def reconfigure_for_prediction(self, experiment_to_run):
        """Switch to test/inference mode: freeze weights, adjust dataset size.

        what_latent_to_use='self' alone does not make Z adapt — LU steps have to be enabled too,
        which the oracle conditions turn off during training. See test_no_of_steps_in_latent_space.
        """
        self.what_latent_to_use = 'self'
        self.no_of_steps_in_weight_space = getattr(self, 'test_no_of_steps_in_weight_space', 0)
        test_LU = getattr(self, 'test_no_of_steps_in_latent_space', None)
        if test_LU is not None:                      # None → keep the training value
            self.no_of_steps_in_latent_space = test_LU
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


class GradualMeanPredictionConfig(MeanPredictionConfig):
    """
    Mean prediction task where the generative mean ramps linearly at morph_rate
    per timestep — starting from 0.5 for the first block, then from the previous
    block's target onward.

    Minimum block size to fully reach the target:
        ceil(|target - start| / morph_rate)
    With default means [0.2, 0.8] and morph_rate=0.05 this is 12 timesteps.
    """

    def __init__(self, experiment_to_run='figure'):
        super().__init__(experiment_to_run)
        self.dataset_name    = 'mean_prediction_gradual'
        self.morph_rate      = 0.05
        self.first_block_mean = None  # set to 0.2 or 0.8 to fix the first block; None = random
        self.update_export_path()


class FlankerTaskConfig(Config):
    """
    Flanker task pretraining configuration.

    5-slot arrow display: [far_left, near_left, center, near_right, far_right].
    Each trial presents a target slot + one companion slot with noisy arrow observations.
    The companion's direction is correlated with the target according to p_corr_by_distance.
    The RNN must infer the target direction; speed pressure is applied via temporally
    decaying loss weights (Option A: temporal discount).

    Within each context block the target slot is fixed; Z learns to encode which slot is
    authoritative. Between blocks the target slot rotates.

    Trial structure matches seq_len = stride = arrows_duration so each BPTT batch is
    exactly one complete trial with no overlap.
    """

    def __init__(self, experiment_to_run='default'):
        super().__init__()
        self.dataset_name        = 'flanker_pretrain'
        self.experiment_to_run   = experiment_to_run

        # Post-gating (apply the multiplicative Z gate after each RNN step, rather than
        # before) is what the pretraining behind every current flanker result actually
        # used. Overrides Config's pre_gating=True/post_gating=False default.
        self.pre_gating  = False
        self.post_gating = True

        # ── Trial structure (user-facing names) ───────────────────────────────
        # 10 timesteps -> 9 usable response steps. It was 5 (4 response steps), which made
        # the interpolated RT density bumpy and left no room for a delayed target onset;
        # both were the blocker named in docs/flanker_task.md "Deferred work". Changing it
        # after construction requires set_arrows_duration(), which keeps seq_len, stride,
        # the block counts and temporal_loss_weights in sync — a bare assignment does not.
        self.arrows_duration          = 10   # timesteps per trial → sets seq_len & stride
        self.trials_per_context_block = 2   # trials with the same target slot per block
        # Total Stage-1 trials. Session length is specified in trials, not timesteps, so
        # it stays correct if arrows_duration ever changes; call set_n_pretrain_trials()
        # to change it after construction, which keeps the derived block counts in sync.
        self.n_pretrain_trials        = 4000
        self.n_training_contexts      = self.n_pretrain_trials // self.trials_per_context_block

        # ── Task dimensions ───────────────────────────────────────────────────
        self.input_size  = 6   # 5 noisy slot observations + 1 hidden target direction
        self.output_size = 6
        self.hidden_size = 64
        self.seq_len     = self.arrows_duration   # one batch = one trial
        self.stride      = self.arrows_duration   # non-overlapping trials

        # ── Input/output masking ──────────────────────────────────────────────
        self.input_feed_mask      = [1, 1, 1, 1, 1, 0]   # hide dim 5 (true direction)
        self.output_loss_mask     = [0, 0, 0, 0, 0, 1]   # loss only on dim 5
        self.predict_first_frame  = True   # t=0 uses zero frame; response window from t=1

        # ── Speed pressure: temporal discount loss (Option A) ─────────────────
        # temporal_decay_factor=0 → equal weights; larger → concentrate at t=1.
        #
        # lambda is a PER-TIMESTEP decay, so it must be rescaled whenever arrows_duration
        # moves or the tail of a longer window collapses to zero. Parameterise by the
        # end-of-window weight: exp(-lambda * (n_response_steps - 1)). The 5-step trial ran
        # lambda=0.3 over 4 response steps, i.e. an end weight of exp(-0.9) = 0.407; 0.112
        # preserves that same 0.407 over the 9 response steps of the 10-step trial.
        self.response_start_timestep = 1
        self.temporal_decay_factor   = 0.112
        self._set_temporal_weights()    # populates self.temporal_loss_weights

        # ── Target onset delay ("flankers first") ─────────────────────────────
        # Timesteps by which the TARGET arrow's onset is delayed. The flankers are present
        # from frame 0 as usual; for the first `target_delay` frames the target slot carries
        # background noise only. 0 (the default) is simultaneous onset, i.e. the task every
        # result before this was measured on.
        #
        # It is deliberately inert everywhere except FlankerRandomTrialsDataset: the human
        # onset asynchrony is a TEST-stage probe of a read-out trained on simultaneous
        # onset, so FlankerTaskDataset asserts it is 0.
        #
        # Nothing derived depends on it, and that is the design rather than an oversight.
        # response_start_timestep stays 1 and temporal_loss_weights are untouched, so the
        # model is asked for the target direction from t=1 whether or not the target has
        # arrived, and RT is measured from trial start. Zeroing the loss over the pre-target
        # window, or re-referencing RT to onset, would normalise away the effect being
        # measured — the question is precisely whether the response is DELAYED when the
        # target is late, and whether the flankers alone are enough to answer early.
        self.target_delay            = 0

        # ── Flanker stimulus parameters ───────────────────────────────────────
        self.n_slots            = 5
        # NOTE IMPORTANT. Correlation cannot be allowed to dip below 0.5 at any distance!!

        # p_corr_by_distance[d] = probability companion matches target at distance d
        # self.p_corr_by_distance = [1.0, 0.65, 0.55, 0.25, 0.1]
        # this steeper corr significantly improved model's capture int the flanker_sweep setup.
        self.p_corr_by_distance = [1.0, 0.75, 0.52, 0.51, 0.5]
        
        # This is a tight balance. 
        # Increasing the corr increases congruent trials n and lowers incongruent. 
        # But this makes the cong trials easier, but the model never learns to respond to context_ids which has the target slot
        # But lowering the corr increases incongruent trials to the point that the model 
        # learns to ignore the companion and just respond to the target slot, so no spatial structure is learned.
        
        # Power law: fast drop at d=1, then slower tail. p(d) = 1 / (1+d)^α
        # α=1 → [1.0, 0.50, 0.33, 0.25, 0.20]  (gentle, 1/n)
        # α=1.5 → [1.0, 0.35, 0.19, 0.13, 0.09]  (sharper initial, slower tail)
        # α=2 → [1.0, 0.25, 0.11, 0.06, 0.04]  (very sharp at d=1)
        # self.correlation_lambda = lambda alpha: [1 / (1 + d) ** alpha for d in range(5)]
        # self.p_corr_by_distance = self.correlation_lambda(.5)
        # if self.p_corr_by_distance[2] <= 0.5:
        #     print(f"Warning: p_corr_by_distance[2] = {self.p_corr_by_distance[2]:.3f} is not > 0.5. Consider increasing correlation_lambda to strengthen the congruent/incongruent effect.")



        self.arrow_noise_std    = .9
        # run_flanker.py and flanker_sweep_config both override this to 0, and that matters:
        # at 0.1 the slots holding no arrow still carry noise the model reads as evidence,
        # and setting it to 0 removes the near-congruent-worse-than-far-congruent artifact
        # on its own, without oracle gate jitter (20 seeds, arrow_noise_std 0.9).
        self.bg_noise_std       = 0.1
        self.signal_strength    = 1.0

        # ── Latent / oracle mode ──────────────────────────────────────────────
        # 'context_ids': oracle — target slot integer from hlcids fed as one-hot Z (no LU needed)
        # 'self'  : model learns Z through LU (blind to slot identity at training time)
        self.what_latent_to_use = 'context_ids'
        self.latent_dims        = [5]   # Z_dim=5 matches n_slots; softmax → one-hot over slots
        self.latent_chunks      = 1
        self.exponential_increase_steepness = [2]
        self.Z_decay_mode = 'grad'
        self.Z_optimizer = 'SGD'
        if self.Z_optimizer in ('sgd', 'SGD'):
            self.Z_lr               = 300.
            self.Z_decay            = 2.6e-4
        else:
            self.Z_lr               = 0.3
            self.Z_decay            = 0.0001
        # ── Derived NeuraGEM internal params (from user-facing names above) ───
        self.block_size             = self.trials_per_context_block * self.arrows_duration
        self.no_of_blocks           = self.n_training_contexts
        self.blocked_phase_length   = self.no_of_blocks * self.block_size
        self.block_duration_distribution = 'fixed_block_size'

        self.update_export_path()
        self._validate()

    def set_n_pretrain_trials(self, n_pretrain_trials):
        """Set Stage-1 session length in trials and keep the derived block counts consistent."""
        self.n_pretrain_trials    = int(n_pretrain_trials)
        self.n_training_contexts  = self.n_pretrain_trials // self.trials_per_context_block
        self.no_of_blocks         = self.n_training_contexts
        self.blocked_phase_length = self.no_of_blocks * self.block_size
        return self

    def set_arrows_duration(self, arrows_duration):
        """Set the trial length in timesteps and keep every derived field consistent.

        seq_len, stride and block_size are all computed from arrows_duration at
        construction, and temporal_loss_weights is built to that length, so assigning the
        attribute on its own leaves the config internally inconsistent — the model would
        run 10-step trials against a 5-long weight vector and a stride of 5. That is not a
        hypothetical: flanker_sweep applies its overrides with a plain setattr, so this is
        the entry point a sweep must use to vary it.
        """
        self.arrows_duration      = int(arrows_duration)
        self.seq_len              = self.arrows_duration
        self.stride               = self.arrows_duration
        self.block_size           = self.trials_per_context_block * self.arrows_duration
        self.blocked_phase_length = self.no_of_blocks * self.block_size
        self._set_temporal_weights()
        return self

    def _validate(self):
        super()._validate()
        # Needs at least one response step, and the target has to reach an output. With
        # predict_first_frame=True the model at timestep t has seen frames 0..t-1, so a
        # target appearing at frame `target_delay` first influences the output at
        # t = target_delay + 1; a delay of arrows_duration - 1 would never be seen at all.
        assert 0 <= self.target_delay <= self.arrows_duration - 2, (
            f'target_delay={self.target_delay} must be in [0, arrows_duration - 2] '
            f'(arrows_duration={self.arrows_duration}), or the target never reaches an output.'
        )
        assert len(self.temporal_loss_weights) == self.arrows_duration == self.seq_len, (
            f'temporal_loss_weights has {len(self.temporal_loss_weights)} entries for '
            f'arrows_duration={self.arrows_duration}, seq_len={self.seq_len}. '
            f'Use set_arrows_duration() rather than assigning arrows_duration directly.'
        )

    def _set_temporal_weights(self):
        """Compute temporal_loss_weights from response_start_timestep and temporal_decay_factor."""
        weights = [0.0] * self.response_start_timestep
        n_resp  = self.arrows_duration - self.response_start_timestep
        weights += [np.exp(-self.temporal_decay_factor * i) for i in range(n_resp)]
        self.temporal_loss_weights = weights   # length == arrows_duration == seq_len


class FlankerTaskStage2Config(FlankerTaskConfig):
    """
    Stage 2: frozen weights, self-learned Z on full-flanker congruency blocks.
    Inherits all stimulus params from FlankerTaskConfig; overrides dataset,
    latent mode, weight-update count, and block structure.
    """

    def __init__(self, experiment_to_run='default'):
        super().__init__(experiment_to_run)
        self.dataset_name                = 'flanker_stage2'
        self.what_latent_to_use          = 'self'
        self.no_of_steps_in_weight_space = 0         # weights frozen throughout
        self.trials_per_context_block    = 10        # shorter blocks
        self.block_size                  = self.trials_per_context_block * self.arrows_duration
        self.n_training_contexts         = 4
        self.no_of_blocks                = self.n_training_contexts
        self.blocked_phase_length        = self.no_of_blocks * self.block_size

        self.update_export_path()
        self._validate()


class FlankerTaskStage3Config(FlankerTaskStage2Config):
    """
    Stage 3: frozen weights, self-learned Z.
    4 block types cross distance × congruency (near/far × cong/incong).
    """

    def __init__(self, experiment_to_run='default'):
        super().__init__(experiment_to_run)
        self.dataset_name        = 'flanker_stage3'
        self.n_training_contexts = 16   # 4 full cycles of the 4 block types
        self.no_of_blocks        = self.n_training_contexts
        self.blocked_phase_length = self.no_of_blocks * self.block_size

        self.update_export_path()
        self._validate()


class FlankerRandomTrialsConfig(FlankerTaskConfig):
    """
    Test stage: frozen weights, self-inferred Z, fully randomized trials.

    This is the only stage that gets analyzed. Trials are drawn i.i.d. from the four
    near/far x congruent/incongruent types, matching the randomly interleaved design
    humans perform. Blocked presentation confounds condition with time-in-block and
    with adaptation-to-switch, so every condition effect is instead obtained by
    masking this single random session.

    block_size = arrows_duration, so each DataLoader batch is exactly one trial and
    Z receives exactly one LU update per trial.
    """

    def __init__(self, experiment_to_run='default'):
        super().__init__(experiment_to_run)
        self.dataset_name = 'flanker_random'

        # Weights frozen; Z is the only thing that adapts.
        self.what_latent_to_use          = 'self'
        self.no_of_steps_in_weight_space = 0

        self.n_trials             = 5000
        self.block_size           = self.arrows_duration   # 1 trial per batch
        self.n_training_contexts  = self.n_trials
        self.no_of_blocks         = self.n_trials
        self.blocked_phase_length = self.n_trials * self.arrows_duration

        # Trial-type proportions. 0.5 congruent matches the human task; the sweep
        # varies it to test the list-wide proportion-congruent prediction.
        self.p_congruent = 0.5
        self.p_near      = 0.5
        # self.Z_optimizer = 'SGD' # HAS no effect, model already built by the time this config comes to play

        self.update_export_path()
        self._validate()

    def set_n_trials(self, n_trials):
        """Set session length in trials and keep the derived block counts consistent."""
        self.n_trials             = int(n_trials)
        self.n_training_contexts  = self.n_trials
        self.no_of_blocks         = self.n_trials
        self.blocked_phase_length = self.n_trials * self.arrows_duration
        return self

    def set_arrows_duration(self, arrows_duration):
        """Trial length, with this stage's own block derivation.

        One DataLoader batch is exactly one trial here, so block_size is arrows_duration
        rather than trials_per_context_block * arrows_duration, and the session length is
        counted in trials. The base implementation would silently give this stage two
        trials per batch and therefore one Z update per two trials.
        """
        self.arrows_duration      = int(arrows_duration)
        self.seq_len              = self.arrows_duration
        self.stride               = self.arrows_duration
        self.block_size           = self.arrows_duration
        self.blocked_phase_length = self.n_trials * self.arrows_duration
        self._set_temporal_weights()
        return self


class FlankerTaskStage4Config(FlankerRandomTrialsConfig):
    """Deprecated name for FlankerRandomTrialsConfig. Kept so the archived
    blocked-stage script and older notebooks still import."""


# Backwards-compatibility aliases
seq_learnConfig = SeqLearnConfig
CSWConfig = SeqLearnConfig
