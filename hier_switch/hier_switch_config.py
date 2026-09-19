"""
hier_switch_config.py — hierarchical cue→rule reversal task (Lam et al., Nature 2024).

Tree shrews hear a 16-pulse cue. 9 pulses are informative (high-pass HP or low-pass LP)
and 7 are white noise (WN). The dominant pulse type is the *cue*, and the cue says which
modality to attend — vision or audition. A visual and an auditory target then appear on
opposite sides, and the animal picks the side of the attended one. The cue→rule mapping
reverses covertly every 30–60 trials.

Two uncertainties live in this task:
  cueing uncertainty  conflict = non-dominant / dominant pulse count (7:2 → 0.29, 6:3 → 0.5)
  rule uncertainty    has the context reversed?

One trial is one batch. Hidden state resets every trial (stateful_hidden=False), so Z is the
only thing carried across trials.

    frames 0 … n_pulses-1              cue pulses, one per frame
    frames n_pulses … target_onset-1   delay
    frames target_onset … trial_len-1  targets on; response window

The response is correct_side = vis_side × cue_sign × context_sign. It is a three-way parity,
so a reversal flips the correct answer on every trial.

Channel layout (augmented-input trick, as in flanker / mean prediction). The label dims ride
through logger.inputs unmasked, so every analysis reads them back without extra logging:

    idx  0   1   2   3    4    5    6         7        8
         HP  LP  WN  vis  aud  cue  conflict  context  correct
    in   1   1   1   1    1    0    0         0        0     ← input_feed_mask
    out  0   0   0   0    0    0    0         0        1     ← output_loss_mask

The target is the correct side. In a two-choice task that carries the same information as
binary reward: an error says the other side was right.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from configs import Config
import hier_switch_dataset  # noqa: F401  (registers 'hier_switch' in DATASET_REGISTRY)


CHANNELS = ('HP', 'LP', 'WN', 'vis', 'aud', 'cue', 'conflict', 'context', 'correct')


class HierSwitchConfig(Config):
    """Config for the hierarchical cue→rule reversal task.

    Session lengths are in trials. Change them with set_n_trials(), and change the trial
    timing with set_timing(). Both keep the derived fields (seq_len, stride, block_size,
    blocked_phase_length, temporal_loss_weights) in sync; a bare assignment does not.
    """

    def __init__(self, experiment_to_run='default'):
        super().__init__()
        self.dataset_name      = 'hier_switch'
        self.experiment_to_run = experiment_to_run
        # Absolute, so output lands in the repo's exports/ whatever the working directory.
        self.export_folder     = os.path.join(_ROOT, 'exports') + os.sep

        # ── Cue ───────────────────────────────────────────────────────────────
        self.n_pulses        = 16
        self.n_informative   = 9
        # (dominant, non-dominant) counts over the informative pulses. Conflict is
        # non-dominant / dominant: 0, .125, .286, .5, .8. The paper's low / high levels are
        # 7:2 (0.29) and 6:3 (0.5).
        self.conflict_counts = [(9, 0), (8, 1), (7, 2), (6, 3), (5, 4)]
        self.p_conflict      = None      # None = uniform over conflict_counts
        # Gaussian noise on the three pulse channels of every pulse frame. It is what makes
        # conflict an actual sensory uncertainty: with noiseless pulses, counting recovers
        # the cue exactly at every conflict level.
        self.pulse_noise_std = 0.5

        # ── Timing (derived by set_timing) ────────────────────────────────────
        self.delay_steps    = 2
        self.response_steps = 7
        # Speed pressure over the response window, per timestep, as in the flanker task.
        # With predict_first_frame=True the first output that has seen the targets is at
        # target_onset + 1, so a 7-frame target period gives 6 response outputs.
        # 0.178 = -ln(0.41) / 5 puts the end-of-window weight at 0.41, the flanker setting.
        self.temporal_decay_factor = 0.178

        # ── Blocks ────────────────────────────────────────────────────────────
        self.block_len_range = (30, 60)  # trials per context block, uniform, inclusive
        self.first_context   = None      # None = random; 0 or 1 to pin it
        # Training-stream curriculum over block length: [(until_trial, (lo, hi)), ...] sets
        # the range for blocks that start before until_trial; block_len_range after that and
        # in the test stream always. None = paper blocks from the first trial.
        #
        # Why it exists: with 30-60-trial blocks from scratch, nothing is learned — NG, RNN
        # alike sit at chance (tune_v1). The answer's dependence on vis x cue flips sign with
        # the context, so averaged over blocks the weights see no gradient for that product,
        # and a block is too short to learn it inside one (Oracle Z needs ~2000 trials to
        # learn the task even when handed the context). A long first block lets the weights
        # build the machinery; no context label is used at any point.
        #
        # Default: one 600-trial block, then 200-trial blocks. tune_v3 ran 3000 then 1000s;
        # the RNN re-learned each 1000-trial block within its first ~10%, so blocks were cut
        # to 20% — long enough to relearn in, short enough that Z has reversals to track.
        # None = paper blocks from the first trial.
        #
        # Default now (tune_v12): no long first block. The passive phase below teaches the
        # basics of the task, and active training runs on 200-trial blocks from its first trial.
        self.train_block_schedule = [(10**9, (200, 200))]

        # ── Session lengths (trials) ──────────────────────────────────────────
        self.n_train_trials = 5000     # active phase (weights + Z)
        self.n_test_trials  = 1000
        # Passive phase (train_model Phase 1): weights learn the basics of the task while Z is
        # held at Z_init. Early errors carry no context information, and letting Z follow them
        # drags it to the uniform gate before the weights have anything for it to gate.
        # passive_n_blocks equal blocks of alternating context, so both mappings are seen.
        # 2 x 1500 was too short to learn in for most seeds (tune_v12); 2 x 2000 taught the
        # task to 7/10, and 6/10 went on to discover the contexts (tune_v13). These defaults
        # (2 x 2000 passive, 5000 active) are that verified configuration. A longer passive
        # phase (2 x 3000) with 4000 active trials is an untested alternative.
        # 0 trials = off. Set with set_n_passive_trials().
        self.n_passive_trials = 4000
        self.passive_n_blocks = 2
        # Picks the RNG stream the dataset draws from, keyed on (env_seed, data_stream).
        # The test phase uses its own stream, so it does not replay the training prefix.
        self.data_stream = 0

        # ── Dimensions ────────────────────────────────────────────────────────
        self.channels         = {name: i for i, name in enumerate(CHANNELS)}
        self.input_size       = len(CHANNELS)
        self.output_size      = len(CHANNELS)
        self.input_feed_mask  = [1, 1, 1, 1, 1, 0, 0, 0, 0]
        self.output_loss_mask = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.hidden_size      = 64
        self.predict_first_frame = True

        # ── Latent ────────────────────────────────────────────────────────────
        # Softmax over the 2 dims: one degree of freedom, *which* gate, and no overall gain.
        # Z_init = 0 is the uniform gate. The model is built with Z set to Z_init
        # (hier_switch_train.build_model).
        #
        # Why softmax (tune_v3): with a sigmoid the task is learned and re-learned after each
        # reversal, but by the weights. Z's two dims rise together and act as a gain, because
        # after a reversal "turn the output down" always lowers squared error while "switch
        # the mapping" only pays once the weights implement two mappings. A 2-way softmax has
        # no gain direction to escape into.
        #
        # 'none' was tried first and fails (tune_v2, tune_v3). While the output cannot yet
        # predict the sign, the cheapest way for the latent update to cut squared error is to
        # shrink the output, i.e. shrink the gate. With a raw Z that runs to Z ≈ 0, a zero
        # gate, which silences the network and blocks the weights' gradient. It sticks
        # there even after a passive phase has taught the task: after the first reversal the
        # update shrinks the gate again rather than flipping it. Use 'none' with Z_init=0.5.
        self.latent_dims        = [2]
        self.latent_chunks      = 1
        self.latent_activation  = 'softmax'
        
        self.use_mul_gating = True
        self.pre_gating = True
        self.post_gating = not self.pre_gating
        
        # A sharp softmax. At temp 1 the oracle one-hot is a [0.73, 0.27] gate, and even the
        # Oracle cannot learn to use it (tune_v4: its learning curve = the RNN's). At 0.5 it
        # is [0.88, 0.12] and the Oracle learns in ~2500 trials (tune_v5).
        self.softmax_temp       = 0.5
        self.Z_init             = 0.0      # the uniform gate
        # Uniform weight over the trial's timesteps when pooling dL/dZ. The raw gradient is
        # already concentrated where the loss is (t 19-24), and steepness 0 vs 2 made no
        # difference to Oracle inference (tune_v7).
        self.exponential_increase_steepness   = [0]
        self.exponential_increase_multipliers = [1]
        # SGD, not Adam. Adam divides the gradient by its own running magnitude, and that
        # magnitude is what should carry how much an error says about the context.
        self.Z_optimizer    = 'SGD'
        # Plain L2 weight decay on raw Z, i.e. toward 0, the uniform softmax gate. Under SGD
        # 'grad' is identical to the optimizer's own weight_decay ('optimizer' mode): each LU
        # step is Z <- Z - Z_lr * (grad + Z_decay * Z).
        self.Z_decay_mode   = 'grad'
        self.Z_momentum     = 0.0
        # Bracketed on the Oracle: trained with the true context as Z, then tested with the
        # oracle removed and Z optimised from uniform on the paper's 30-60 blocks.
        #   tune_v5 (no weight decay): temp 0.5, Z_lr 1000 best — 0.91 steady state, but ~10
        #   trials to switch; >= 3000 thrashes. The usable range scales with 1/temp.
        #   tune_v7 (weight decay): decay is what sets the switch speed. Its strength per step
        #   is Z_lr * Z_decay. Z_lr 3000 with Z_decay 3.3e-5 is best: 0.84 overall, 0.89 steady,
        #   crosses chance 3 trials after a reversal (no decay: 0.73 / 0.87 / 10 trials).
        #   Z_lr * Z_decay = 0.3 switches in ~1 trial but cannot hold the context (0.85 steady).
        # NG discovery (tune_v9-v12) wants Z_lr * Z_decay ~ 0.03: less and Z never flips with the
        # context, more and Z is held at the middle before the weights specialise. Z_lr 1e4
        # splits Z earliest (tune_v11). Discovering runs: test 0.83-0.86 overall, 0.88-0.91
        # steady, chance crossed 2-3 trials after a reversal.
        self.Z_lr           = 1e4
        self.Z_decay        = 3e-6
        self.pass_previous_latent = True
        self.loss_reduction_LU = 'mean'
        self.loss_reduction_WU = 'mean'
        # Leave at 1e-3. Neither direction helped: 5e-4 changed nothing (tune_v11), and 3e-3
        # stopped every seed from learning the task at all (tune_v14).
        self.WU_lr = 0.001

        # Oracle (what_latent_to_use='context_ids'): the context id 0/1 as a one-hot Z.
        self.oracle_context_encoding = 'one_hot'
        self.oracle_context_values   = [0.0, 1.0]

        # ── Test phase (reconfigure_for_prediction) ───────────────────────────
        self.test_no_of_steps_in_weight_space = 0      # 0 = weights frozen
        self.test_no_of_steps_in_latent_space = None   # None = keep the training value

        self.block_duration_distribution = 'fixed'  # one base-class "block" = one trial
        self.set_timing()
        self.set_n_trials(self.n_train_trials)
        self.set_n_passive_trials(self.n_passive_trials)
        self.update_export_path()
        self._validate()

    # ── Derived fields ────────────────────────────────────────────────────────

    def set_timing(self, n_pulses=None, delay_steps=None, response_steps=None):
        """Set the trial timing and keep every field derived from it consistent."""
        if n_pulses is not None:
            self.n_pulses = int(n_pulses)
        if delay_steps is not None:
            self.delay_steps = int(delay_steps)
        if response_steps is not None:
            self.response_steps = int(response_steps)
        self.target_onset = self.n_pulses + self.delay_steps
        self.trial_len    = self.target_onset + self.response_steps
        self.seq_len      = self.trial_len
        self.stride       = self.trial_len
        self.block_size   = self.trial_len
        # First output that has seen the targets (predict_first_frame: output t sees frames < t).
        self.response_start_timestep = self.target_onset + 1
        self._set_temporal_weights()
        if hasattr(self, 'no_of_blocks'):
            self.blocked_phase_length = self.no_of_blocks * self.trial_len
        if hasattr(self, 'n_passive_trials'):
            self.passive_phase_length = self.n_passive_trials * self.trial_len
        return self

    def set_n_trials(self, n_trials):
        """Set the training session length in trials."""
        self.n_train_trials       = int(n_trials)
        self.no_of_blocks         = self.n_train_trials
        self.blocked_phase_length = self.n_train_trials * self.trial_len
        return self

    def set_n_passive_trials(self, n_trials):
        """Length of the passive (weights-only) phase in trials; 0 turns it off."""
        self.n_passive_trials = int(n_trials)
        self.add_passive_learning_phase = self.n_passive_trials > 0
        self.passive_phase_length = self.n_passive_trials * self.trial_len
        return self

    def _set_temporal_weights(self):
        """Zero before the targets are seen, then exp(-λ·i) over the response window."""
        n_resp = self.trial_len - self.response_start_timestep
        self.temporal_loss_weights = (
            [0.0] * self.response_start_timestep
            + [float(np.exp(-self.temporal_decay_factor * i)) for i in range(n_resp)]
        )

    def ch(self, name):
        """Index of a named channel in the input / output vector."""
        return self.channels[name]

    # ── Phases ────────────────────────────────────────────────────────────────

    def reconfigure_for_prediction(self, experiment_to_run):
        """Test phase: Z self-inferred, weights frozen unless test_no_of_steps_in_weight_space
        says otherwise, n_test_trials long, drawn from its own RNG stream."""
        self.what_latent_to_use = 'self'
        self.no_of_steps_in_weight_space = int(getattr(self, 'test_no_of_steps_in_weight_space', 0))
        test_LU = getattr(self, 'test_no_of_steps_in_latent_space', None)
        if test_LU is not None:
            self.no_of_steps_in_latent_space = test_LU
        self.no_of_blocks = int(self.n_test_trials)
        self.data_stream  = 1
        self.update_export_path()

    def _validate(self):
        super()._validate()
        assert len(self.temporal_loss_weights) == self.trial_len == self.seq_len, (
            f'temporal_loss_weights has {len(self.temporal_loss_weights)} entries for '
            f'trial_len={self.trial_len}. Use set_timing() rather than assigning directly.')
        assert self.predict_first_frame, (
            'predict_first_frame=False is not supported: the target is constant within a '
            'trial, and the shorter output sequence would misalign the per-trial reshape.')
        for dom, non in self.conflict_counts:
            assert dom + non == self.n_informative and dom > non, (
                f'conflict count ({dom}, {non}) must sum to n_informative='
                f'{self.n_informative} with a strict majority.')
        assert self.n_informative <= self.n_pulses
        assert len(self.input_feed_mask) == len(self.output_loss_mask) == self.input_size
