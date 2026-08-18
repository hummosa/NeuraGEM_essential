"""
datasets.py — Dataset classes for NeuraGEM tasks.

Every dataset returns 3-tuples from its DataLoader:
    (data, context_ids, hlcids)
    data:   (batch, seq_len, input_size)  — observations
    context_ids: (batch, seq_len, 1)           — low-level context IDs
    hlcids: (batch, seq_len, 1)           — high-level context IDs

Adding a new dataset
--------------------
1. Subclass BaseTaskDataset and implement generate_sequences().
2. Register it: DATASET_REGISTRY['my_task'] = MyDataset
3. Set config.dataset_name = 'my_task' and config.input_size to match.

Example (2D Gaussian with two contexts):

    class My2DDataset(BaseTaskDataset):
        def generate_sequences(self):
            data, context_ids, hlcids = [], [], []
            for block_idx, block_size in enumerate(self.block_sizes):
                mean = [0.2, 0.8] if block_idx % 2 == 0 else [0.8, 0.2]
                block = self.rng.normal(mean, self.config.default_std, (block_size, 2))
                data.extend(block)
                context_ids.extend([float(block_idx % 2)] * block_size)
                hlcids.extend([0.0] * block_size)
            return data, context_ids, hlcids

    DATASET_REGISTRY['my_2d'] = My2DDataset

Then in your script:
    config = ContextualSwitchingTaskConfig()
    config.input_size = 2
    config.output_size = 2
    config.dataset_name = 'my_2d'
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_blobs
from functions_and_utils import *
from configs import *


# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry and factory
# ──────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {}  # populated at the bottom of this file


def create_datasets_and_loaders(config, pattern=None):
    """
    Build train and test datasets and wrap them in DataLoaders.

    Returns: (dataset, dataset_test, train_loader, test_loader)

    Each loader yields (data, context_ids, hlcids) with shapes
    (batch, seq_len, input_size), (batch, seq_len, 1), (batch, seq_len, 1).
    """
    cls = DATASET_REGISTRY.get(config.dataset_name)
    if cls is None:
        raise ValueError(
            f"Unknown dataset '{config.dataset_name}'. "
            f"Register it in DATASET_REGISTRY or choose from: {list(DATASET_REGISTRY)}"
        )

    if pattern is None:
        dataset = cls(config)
        test_cls_name = getattr(config, 'test_dataset_name', config.dataset_name)
        test_cls = DATASET_REGISTRY.get(test_cls_name, cls)
        dataset_test = test_cls(config)
    else:
        dataset = cls(config)
        dataset_test = TaskDataset_tests(config.no_of_blocks, config, pattern)

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    return dataset, dataset_test, train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseTaskDataset(Dataset, ABC):
    """
    Template for NeuraGEM datasets.

    Subclasses implement generate_sequences() which returns three flat lists:
        data_sequence   — list of observations; each element has shape (input_size,) or is scalar
        llcid_sequence  — list of scalar low-level context IDs (one per timestep)
        hlcid_sequence  — list of scalar high-level context IDs (one per timestep)

    __getitem__ windows these lists into (seq_len, input_size) tensors automatically,
    using config.seq_len, config.stride, and config.input_size.
    """

    def __init__(self, config, no_of_blocks=None):
        self.config = config
        self.num_blocks = no_of_blocks if no_of_blocks is not None else config.no_of_blocks
        self.block_size = config.block_size
        self.rng = np.random.default_rng(config.env_seed)

        self.block_sizes = self._generate_block_sizes()
        self.data_sequence, self.llcid_sequence, self.hlcid_sequence = self.generate_sequences()

    @abstractmethod
    def generate_sequences(self):
        """
        Return (data_sequence, llcid_sequence, hlcid_sequence) as flat lists.
        len(data_sequence) == len(llcid_sequence) == len(hlcid_sequence) == total timesteps.
        Each element of data_sequence should be a scalar or 1D array of length config.input_size.
        """

    def _generate_block_sizes(self):
        dist = self.config.block_duration_distribution
        if dist in ('fixed_block_size', 'fixed'):
            return [self.block_size] * self.num_blocks
        if dist == 'geometric':
            min_sz = max(1, int(self.block_size / 1.5))
            max_sz = int(self.block_size * 2)
            sizes = []
            for _ in range(self.num_blocks):
                s = self.rng.geometric(2 / self.block_size) + min_sz
                sizes.append(min(s, max_sz))
            return sizes
        raise ValueError(f"Unknown block_duration_distribution '{dist}'. "
                         "Choose 'fixed_block_size' or 'geometric'.")

    def __len__(self):
        return (len(self.data_sequence) - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, index):
        start = index * self.config.stride
        end = start + self.config.seq_len

        data = self.data_sequence[start:end]
        context_ids = self.llcid_sequence[start:end]
        hlcids = self.hlcid_sequence[start:end]

        # Convert to tensors with explicit shape (seq_len, input_size)
        data_t = torch.tensor(np.array(data), dtype=torch.float32).reshape(self.config.seq_len, self.config.input_size)
        cid_size = getattr(self.config, 'context_id_size', 1)
        llcid_t = torch.tensor(np.array(context_ids, dtype=np.float32)).reshape(self.config.seq_len, cid_size)
        hlcid_t = torch.tensor(np.array(hlcids, dtype=np.float32)).reshape(self.config.seq_len, 1)

        return data_t, llcid_t, hlcid_t


# ──────────────────────────────────────────────────────────────────────────────
# Contextual Switching Task (1D)
# ──────────────────────────────────────────────────────────────────────────────

class TaskDataset(BaseTaskDataset):
    """
    1D contextual switching task.

    Observations are scalars drawn from one of two Gaussians parameterized by
    config.training_data_means and config.default_std. Context switches in blocks
    with no explicit label — the model must infer context from prediction errors.
    """

    def __init__(self, config, no_of_blocks=None):
        super().__init__(config, no_of_blocks)

    def generate_sequences(self):
        latent_values = self.config.training_data_means
        data_rng = np.random.default_rng(self.config.env_seed)

        # Choose starting context
        if self.config.start_always_on_the_same_block:
            current_latent = min(latent_values)
        else:
            current_latent = self.rng.choice(latent_values)

        llcid_seq, hlcid_seq, data_seq = [], [], []
        high_level_latent = self.rng.choice([1, 2])
        noise_state = 0.0  # AR(1) state; carried across blocks for continuity

        for i, block_size in enumerate(self.block_sizes):
            # Switch context each block
            options = [v for v in latent_values if v != current_latent]
            if options:
                current_latent = self.rng.choice(options)

            # High-level latent changes every N blocks
            if i % self.config.high_level_latent_change_interval_in_blocks == 0:
                high_level_latent = self.rng.choice([1, 2])

            # Sample observations for this block
            data_rng_for_block = data_rng
            if self.config.use_high_task_structure:
                data_rng_for_block = np.random.default_rng(int(high_level_latent) + self.config.env_seed)

            if getattr(self.config, 'correlated_noise', False):
                alpha = np.exp(-1.0 / self.config.noise_correlation_tau)
                drive_std = self.config.default_std * np.sqrt(1 - alpha ** 2)
                for _ in range(block_size):
                    noise_state = alpha * noise_state + drive_std * data_rng_for_block.standard_normal()
                    data_seq.append(current_latent + noise_state)
            else:
                block_data = data_rng_for_block.normal(current_latent, self.config.default_std, block_size)
                data_seq.extend(block_data)
            llcid_seq.extend([current_latent] * block_size)
            hlcid_seq.extend([high_level_latent] * block_size)

        return data_seq, llcid_seq, hlcid_seq

    def truncate_data_sequence(self, end=None):
        """Shorten sequences in place (used for COIN-style experiments)."""
        if end is None:
            end = self.block_size + getattr(self.config, 'length_of_opposite_block_sequence', 4)
        self.data_sequence = self.data_sequence[:end]
        self.llcid_sequence = self.llcid_sequence[:end]
        self.hlcid_sequence = self.hlcid_sequence[:end]


# ──────────────────────────────────────────────────────────────────────────────
# Contextual Switching Task (2D)
# ──────────────────────────────────────────────────────────────────────────────

class TaskDataset2D(BaseTaskDataset):
    """
    2D contextual switching task.

    Observations are 2D vectors drawn from one of three 2D Gaussians built from
    permutations of config.training_data_means. Optionally uses sklearn make_blobs
    data if config.use_EM_demo_data is set.
    """

    def __init__(self, config, no_of_blocks=None):
        base_means = config.training_data_means
        self._latent_means_2d = [
            np.array([base_means[0], base_means[1]]),
            np.array([base_means[1], base_means[0]]),
            np.array([base_means[1], base_means[1]]),
        ]
        self._use_em = getattr(config, 'use_EM_demo_data', False)
        if self._use_em:
            self._em_data = self._build_em_data(seed=42)
        super().__init__(config, no_of_blocks)

    def _build_em_data(self, seed):
        centers = np.column_stack(([-0.4, 0.5, 0.0], [0.0, 0.5, 0.8]))
        stds = np.column_stack(([0.15, 0.2, 0.1], [0.15, 0.2, 0.1]))
        X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=stds, random_state=seed)
        return {cl: X[y == cl] for cl in np.unique(y)}

    def generate_sequences(self):
        data_rng = np.random.default_rng(self.config.env_seed)
        current_mean = self._latent_means_2d[0] if self.config.start_always_on_the_same_block \
            else self.rng.choice(self._latent_means_2d)
        high_level_latent = float(self.rng.integers(1, 3))

        data_seq, llcid_seq, hlcid_seq = [], [], []

        for i, block_size in enumerate(self.block_sizes):
            options = [m for m in self._latent_means_2d if not np.array_equal(m, current_mean)]
            if options:
                current_mean = options[self.rng.integers(len(options))]

            if i % self.config.high_level_latent_change_interval_in_blocks == 0:
                high_level_latent = float(self.rng.integers(1, 3))

            if self._use_em:
                cluster_key = self.rng.integers(len(self._em_data))
                cluster_data = self._em_data[cluster_key]
                idx = self.rng.integers(0, len(cluster_data), size=block_size)
                block_data = cluster_data[idx]
                llcid_val = float(cluster_key)
            else:
                block_data = data_rng.normal(current_mean, self.config.default_std, (block_size, 2))
                llcid_val = float(np.dot(current_mean, [1.0, 1.5]))

            data_seq.extend(block_data)
            llcid_seq.extend([llcid_val] * block_size)
            hlcid_seq.extend([high_level_latent] * block_size)

        return data_seq, llcid_seq, hlcid_seq


# ──────────────────────────────────────────────────────────────────────────────
# Out-of-Distribution Test Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TaskOODDataset(BaseTaskDataset):
    """
    Sweeps context means from -0.2 to 1.2 (step 0.1) to test generalization
    beyond the training range [0.2, 0.8].
    """

    def __init__(self, config, no_of_blocks=None):
        self._mean_values = np.round(np.arange(-0.2, 1.3, 0.1), 2)
        # Override no_of_blocks to match the sweep length
        config = config  # keep reference
        super().__init__(config, no_of_blocks=len(self._mean_values))

    def _generate_block_sizes(self):
        # One fixed-size block per mean value
        return [self.block_size] * len(self._mean_values)

    def generate_sequences(self):
        data_seq, llcid_seq, hlcid_seq = [], [], []
        for mean in self._mean_values:
            block = self.rng.normal(mean, self.config.default_std, self.block_size)
            data_seq.extend(block)
            llcid_seq.extend([float(mean)] * self.block_size)
            hlcid_seq.extend([0.0] * self.block_size)
        return data_seq, llcid_seq, hlcid_seq


# ──────────────────────────────────────────────────────────────────────────────
# Sequence Learning Task (Beukers et al. 2024)
# ──────────────────────────────────────────────────────────────────────────────

class seq_learnDataset(Dataset):
    """
    Blocked sequence-learning task from Beukers et al. (2024).
    https://osf.io/preprints/psyarxiv/9bptj

    Two task types (high-level latent 0 or 1), each with two transition patterns
    (low-level latent 0 or 1), over a 10-state one-hot space.

    Sequences:
        Task 0, LLLat 0: [0, 1, 3, 5, 7, 9]
        Task 0, LLLat 1: [0, 1, 4, 6, 8, 9]
        Task 1, LLLat 0: [0, 2, 3, 6, 7, 9]
        Task 1, LLLat 1: [0, 2, 4, 5, 8, 9]
    """

    def __init__(self, config, no_of_blocks=None):
        self.config = config
        self.no_of_blocks = no_of_blocks if no_of_blocks is not None else config.no_of_blocks
        self.block_size = config.block_size
        self.task_length = 6
        self.no_of_tasks = 2
        self.space_size = 10
        self.rng = np.random.RandomState(config.env_seed)
        self.shuffle_or_interleave = config.shuffle_or_interleave
        self.random_transition_shuffle_or_interleave = config.random_transition_shuffle_or_interleave

        self.states, self.high_level_latents, self.low_level_latents = self._generate_data()
        self.data = list(zip(self.states, self.high_level_latents, self.low_level_latents))

    def __len__(self):
        return (self.no_of_blocks * self.block_size - self.config.seq_len) // self.config.stride + 1

    def __getitem__(self, idx):
        start = idx * self.config.stride
        end = start + self.config.seq_len
        states, hlats, llats = zip(*self.data[start:end])
        states = torch.from_numpy(np.stack(states, dtype=np.float32)).reshape(-1, self.space_size)
        hlats = torch.from_numpy(np.stack(hlats, dtype=np.float32)).reshape(-1, 1)
        llats = torch.from_numpy(np.stack(llats, dtype=np.float32)).reshape(-1, 1)
        return states, hlats, llats

    def _generate_data(self):
        states, low_level_latent_list, high_level_latent_list = [], [], []

        for block in range(self.no_of_blocks):
            if self.shuffle_or_interleave == 'shuffle':
                high_level_latent = self.rng.choice(self.no_of_tasks)
            else:
                high_level_latent = int(block % 2)

            if self.random_transition_shuffle_or_interleave == 'shuffle':
                low_level_latents = self.rng.choice([0, 1], size=(self.block_size // self.task_length) + 1)
            else:
                ll = int((block + 1) % 2)
                low_level_latents = [ll] * (self.block_size // self.task_length)

            block_counter = 0
            for low_level_latent in low_level_latents:
                task_sequence = self._generate_states(high_level_latent, low_level_latent)
                block_counter += len(task_sequence)
                if block_counter > self.block_size:
                    fill = self.block_size - (block_counter - self.task_length)
                    states.extend(task_sequence[:fill])
                    low_level_latent_list.extend([low_level_latent] * fill)
                    high_level_latent_list.extend([high_level_latent] * fill)
                    break
                else:
                    states.extend(task_sequence)
                    low_level_latent_list.extend([low_level_latent] * self.task_length)
                    high_level_latent_list.extend([high_level_latent] * self.task_length)

        states = np.eye(self.space_size)[states] * self.config.observation_scale
        return states, low_level_latent_list, high_level_latent_list

    def _generate_states(self, high_level_latent, low_level_latent):
        """Return the integer state sequence for a given task and transition type."""
        use_det2 = self.config.seq_learn_use_deterministic_transition_2
        if high_level_latent == 0:
            return [0, 1, 3, 5, 7, 9] if (low_level_latent == 0 or use_det2) else [0, 1, 4, 6, 8, 9]
        else:
            return [0, 2, 3, 6, 7, 9] if (low_level_latent == 0 or use_det2) else [0, 2, 4, 5, 8, 9]


# ──────────────────────────────────────────────────────────────────────────────
# Rotating Targets Task (Yu et al. 2025)
# ──────────────────────────────────────────────────────────────────────────────

class RotatingTargetsDataset(BaseTaskDataset):
    """
    Predictive-inference task from Yu et al. (2025).

    5 shield colors each have a target location on a circular arena. After several
    mini-blocks (a state-block), all targets rotate by the same angle without warning.
    Each paper-trial maps to two alternating timesteps:
        cue     t:   [color_onehot (n_colors), 0.0, 0.0]
        outcome t+1: [0.0, ..., 0.0, attack_x, attack_y]

    With frame prediction (predict_first_frame=False), the model at the cue step
    must predict the attack position — the key learning signal.
    """

    def generate_sequences(self):
        n_colors  = self.config.n_colors
        noise_std = self.config.noise_std
        rotations = self._get_rotations()
        angles_0  = np.linspace(0, 2 * np.pi, n_colors, endpoint=False)
        base_targets = self.config.target_radius * np.stack(
            [np.cos(angles_0), np.sin(angles_0)], axis=1)  # (n_colors, 2)

        data_seq, llcid_seq, hlcid_seq = [], [], []
        self._prev_rotation_deg = None
        for block_idx, block_size in enumerate(self.block_sizes):
            rotation_deg = self._rotation_for_block(block_idx, rotations)
            rotation_rad = np.deg2rad(rotation_deg)
            c, s = np.cos(rotation_rad), np.sin(rotation_rad)
            R = np.array([[c, -s], [s, c]])
            rotated_targets = (R @ base_targets.T).T  # (n_colors, 2)
            # Empty unless config.enable_context_output() was called, in which case these dims
            # carry the rotation for the model to report back — hidden from its input by
            # input_feed_mask, supervised by output_loss_mask.
            ctx = self._context_vector(rotation_deg, rotation_rad)
            n_miniblocks = block_size // (n_colors * 2)
            for _ in range(n_miniblocks):
                color_order = self.rng.permutation(n_colors)
                for color in color_order:
                    attack = rotated_targets[color] + noise_std * self.rng.standard_normal(2)
                    color_onehot = np.zeros(n_colors)
                    color_onehot[color] = 1.0
                    cue_obs     = np.concatenate([color_onehot, ctx, [0.0, 0.0]])
                    outcome_obs = np.concatenate([np.zeros(n_colors), ctx, attack])
                    data_seq.extend([cue_obs, outcome_obs])
                    llcid_seq.extend([rotation_rad, rotation_rad])
                    hlcid_seq.extend([float(block_idx), float(block_idx)])
        return data_seq, llcid_seq, hlcid_seq

    def _get_rotations(self):
        return self.config.train_rotations

    def _rotation_for_block(self, block_idx, rotations):
        """Rotation (degrees) for one state-block.

        'cyclic' (the default) walks the list in order — deterministic and seed-independent, so
        the schedule is predictable. 'random_no_repeat' redraws each block, excluding the previous
        one so llcid always changes at a boundary (get_block_switches keys on the rotation value,
        not the block index, and would merge two same-angle blocks).
        """
        order = getattr(self.config, 'rotation_block_order', 'cyclic')
        if order == 'cyclic':
            return rotations[block_idx % len(rotations)]
        if order != 'random_no_repeat':
            raise ValueError(f"Unknown rotation_block_order '{order}'. "
                             "Choose 'cyclic' or 'random_no_repeat'.")
        choices = [r for r in rotations if r != self._prev_rotation_deg]
        if not choices:                      # single rotation configured — nothing to alternate
            choices = list(rotations)
        rotation_deg = float(self.rng.choice(choices))
        self._prev_rotation_deg = rotation_deg
        return rotation_deg

    def _context_vector(self, rotation_deg, rotation_rad):
        """Context dims appended to every observation, or an empty array when disabled.

        See RotatingTargetsConfig.enable_context_output for the layout and the encodings.
        """
        encoding = getattr(self.config, 'context_output_encoding', None)
        if encoding is None:
            return np.empty(0)
        if encoding == 'circular':
            r = self.config.target_radius
            return np.array([r * np.cos(rotation_rad), r * np.sin(rotation_rad)])
        if encoding == 'one_hot':
            rotations = self._get_rotations()
            vec = np.zeros(len(rotations))
            vec[int(np.argmin(np.abs(np.asarray(rotations) - rotation_deg)))] = 1.0
            return vec
        raise ValueError(f"Unknown context_output_encoding '{encoding}'. "
                         "Choose 'circular' or 'one_hot'.")


class RotatingTargetsTestDataset(RotatingTargetsDataset):
    """Test split: uses config.test_rotations instead of train_rotations."""

    def _get_rotations(self):
        return self.config.test_rotations or self.config.train_rotations


# ──────────────────────────────────────────────────────────────────────────────
# Mean Prediction Task
# ──────────────────────────────────────────────────────────────────────────────

class MeanPredictionDataset(BaseTaskDataset):
    """
    1D contextual switching task where the training target is the latent mean.

    Each timestep element is 2D: [observation, ground_truth_mean].
    Use with MeanPredictionConfig, which sets:
        input_feed_mask  = [1, 0]  — hides dim 1 (mean) from the model input
        output_loss_mask = [0, 1]  — trains only on the mean-prediction output dim
    The model sees noisy observations and must learn to output the inferred mean.
    """

    def generate_sequences(self):
        latent_values = self.config.training_data_means
        data_rng = np.random.default_rng(self.config.env_seed)

        current_latent = (min(latent_values) if self.config.start_always_on_the_same_block
                          else self.rng.choice(latent_values))

        data_seq, llcid_seq, hlcid_seq = [], [], []
        for i, block_size in enumerate(self.block_sizes):
            options = [v for v in latent_values if v != current_latent]
            if options:
                current_latent = self.rng.choice(options)
            block_obs = data_rng.normal(current_latent, self.config.default_std, block_size)
            for obs in block_obs:
                data_seq.append([obs, current_latent])  # [observation, mean]
            llcid_seq.extend([current_latent] * block_size)
            hlcid_seq.extend([0.0] * block_size)
        return data_seq, llcid_seq, hlcid_seq


# ──────────────────────────────────────────────────────────────────────────────
# Flanker Task
# ──────────────────────────────────────────────────────────────────────────────

class FlankerTaskDataset(BaseTaskDataset):
    """
    Flanker task pretraining dataset.

    5 arrow slots: [far_left, near_left, center, near_right, far_right].
    Each context block has a fixed target slot; companion slot is drawn randomly.
    Companion direction is correlated with target per config.p_corr_by_distance[distance].
    Arrow observations are noisy continuous values (positive = right, negative = left),
    resampled with fresh noise each timestep so evidence accumulates over time.

    Input: 6D augmented vector [5 slot obs | hidden target direction].
    Context IDs: float(target_slot) — routed to oracle Z via what_latent_to_use='context_ids'.
    HLCIDs: congruency flag (1.0 congruent, 0.0 incongruent) — for analysis/logging only.
    """

    def generate_sequences(self):
        cfg = self.config
        data_seq, context_ids_seq, hlcid_seq = [], [], []

        for block_idx, blk_size in enumerate(self.block_sizes):
            target_slot = block_idx % cfg.n_slots   # rotate through slots across blocks
            n_trials = blk_size // cfg.arrows_duration
            for _ in range(n_trials):
                true_direction = self.rng.choice([-1.0, 1.0])
                companion_slot = int(self.rng.choice(
                    [s for s in range(cfg.n_slots) if s != target_slot]
                ))
                dist      = abs(companion_slot - target_slot)
                p_corr    = cfg.p_corr_by_distance[dist]
                companion_dir = (true_direction if self.rng.random() < p_corr
                                 else -true_direction)
                congruent = float(companion_dir == true_direction)
                for _ in range(cfg.arrows_duration):
                    obs = self.rng.normal(0.0, cfg.bg_noise_std, cfg.n_slots).astype(np.float32)
                    obs[target_slot]    = float(true_direction  * cfg.signal_strength
                                                + self.rng.normal(0, cfg.arrow_noise_std))
                    obs[companion_slot] = float(companion_dir   * cfg.signal_strength
                                                + self.rng.normal(0, cfg.arrow_noise_std))
                    full_obs = np.append(obs, np.float32(true_direction))   # dim 5 = hidden
                    data_seq.append(full_obs)
                    context_ids_seq.append(float(target_slot))  # slot integer → oracle Z
                    hlcid_seq.append(congruent)                  # 1.0 congruent, 0.0 incongruent

        return data_seq, context_ids_seq, hlcid_seq


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY['contextual_switching_task'] = TaskDataset
DATASET_REGISTRY['contextual_switching_task_2D'] = TaskDataset2D
DATASET_REGISTRY['seq_learn'] = seq_learnDataset
DATASET_REGISTRY['rotating_targets'] = RotatingTargetsDataset
DATASET_REGISTRY['rotating_targets_test'] = RotatingTargetsTestDataset
DATASET_REGISTRY['mean_prediction'] = MeanPredictionDataset


# ──────────────────────────────────────────────────────────────────────────────
# Gradual Mean Prediction Task
# ──────────────────────────────────────────────────────────────────────────────

class GradualMeanPredictionDataset(BaseTaskDataset):
    """
    Mean prediction task with linearly ramped context transitions.

    Instead of abrupt switches, the generative mean steps toward the block's
    target at config.morph_rate per timestep, starting from 0.5 at t=0.
    The label (dim 1) at each timestep is the *instantaneous* ramping mean,
    so the model must track the current mean throughout the ramp.

    Minimum block size to fully reach the target:
        ceil(|target - current| / morph_rate)
    With default means [0.2, 0.8] and morph_rate=0.05 this is 12 timesteps.
    """

    def generate_sequences(self):
        latent_values = self.config.training_data_means
        morph_rate = getattr(self.config, 'morph_rate', 0.05)
        data_rng = np.random.default_rng(self.config.env_seed)

        first_block_mean = getattr(self.config, 'first_block_mean', None)
        if first_block_mean is not None:
            target_mean = float(first_block_mean)
        elif self.config.start_always_on_the_same_block:
            target_mean = min(latent_values)
        else:
            target_mean = self.rng.choice(latent_values)

        data_seq, llcid_seq, hlcid_seq = [], [], []
        current_mean = 0.5  # always begin at the midpoint

        for i, block_size in enumerate(self.block_sizes):
            if i > 0:
                options = [v for v in latent_values if v != target_mean]
                target_mean = self.rng.choice(options) if options else target_mean

            for _ in range(block_size):
                # Step current_mean one morph_rate increment toward target
                diff = target_mean - current_mean
                step = np.sign(diff) * min(morph_rate, abs(diff))
                current_mean = current_mean + step

                obs = data_rng.normal(current_mean, self.config.default_std)
                data_seq.append([obs, target_mean])
                llcid_seq.append(float(target_mean))
                hlcid_seq.append(0.0)

        return data_seq, llcid_seq, hlcid_seq


DATASET_REGISTRY['mean_prediction_gradual'] = GradualMeanPredictionDataset
DATASET_REGISTRY['flanker_pretrain'] = FlankerTaskDataset


class FlankerTaskStage2Dataset(BaseTaskDataset):
    """
    Stage 2 full-flanker dataset. Target always center (slot 2).
    All 4 other slots are flankers. Blocks alternate between fully
    congruent (all flankers = target direction) and fully incongruent
    (all flankers = opposite direction).
    context_ids: congruency flag (1.0 / 0.0) for block shading.
    hlcids: float(target_slot) = 2.0 always.
    """

    def generate_sequences(self):
        cfg = self.config
        data_seq, context_ids_seq, hlcid_seq = [], [], []
        target_slot = cfg.n_slots // 2   # always center (2)

        for block_idx, blk_size in enumerate(self.block_sizes):
            is_congruent   = (block_idx % 2 == 0)
            congruent_flag = 1.0 if is_congruent else 0.0

            n_trials = blk_size // cfg.arrows_duration
            for _ in range(n_trials):
                true_direction = self.rng.choice([-1.0, 1.0])
                flanker_dir    = true_direction if is_congruent else -true_direction

                for _ in range(cfg.arrows_duration):
                    obs = np.zeros(cfg.n_slots, dtype=np.float32)
                    for slot in range(cfg.n_slots):
                        direction = true_direction if slot == target_slot else flanker_dir
                        obs[slot] = float(direction * cfg.signal_strength
                                          + self.rng.normal(0, cfg.arrow_noise_std))
                    full_obs = np.append(obs, np.float32(true_direction))
                    data_seq.append(full_obs)
                    context_ids_seq.append(float(target_slot))  # slot integer → oracle Z
                    hlcid_seq.append(congruent_flag)             # congruency for analysis

        return data_seq, context_ids_seq, hlcid_seq


DATASET_REGISTRY['flanker_stage2'] = FlankerTaskStage2Dataset


class FlankerTaskStage3Dataset(BaseTaskDataset):
    """
    Stage 3: 4 block types crossing distance × congruency.
    Target always center (slot 2). No no-flanker baseline.

    Block types (block_idx % 4):
      0 — near-congruent:   slots 1 & 3, flanker = target direction
      1 — near-incongruent: slots 1 & 3, flanker = opposite direction
      2 — far-congruent:    slots 0 & 4, flanker = target direction
      3 — far-incongruent:  slots 0 & 4, flanker = opposite direction

    context_ids: congruency flag (1.0/0.0) for block shading.
    hlcids: block type (0.0–3.0) for distance × congruency analysis.
    """

    _BLOCK_TYPES = [
        ([1, 3], True),   # 0: near-congruent
        ([1, 3], False),  # 1: near-incongruent
        ([0, 4], True),   # 2: far-congruent
        ([0, 4], False),  # 3: far-incongruent
    ]

    def generate_sequences(self):
        cfg = self.config
        data_seq, context_ids_seq, hlcid_seq = [], [], []
        target_slot = cfg.n_slots // 2   # always center (2)

        for block_idx, blk_size in enumerate(self.block_sizes):
            block_type                 = block_idx % 4
            flanker_slots, is_congruent = self._BLOCK_TYPES[block_type]
            congruent_flag             = 1.0 if is_congruent else 0.0

            n_trials = blk_size // cfg.arrows_duration
            for _ in range(n_trials):
                true_direction = self.rng.choice([-1.0, 1.0])
                flanker_dir    = true_direction if is_congruent else -true_direction

                for _ in range(cfg.arrows_duration):
                    obs = self.rng.normal(0.0, cfg.bg_noise_std, cfg.n_slots).astype(np.float32)
                    obs[target_slot] = float(true_direction * cfg.signal_strength
                                             + self.rng.normal(0, cfg.arrow_noise_std))
                    for fs in flanker_slots:
                        obs[fs] = float(flanker_dir * cfg.signal_strength
                                        + self.rng.normal(0, cfg.arrow_noise_std))
                    full_obs = np.append(obs, np.float32(true_direction))
                    data_seq.append(full_obs)
                    context_ids_seq.append(congruent_flag)   # for block shading
                    hlcid_seq.append(float(block_type))      # 0–3 for analysis

        return data_seq, context_ids_seq, hlcid_seq


DATASET_REGISTRY['flanker_stage3'] = FlankerTaskStage3Dataset


class FlankerRandomTrialsDataset(BaseTaskDataset):
    """
    Fully randomized trials (no blocks) — the design humans actually perform.

    Each trial independently draws one of 4 types (near/far × cong/incong):
    congruency with probability cfg.p_congruent, and near-vs-far with probability
    cfg.p_near within each congruency. block_size should equal arrows_duration so
    each DataLoader batch = 1 trial.

    context_ids: congruency flag (1.0 / 0.0) for block shading.
    hlcids: trial type (0.0–3.0) — 0 near-cong, 1 near-incong, 2 far-cong, 3 far-incong.
    """

    _TRIAL_TYPES = [
        ([1, 3], True),   # 0: near-congruent
        ([1, 3], False),  # 1: near-incongruent
        ([0, 4], True),   # 2: far-congruent
        ([0, 4], False),  # 3: far-incongruent
    ]

    def generate_sequences(self):
        cfg = self.config
        data_seq, context_ids_seq, hlcid_seq = [], [], []
        target_slot = cfg.n_slots // 2   # always center (2)

        p_congruent = getattr(cfg, 'p_congruent', 0.5)
        p_near      = getattr(cfg, 'p_near', 0.5)

        for blk_size in self.block_sizes:
            n_trials = blk_size // cfg.arrows_duration
            for _ in range(n_trials):
                is_congruent = self.rng.random() < p_congruent
                is_near      = self.rng.random() < p_near
                # 0 near-cong, 1 near-incong, 2 far-cong, 3 far-incong
                trial_type   = (0 if is_near else 2) + (0 if is_congruent else 1)
                flanker_slots, _ = self._TRIAL_TYPES[trial_type]
                true_direction              = self.rng.choice([-1.0, 1.0])
                flanker_dir                 = true_direction if is_congruent else -true_direction
                congruent_flag              = 1.0 if is_congruent else 0.0

                for _ in range(cfg.arrows_duration):
                    obs = self.rng.normal(0.0, cfg.bg_noise_std, cfg.n_slots).astype(np.float32)
                    obs[target_slot] = float(true_direction * cfg.signal_strength
                                             + self.rng.normal(0, cfg.arrow_noise_std))
                    for fs in flanker_slots:
                        obs[fs] = float(flanker_dir * cfg.signal_strength
                                        + self.rng.normal(0, cfg.arrow_noise_std))
                    full_obs = np.append(obs, np.float32(true_direction))
                    data_seq.append(full_obs)
                    context_ids_seq.append(congruent_flag)
                    hlcid_seq.append(float(trial_type))

        return data_seq, context_ids_seq, hlcid_seq


DATASET_REGISTRY['flanker_random'] = FlankerRandomTrialsDataset

# Deprecated names, kept so the archived blocked-stage script still runs.
DATASET_REGISTRY['flanker_stage4'] = FlankerRandomTrialsDataset
FlankerTaskStage4Dataset = FlankerRandomTrialsDataset
