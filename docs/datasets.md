# Datasets

`datasets.py` — base class `BaseTaskDataset`, concrete classes `TaskDataset`, `TaskDataset2D`, `TaskOODDataset`, `seq_learnDataset`, registry `DATASET_REGISTRY`

---

## Adding a New Dataset (Start Here)

This is the main thing a newcomer will want to do. The full workflow is three steps:

### Step 1 — Subclass `BaseTaskDataset`

Implement `generate_sequences()`, which returns three flat lists of equal length (one entry per timestep):

```python
class My2DGaussianDataset(BaseTaskDataset):
    """Two 2D Gaussian contexts that switch in blocks."""

    def generate_sequences(self):
        data_seq, llcid_seq, hlcid_seq = [], [], []
        contexts = [[0.2, 0.8], [0.8, 0.2]]
        current = 0

        for i, block_size in enumerate(self.block_sizes):
            current = 1 - current                           # alternate contexts
            mean = contexts[current]
            block = self.rng.normal(mean, self.config.default_std, (block_size, 2))
            data_seq.extend(block)
            llcid_seq.extend([float(current)] * block_size)
            hlcid_seq.extend([0.0] * block_size)

        return data_seq, llcid_seq, hlcid_seq
```

`BaseTaskDataset.__init__` automatically calls `generate_sequences()` and stores the results. `__getitem__` windows them into tensors of shape `(seq_len, input_size)` using `config.seq_len`, `config.stride`, and `config.input_size`.

### Step 2 — Register it

```python
from datasets import DATASET_REGISTRY
DATASET_REGISTRY['my_2d_gaussian'] = My2DGaussianDataset
```

Or add the line directly at the bottom of `datasets.py` alongside the existing registrations.

### Step 3 — Update the config

```python
config = ContextualSwitchingTaskConfig()
config.input_size = 2           # observations are 2D
config.output_size = 2          # model predicts 2D
config.dataset_name = 'my_2d_gaussian'

logger, model, config, figs = train_model(config, seed=0)
```

That's it. No edits to `create_datasets_and_loaders` are needed.

---

## `BaseTaskDataset` — Abstract Base Class

All datasets that use block-structured data should inherit from this class.

```python
class BaseTaskDataset(Dataset, ABC):
    def __init__(self, config, no_of_blocks=None): ...
    def _generate_block_sizes(self): ...   # uses config.block_duration_distribution
    def generate_sequences(self): ...      # ABSTRACT — implement this
    def __len__(self): ...
    def __getitem__(self, index): ...      # windows sequences into tensors
```

### Constructor

```python
dataset = MyDataset(config)
dataset = MyDataset(config, no_of_blocks=20)   # override no_of_blocks from config
```

`no_of_blocks` defaults to `config.no_of_blocks`. Pass it explicitly when you want to create a dataset of a specific length (e.g., in `run_generalized_tests`).

### Block size generation (`_generate_block_sizes`)

Builds `self.block_sizes` — a list of `num_blocks` integers.

| `config.block_duration_distribution` | Block lengths |
|---|---|
| `'fixed_block_size'` or `'fixed'` | All exactly `config.block_size` |
| `'geometric'` | Each drawn from `Geometric(2/block_size)`, clipped to `[block_size/1.5, block_size*2]` |

The block sizes are available as `self.block_sizes` inside `generate_sequences()`.

### `__getitem__`

```python
dataset[index]:
    start = index * config.stride
    data_slice  = data_seq[start : start + seq_len]      # list of scalars or arrays
    llcid_slice = llcid_seq[start : start + seq_len]
    hlcid_slice = hlcid_seq[start : start + seq_len]

    data_t  = tensor(data_slice).reshape(seq_len, config.input_size)   # ← uses input_size
    llcid_t = tensor(llcid_slice).reshape(seq_len, 1)
    hlcid_t = tensor(hlcid_slice).reshape(seq_len, 1)

    return data_t, llcid_t, hlcid_t
```

The reshape uses `config.input_size`, so changing input dimensionality only requires setting the right value in config — no dataset code change needed.

---

## Factory Function

```python
dataset, dataset_test, train_loader, test_loader = create_datasets_and_loaders(config, pattern=None)
```

Looks up `config.dataset_name` in `DATASET_REGISTRY`, instantiates train and test copies, and wraps them in `DataLoader`s with `shuffle=False` (order must be preserved for latent carryover).

All loaders yield 3-tuples:
```
(data, llcids, hlcids)
data:   (batch, seq_len, input_size)
llcids: (batch, seq_len, 1)
hlcids: (batch, seq_len, 1)
```

`pattern` is a filter for block selection used in some OOD test splits (rare).

### Registry

```python
DATASET_REGISTRY = {
    'contextual_switching_task':    TaskDataset,
    'contextual_switching_task_2D': TaskDataset2D,
    'seq_learn':                    seq_learnDataset,
    'rotating_targets':             RotatingTargetsDataset,
    'rotating_targets_test':        RotatingTargetsTestDataset,
    # add your dataset here
}
```

When a config defines `test_dataset_name`, `create_datasets_and_loaders` will instantiate the test set from that key instead of `dataset_name`. This is how `RotatingTargetsConfig` separates train rotations from test rotations.

---

## `TaskDataset` — 1D Contextual Switching

**`dataset_name = 'contextual_switching_task'`**

Scalar observations drawn from `Normal(mean_A, std)` or `Normal(mean_B, std)` depending on the current context block. Contexts switch each block with no explicit label — the model must infer context from prediction errors.

### Data generation (`generate_sequences`)

1. Assigns a starting context (context A = `min(training_data_means)` if `start_always_on_the_same_block`).
2. For each block: samples from the other context, then draws `block_size` observations from `Normal(current_mean, default_std)`.
3. High-level context changes every `high_level_latent_change_interval_in_blocks` blocks.
4. If `use_high_task_structure=True`, the per-block RNG is seeded from the high-level latent, creating structured within-block correlations.

### Key config fields

| Field | Effect |
|---|---|
| `training_data_means` | `[0.2, 0.8]` — means for contexts A and B |
| `default_std` | 0.1 — observation noise |
| `no_of_blocks` | Total number of blocks |
| `block_size` | Observations per block |
| `block_duration_distribution` | Fixed vs. geometric block lengths |
| `start_always_on_the_same_block` | If True, always begin with context A |
| `use_high_task_structure` | Enable hierarchical context seeding |

### `truncate_data_sequence(end=None)`

Shortens the stored sequences in place. Used in COIN-style experiments where only a fixed-length window of data is relevant.

---

## `TaskDataset2D` — 2D Contextual Switching

**`dataset_name = 'contextual_switching_task_2D'`**

Same block structure as `TaskDataset`, but observations are 2D vectors. Three possible context means are built from permutations of `training_data_means`.

Set `config.input_size = 2` and `config.output_size = 2` when using this dataset.

### EM demo mode

If `config.use_EM_demo_data = True`, observations are drawn from pre-generated sklearn `make_blobs` clusters (3 clusters, 1000 total samples) instead of Gaussians. Each block is assigned one cluster randomly.

---

## `TaskOODDataset` — Out-of-Distribution Sweep

Tests generalization to means never seen during training. Generates one block per mean value in `np.arange(-0.2, 1.3, 0.1)`, spanning well outside the training range `[0.2, 0.8]`.

Used by `run_generalized_tests()` in `train_and_infer_functions.py`.

---

## `seq_learnDataset` — Sequence Learning Task

**`dataset_name = 'seq_learn'`**

Beukers et al. (2024) blocked training paradigm. Does **not** inherit from `BaseTaskDataset` because it uses a discrete state space rather than continuous observations, but it follows the same 3-tuple return convention.

### Task structure

4 unique 6-step sequences over a 10-state one-hot space:

| Task (hlcid) | Transition (llcid) | Sequence |
|---|---|---|
| 0 | 0 | `[0, 1, 3, 5, 7, 9]` |
| 0 | 1 | `[0, 1, 4, 6, 8, 9]` |
| 1 | 0 | `[0, 2, 3, 6, 7, 9]` |
| 1 | 1 | `[0, 2, 4, 5, 8, 9]` |

States are returned as one-hot vectors scaled by `config.observation_scale`.

### Block ordering

Controlled by `config.shuffle_or_interleave` (`'interleave'` alternates tasks, `'shuffle'` randomizes) and `config.random_transition_shuffle_or_interleave` for low-level transitions within blocks.

### Key config fields

| Field | Effect |
|---|---|
| `task_length` | 6 — fixed sequence length |
| `no_of_blocks` | Total blocks |
| `block_size` | Steps per block |
| `observation_scale` | Scale applied to one-hot states |
| `shuffle_or_interleave` | Task ordering |
| `seq_learn_use_deterministic_transition_2` | Force transition type 2 to be deterministic |

---

## DataLoader Conventions

- `shuffle=False` always — order must be preserved so latent carryover works.
- `batch_size` from `config.batch_size` (always 1 in standard experiments).
- Default `collate_fn` adds a batch dimension: tensors have shape `(1, seq_len, input_size)`.

---

## Relationship to Training Loop

`predictive_learning()` iterates the DataLoader:
```python
for inputs, llcids, hlcids in dataloader:
    # inputs:  (B=1, seq_len, input_size)
    # llcids:  (B=1, seq_len, 1)  — low-level context IDs; passed to model as taskID in oracle mode
    # hlcids:  (B=1, seq_len, 1)  — high-level context IDs; logged for analysis
```

When `config.what_latent_to_use = 'taskID'`, `llcids` is used directly as the oracle context label, bypassing Z optimization entirely. This is the oracle baseline.
