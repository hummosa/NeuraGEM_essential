# Correlated Noise Sweep — Experiment Design

## Goal

Compare how RNN (short horizon), MRNN (long horizon), and NeuraGEM adapt to observations
with temporally correlated noise, sweeping `noise_correlation_tau` across [1, 2, 4, 6, 10].
Metric: mean |pred − ground-truth mean| in the tail of each block, averaged over 20 seeds.
NeuraGEM additionally sweeps `l2_loss` and `LU_lr`.

---

## New files

| File | Purpose |
|---|---|
| `cst_correlated_noise_sweep.py` | Training loop + SLURM array dispatch (mirrors `cst_run_generalization.py`) |
| `plot_correlated_noise_results.py` | Load pickles, average over seeds, produce comparison figures |

---

## Script: `cst_correlated_noise_sweep.py`

### Model definitions

Three model variants, differing only in `seq_len` and whether latent updates are active:

```python
BASE_PARAM_GRIDS = {
    "rnn_short":  {"seq_len": [5],  "noise_correlation_tau": [1,2,4,6,10], "seed": range(20)},
    "rnn_long":   {"seq_len": [50], "noise_correlation_tau": [1,2,4,6,10], "seed": range(20)},
    "neuragem":   {
        "noise_correlation_tau": [1,2,4,6,10],
        "l2_loss":  [0.0001, 0.0008],   # NeuraGEM-specific
        "LU_lr":    [0.05, 0.1],         # NeuraGEM-specific
        "seed":     range(20),
    },
}
```

`rnn_short` and `rnn_long` set `config.no_of_steps_in_latent_space = 0` (same pattern as
`cst_run_generalization.py`). NeuraGEM uses the default.

### Shared training overrides

```python
BASE_TRAIN_OVERRIDES = {
    "blocked_phase_length": 1000,
    "correlated_noise": True,          # the new flag
    "start_always_on_the_same_block": False,
}
```

### SLURM dispatch

Identical to `cst_run_generalization.py`: `build_experiment_jobs` flattens all
param combinations into a list; `SLURM_ARRAY_TASK_ID` picks one job. SBATCH header:

```bash # Note: no need to worry about this now, I will give you a script that I use for dispatch later. 
#SBATCH --array=0-<total_jobs-1>
python cst_correlated_noise_sweep.py
```

### Result files

One pickle per job, saved to:
```
./exports/correlated_noise/<model_name>/<combo_key>/results_seed-<N>.pkl
```
Each pickle: `{"train_logger": ..., "config": ...}` (test phase optional for this experiment).

---

## Metric extraction — `extract_block_corrects()` in `functions_and_utils.py`

Factor the block-level computation currently inside `plot_logger_analysis` panel A into a
standalone utility so it can be called by both the plot function and the sweep analysis:

```python
def extract_block_corrects(logger, last_ts_in_a_block=15,
                           phases_to_include='Learning and inference') -> np.ndarray:
    """Return per-block mean |pred − ground-truth mean|, tail only.
    Shape: (n_blocks,)
    """
```

`plot_logger_analysis` calls this instead of duplicating the logic.

---

## Implementation order

1. Add `extract_block_corrects()` to `functions_and_utils.py`; update `plot_logger_analysis` to call it.
2. Write `cst_correlated_noise_sweep.py` (copy structure from `cst_run_generalization.py`, swap in the new param grids and train overrides).
