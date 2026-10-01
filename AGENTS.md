# NeuraGEM_essential — notes for AI agents

NeuraGEM: an RNN carrying a latent **Z** that is optimized by gradient descent on
prediction error at every timestep (fast), alongside ordinary weight updates (slow).
`config.Z_lr = 0` turns any run into a plain RNN baseline, which is the standard control
throughout the repo.

## Environment

- **Use `.venv/bin/python`, never bare `python`/`python3`.** The system interpreter has
  no torch or numpy. The venv is Python 3.11 (the README's "3.12" is stale).
- Training and sweeps go to Slurm. Syntax checks, single-config smoke tests, and plotting
  from saved results run inline.

## Layout

Core: `configs.py` (all hyperparameters; `Config` base + task subclasses; `_validate()`
for coupled params), `models.py` (`RNN_with_latent`), `datasets.py` (`BaseTaskDataset`
+ `DATASET_REGISTRY`), `train_and_infer_functions.py` (`predictive_learning()`),
`functions_and_utils.py` (`Logger`, analysis helpers), `plot_style.py`.
`docs/overview.md` is the module map.

Experiments are flat `<task>_*.py` families at the root — config / sweep / analysis /
figures per task (flanker, rotation_*, mean_prediction, cst_correlated_noise). The
exception is `hier_switch/`, a self-contained folder that imports core. Match the
existing family's naming when adding to one, and read that task's `docs/*.md` first.

Config subclasses chain: `FlankerTaskConfig` → `Stage2` → `Stage3`, with
`FlankerRandomTrialsConfig` → `Stage4`; `MeanPredictionConfig` extends
`ContextualSwitchingTaskConfig`.

## Labs: exploratory work goes in `labs/`

Testing an idea — an intervention, an effect, a probe — goes in its own
`labs/<date>_<slug>/` folder with a `NOTE.md` (intervention, how to run, outcome with a
figure, action on main), not in a new root script, a new config switch in core, or a new
section of a task doc. Labs import core; core never imports a lab. If an intervention
must reach inside the training loop, the change to core is a few lines that do nothing
unless a config asks for them (the `config.trial_hook` pattern). Full workflow:
`docs/labs.md`; index: `labs/README.md`.

## Running things

- Sweep scripts take a bare positional mode via `sys.argv` (not argparse), e.g.
  `flanker_sweep.py [mode]`. Every sweep entry point has one default variant.
- `hier_switch/hier_switch_tune.py list|run <TAG>`; the grids live in the `GRIDS` dict
  and the array size is derived from `list`, so it can't drift.
- Slurm submission: a `run_*.sh` wrapper that heredocs an `sbatch` script
  (`hier_switch/run_tune.sh` is the model). Show the script before submitting; array
  jobs need explicit approval.
- Untracked output dirs: `exports/` (one subdir per experiment), `slurm/` (job logs),
  `.claude/` — all gitignored. Results go under `exports/<experiment>/`.

## Conventions

- **Figures are paper panels.** Size only via `plot_style.FigSize` presets; call
  `FigSize.dev()` while exploring instead of inflating `figsize`. No `fontsize=` on
  standard elements, no titles by default, model colours from `plot_style`.
  Full rules: `docs/figure_style.md`.
- Runs record through `Logger` (`docs/logging.md`); analysis and plotting read the
  logger rather than recomputing.
- Prefer plain, explainable mechanisms (e.g. `weight_decay`) over custom switches that
  would need defending in a methods section.
- Check in before tuning a new variable or changing direction.
