"""Eyeball one curriculum tree stage by stage with plot_logger_panels.

Loads a pickle written by rotation_curriculum_sweep.py and draws S1, S2 and S3 for one branch,
so the three stages can be compared on the same panels.

Interactive:
    from inspect_curriculum_run import *
    tree = load_tree(0.2)                       # trait arm; 'RNN' also works
    list_stages(tree)
    show_stitched(tree, 'oracle_z', 0.2)        # all three stages on ONE axis
    show_curriculum(tree, cue_mode='oracle_z', z_lr_test=0.2, x2=6000)   # one fig per stage
    show_stage(tree, ('S3', 'oracle_z', None))  # any single stage

> **The curriculum stages are not phases.** `logger.phases` records only the phases inside one
> train_model call, and each stage is a separate call with its own logger. S1 therefore shows
> ['no inference learning', 'Learning and inference'] — the passive warm-up is 386 of 19722
> timesteps, so it is a sliver at the far left — while S2 and S3 show ['Learning and inference']
> alone: they warm-start from a trained model, so they configure no passive phase, and every
> stage runs run_test_phase=False so the Phase-3 names never appear. Use show_stitched() to see
> S1/S2/S3 annotated as phases on a shared axis.

Command line:
    python inspect_curriculum_run.py --list-runs
    python inspect_curriculum_run.py --train 0.2 --cue oracle_z --z-lr-test 0.2 --x2 6000
    python inspect_curriculum_run.py --train RNN --panels rotating_targets_behavior latent_2d loss
    python inspect_curriculum_run.py --train 0.2 --run-name sep60_16800-8400-8400_head-off_decay-grad
    python inspect_curriculum_run.py --train 0.2 --run-name sep60_2000-4000-3000_head-off_decay-grad

Which run is read comes from rotation_curriculum_config.RUN_NAME, which is built from the stage
lengths, the head setting and the decay mode — so changing S1/S2/S3_LENGTH points everything at a
new directory automatically. Two gotchas that follow from that:
  - In a long-running IPython session, `%autoreload 2` patches functions but does NOT reliably
    re-execute module-level assignments, so RUN_NAME keeps its old value and load_tree() quietly
    reads the previous run. Call refresh() after editing the config.
  - Pass run_name=... / --run-name to read a specific run regardless, e.g. to compare a short
    pilot against an earlier long one. list_runs() shows what is on disk.
load_tree() prints the run directory and the stage lengths the tree was actually built with, so a
mismatch is visible immediately.

Panels that work here: 'rotating_targets_behavior', 'latent_2d' / 'latent', 'loss',
'weights_grad_norm'. NOT 'gradients' or 'latent_effective_lr' — compact_logger drops the fields
they read, so they come out empty. 'context_belief' needs a belief head, which this experiment
deliberately runs without.

> **The Z panel is flat and meaningless during a cued (oracle Z) stage.** `logger.latent_values`
> records `model.Z`, the raw latent *parameter*. Under `what_latent_to_use='context_ids'` the
> gate is built from the ground-truth context inside the forward pass and is never written back
> into `model.Z`, so what the panel shows is whatever S1 left there. A flat line in S2 is not
> evidence that the cued stage has no latent dynamics — it has no latent *parameter* dynamics by
> construction, and the gate driving the network is not plotted. show_stage() warns about this.
> (A flat Z in the `z_lr_test=None` S3 fork *is* meaningful: there the latent really is pinned.)
"""

from __future__ import annotations

try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic('load_ext', 'autoreload')
        _ip.run_line_magic('autoreload', '2')
except Exception:
    pass

#: True in an IPython/Jupyter session, where sys.argv belongs to the kernel rather than to us.
_IN_IPYTHON = globals().get('_ip') is not None

import argparse
import pickle
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt

import plot_style
from functions_and_utils import plot_logger_panels
from rotation_curriculum_analysis import _get_stage, stage_label
from rotation_curriculum_config import (
    CUE_MODES, EXPORT_ROOT, Z_LR_TEST, Z_LR_TRAIN, result_path, train_key,
)

plot_style.set_plot_style()

DEFAULT_PANELS = ['rotating_targets_behavior', 'latent_2d', ]

#: Parent of every run directory, e.g. exports/rotation_curriculum/.
RUNS_DIR = EXPORT_ROOT.parent


def refresh():
    """Re-read rotation_curriculum_config after editing it, and rebind what this module took.

    `%autoreload 2` patches functions and classes but does not reliably re-execute module-level
    assignments, so RUN_NAME / EXPORT_ROOT keep their old values in a long-running session and
    load_tree() quietly keeps reading the previous run's directory. Call this after changing the
    stage lengths, the head setting or the decay mode — all three feed RUN_NAME.
    """
    import importlib

    import rotation_curriculum_analysis as _an
    import rotation_curriculum_config as _cfg

    importlib.reload(_cfg)
    importlib.reload(_an)
    globals().update(
        CUE_MODES=_cfg.CUE_MODES, Z_LR_TEST=_cfg.Z_LR_TEST, Z_LR_TRAIN=_cfg.Z_LR_TRAIN,
        EXPORT_ROOT=_cfg.EXPORT_ROOT, RUNS_DIR=_cfg.EXPORT_ROOT.parent,
        result_path=_cfg.result_path, train_key=_cfg.train_key,
        _get_stage=_an._get_stage, stage_label=_an.stage_label,
    )
    print(f'config reloaded -> {_cfg.RUN_NAME}')
    return _cfg


def list_runs(runs_dir: Path | None = None) -> list:
    """Every run directory on disk, with its tree count. '*' marks the one the config points at."""
    runs_dir = Path(RUNS_DIR if runs_dir is None else runs_dir)
    names = sorted(d.name for d in runs_dir.iterdir() if d.is_dir()) if runs_dir.exists() else []
    for name in names:
        n = len(list((runs_dir / name).rglob('results_seed-*.pkl')))
        mark = '*' if name == EXPORT_ROOT.name else ' '
        print(f'  {mark} {name:<48} {n} trees')
    if not names:
        print(f'  (no runs under {runs_dir})')
    return names


def load_tree(z_lr_train: Any = 0.2, noise_std: float = 0.20, seed: int = 0,
              run_name: str | None = None) -> dict:
    """Load one (trait arm, noise, seed) tree.

    `run_name` overrides the directory the config currently resolves to — useful for comparing a
    new short run against an older one, and for sidestepping a stale interactive session (see
    refresh()). Defaults to whatever RUN_NAME is now.
    """
    path = (result_path(z_lr_train, noise_std, seed) if run_name is None else
            Path(RUNS_DIR) / run_name / train_key(z_lr_train) / f'noise_std-{noise_std}'
            / f'results_seed-{seed}.pkl')
    if not path.exists():
        print(f'Not found: {path}\nRuns on disk:')
        list_runs()
        raise FileNotFoundError(
            f'{path}\nTrait arms: {Z_LR_TRAIN}. Pass run_name=... to pick a different run, or '
            f'call refresh() if you edited the config in this session.')
    with path.open('rb') as f:
        tree = pickle.load(f)
    # Print the run directory and the stage lengths the tree was actually built with, so a stale
    # session or a wrong run_name is visible immediately rather than after a confusing figure.
    m = tree['meta']
    print(f'Loaded {path}')
    print(f'  run={path.parents[2].name}  stages={m["s1_length"]}/{m["s2_length"]}/'
          f'{m["s3_length"]}  s3_z_init={m["s3_z_init"]}  seed={m["seed"]}')
    return tree


def list_stages(tree: dict) -> list:
    """Every stage key in the tree, printed and returned."""
    stages = ['S1'] + [('S2', c) for c in tree['S2']] + [('S3', *k) for k in tree['S3']]
    for s in stages:
        print(f'  {str(s):<28} {stage_label(s)}')
    return stages


def show_stage(tree: dict, stage: Any, panels: Sequence[str] | None = None,
               annotate_phases: str = 'latent_2d', width: float = 6, dpi: int = 200,
               x1: int = 0, x2: int | None = None, save_dir: Path | None = None, **kw):
    """Draw one stage. `stage` is 'S1', ('S2', cue) or ('S3', cue, z_lr_test)."""
    logger, config = _get_stage(tree, stage)
    if logger is None:
        raise KeyError(f'{stage!r} is not in this tree; see list_stages().')
    panels = list(DEFAULT_PANELS if panels is None else panels)
    # annotate_phases names a panel to draw the phase boundaries on; silently does nothing if
    # that panel was not requested, which is easy to miss.
    if annotate_phases and annotate_phases not in panels:
        annotate_phases = panels[-1]

    n = sum(len(e.reshape(-1, e.shape[-1])) for e in logger.context_ids)
    print(f'{stage_label(stage):<34} {n} timesteps, phases={[p for p, _ in logger.phases]}')

    if (getattr(config, 'what_latent_to_use', 'self') != 'self'
            and any(p.startswith('latent') for p in panels)):
        print('    note: this stage runs on the oracle gate, which is built inside the forward '
              'pass and never written to model.Z — the Z panel shows the stale S1 value, not '
              'what drove the network.')

    fig = plot_logger_panels(logger, config, panels, annotate_phases=annotate_phases,
                             width=width, dpi=dpi, x1=x1, x2=x2, **kw)
    fig.suptitle(stage_label(stage), fontsize=8, y=1.005)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        name = ('S1' if stage == 'S1' else '_'.join(str(p) for p in stage))
        out = save_dir / f'{name}.png'
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        print(f'  saved -> {out}')
    return fig


def show_curriculum(tree: dict, cue_mode: str = 'oracle_z', z_lr_test: Any = 0.2, **kw):
    """S1 -> S2 -> S3 for one branch, one figure per stage."""
    return [show_stage(tree, s, **kw) for s in
            ('S1', ('S2', cue_mode), ('S3', cue_mode, z_lr_test))]


# ── The three stages on one axis ──────────────────────────────────────────────
#
# The curriculum stages are NOT phases. `logger.phases` only records the phases inside a single
# train_model call ('no inference learning' when a passive phase is configured, then 'Learning
# and inference'), and each stage is a separate call with its own logger. So S1 shows two phases,
# S2 and S3 show one each, and nothing shows all three stages together.
#
# stitch_stages() glues the three loggers end to end and rewrites `phases` so the stage
# boundaries become what annotate_phases draws. Safe here only because every stage shares
# seq_len, stride and input_size — a stitched logger of stages that differed on those would
# misalign silently.

_STITCH_FIELDS = ('inputs', 'predicted_outputs', 'context_ids', 'hlcids',
                  'latent_values', 'training_losses')


def stitch_stages(tree: dict, stages: Sequence[Any], labels: Sequence[str] | None = None):
    """One pseudo-logger spanning several stages, with the stages as phases.

    Returns (logger, config). The config is the first stage's — the panels only read fields that
    are identical across stages (n_colors, masks, train_rotations), but per-stage settings like
    what_latent_to_use are therefore NOT reflected.
    """
    import copy as _copy

    import numpy as np

    parts = [_get_stage(tree, s) for s in stages]
    missing = [s for s, (lg, _) in zip(stages, parts) if lg is None]
    if missing:
        raise KeyError(f'not in this tree: {missing}; see list_stages().')
    labels = [stage_label(s) for s in stages] if labels is None else list(labels)

    out = _copy.copy(parts[0][0])          # shallow: keeps every attribute panels may touch
    lengths, hl_offset = [], 0.0
    stitched_hl = []
    for logger, _ in parts:
        n = sum(len(np.asarray(e).reshape(-1)) for e in logger.context_ids)
        lengths.append(n)
        # Block indices restart at 0 each stage; offset them so the block shading does not
        # recycle colours across stage boundaries.
        hl = np.concatenate([np.asarray(e) for e in logger.hlcids], axis=0)
        stitched_hl.append(hl + hl_offset)
        hl_offset += float(hl.max()) + 1.0

    for field in _STITCH_FIELDS:
        if field == 'hlcids':
            out.hlcids = [np.concatenate(stitched_hl, axis=0)]
            continue
        chunks = [np.asarray(e) for logger, _ in parts for e in getattr(logger, field, [])]
        setattr(out, field, [np.concatenate(chunks, axis=0)] if chunks else [])

    starts = np.cumsum([0] + lengths[:-1])
    out.phases = [(lab, int(s)) for lab, s in zip(labels, starts)]
    print('stitched: ' + '  |  '.join(f'{lab} ({n})' for lab, n in zip(labels, lengths)))
    return out, parts[0][1]


def show_stitched(tree: dict, cue_mode: str = 'oracle_z', z_lr_test: Any = 0.2,
                  panels: Sequence[str] | None = None, annotate_phases: str = 'latent_2d',
                  width: float = 9, dpi: int = 200, x1: int = 0, x2: int | None = None, **kw):
    """S1 -> S2 -> S3 on a single continuous axis, with the stage boundaries annotated."""
    stages = ('S1', ('S2', cue_mode), ('S3', cue_mode, z_lr_test))
    labels = ['S1 uncued', f'S2 {cue_mode}', f'S3 {cue_mode} a_z={z_lr_test}']
    logger, config = stitch_stages(tree, stages, labels)
    panels = list(DEFAULT_PANELS if panels is None else panels)
    if annotate_phases and annotate_phases not in panels:
        annotate_phases = panels[-1]
    return plot_logger_panels(logger, config, panels, annotate_phases=annotate_phases,
                              width=width, dpi=dpi, x1=x1, x2=x2, **kw)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point. Also callable as main() from IPython, where it uses the defaults."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--train', default='0.2', help=f'trait arm, one of {Z_LR_TRAIN}')
    p.add_argument('--noise', type=float, default=0.20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-name', default='sep60_2000-4000-3000_head-off_decay-grad',
                   help='run directory to read; default is whatever the config resolves to')
    p.add_argument('--list-runs', action='store_true', help='list runs on disk and exit')
    p.add_argument('--cue', default='oracle_z', choices=list(CUE_MODES))
    p.add_argument('--z-lr-test', default='0.2',
                   help=f"S3 fork, one of {Z_LR_TEST} ('None' for the frozen-Z control)")
    p.add_argument('--panels', nargs='+', default=DEFAULT_PANELS)
    p.add_argument('--x1', type=int, default=0)
    p.add_argument('--x2', type=int, default=None)
    p.add_argument('--width', type=float, default=6)
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--save-dir', default=None, help='write PNGs here instead of showing')
    p.add_argument('--list', action='store_true', help='list the stages and exit')
    # In IPython sys.argv is the kernel's, so `main()` there means "defaults", not "parse the
    # kernel's flags". From a shell, fall through to sys.argv as usual.
    if argv is None and _IN_IPYTHON:
        argv = []
    args = p.parse_args(argv)

    def _num(s):
        if s in ('RNN', 'None', 'none'):
            return None if s != 'RNN' else 'RNN'
        return float(s)

    if args.list_runs:
        list_runs()
        return

    tree = load_tree(_num(args.train), args.noise, args.seed, run_name=args.run_name)
    if args.list:
        list_stages(tree)
        return

    show_curriculum(tree, cue_mode=args.cue, z_lr_test=_num(args.z_lr_test),
                    panels=args.panels, x1=args.x1, x2=args.x2, width=args.width,
                    dpi=args.dpi, save_dir=args.save_dir)
    if args.save_dir is None:
        plt.show()


if __name__ == '__main__':
    main()
