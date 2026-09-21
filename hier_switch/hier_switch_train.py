"""
hier_switch_train.py — train NeuraGEM (NG), a plain RNN, or an oracle-Z network on the
hierarchical cue→rule reversal task, then show what it learned.

    .venv/bin/python hier_switch/hier_switch_train.py            # MODEL below
    .venv/bin/python hier_switch/hier_switch_train.py RNN 1      # model, seed

Phases (train_model):
    Learning and inference   WU + LU from scratch, blocked contexts. NG has to discover the
                             contexts itself: no oracle, no context label anywhere in the loss.
    Inference only           weights frozen, Z inferred, fresh trials (own RNG stream).

The three models are one architecture:
    NG      Z updated by gradient descent on the response error, once per trial.
    RNN     LU off, so Z stays at Z_init for good. The hidden state resets every trial, so
            the only route to adaptation is the weights. They stay plastic in the test phase
            as well, since the RNN has nothing else to adapt with.
    Oracle  trained with the true context as a one-hot Z (the weights learn to use a context
            gate). At test the oracle is gone and Z is optimised like NG's. run_test() does this
            on a copy of the trained model at any Z_lr, so one Oracle training brackets the
            latent learning rate that works with weights known to use Z.

Figures (exports/hier_switch/<run_name>/):
    panels_full.pdf     whole run: behaviour, P(correct), Z, dL/dZ
    panels_train.pdf    the last blocks of training
    panels_test.pdf     the first blocks of the test phase
"""

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

#%%
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import plot_style
from plot_style import FigSize
from models import RNN_with_latent
from train_and_infer_functions import train_model, predictive_learning
from functions_and_utils import plot_logger_panels, Logger
from datasets import create_datasets_and_loaders

from hier_switch_config import HierSwitchConfig
from hier_switch_analyses import extract_trials, summarize, print_summary

plot_style.set_plot_style()

MODEL = 'NG'          # 'NG', 'RNN' or 'Oracle'
SEED = 0
RUN_TEST_PHASE = True

if __name__ == '__main__' and len(sys.argv) > 1 and 'ipykernel' not in sys.argv[0]:
    MODEL = sys.argv[1]
    if len(sys.argv) > 2:
        SEED = int(sys.argv[2])


def make_config(model_type, **overrides):
    """Config for one model type. `overrides` are plain attribute assignments, except
    n_train_trials and n_passive_trials, which go through their setters."""
    cfg = HierSwitchConfig(experiment_to_run=f'hier_switch_{model_type}')
    if model_type == 'RNN':
        cfg.no_of_steps_in_latent_space = 0
        cfg.test_no_of_steps_in_weight_space = 1
    elif model_type == 'Oracle':
        cfg.what_latent_to_use = 'context_ids'
        cfg.no_of_steps_in_latent_space = 0
        cfg.test_no_of_steps_in_latent_space = 1
    elif model_type != 'NG':
        raise ValueError(f"MODEL must be 'NG', 'RNN' or 'Oracle', got {model_type!r}")
    n_train = overrides.pop('n_train_trials', None)
    n_passive = overrides.pop('n_passive_trials', None)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    if n_train is not None:
        cfg.set_n_trials(n_train)       # keeps no_of_blocks / blocked_phase_length in sync
    if n_passive is not None:
        cfg.set_n_passive_trials(n_passive)
    return cfg


def build_model(cfg, seed):
    """Seeded model with Z at Z_init rather than zeros.

    With latent_activation='none' a zero Z is a zero gate: every gated hidden unit is wiped at
    every step. Starting from a constant Z_init keeps the gate open, and it is where the RNN's
    Z stays for good.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RNN_with_latent(cfg).to(cfg.device)
    model.set_Z(z_init_like(model.Z, cfg.Z_init))
    return model


def z_init_like(Z, z_init):
    """Z_init broadcast to Z's shape: a scalar on every unit, or one value per unit."""
    return torch.as_tensor(z_init, dtype=Z.dtype, device=Z.device).expand_as(Z).clone()


def run(model_type=MODEL, seed=SEED, run_test_phase=RUN_TEST_PHASE, run_name=None,
        save_model=False, **overrides):
    """Train (passive + active), optionally test, optionally save the trained model.

    save_model writes <export_path>/model.pt: the whole model, whose .config is the run's
    config. After a test phase that config is in test mode (weights frozen, test RNG
    stream); run_test reconfigures a copy for testing anyway. Load with load_model().
    train_model's own save_models is not used: its file name is keyed only on seed, length
    and experiment_to_run, so different runs overwrite each other.
    """
    cfg = make_config(model_type, **overrides)
    cfg.run_name = run_name or f'{model_type}_seed{seed}'
    model = build_model(cfg, seed)
    logger, model, cfg, _ = train_model(cfg, seed=seed, pretrained_model=model,
                                        save_models=False, run_test_phase=run_test_phase)
    if save_model:
        path = cfg.export_path + 'model.pt'
        torch.save(model, path)
        print(f'Saved model: {path}')
    return logger, model, cfg


def load_model(path):
    """(model, config) from a model.pt written by run(save_model=True)."""
    model = torch.load(path, weights_only=False)   # a whole pickled model, not a state dict
    return model, model.config


def run_test(model, cfg, Z_lr=None, run_name=None, n_trials=None, perturb=None, **latent):
    """A test session on a copy of a trained model: weights frozen, Z inferred.

    Z restarts from Z_init (the uninformed gate) and is updated by LU at `Z_lr`, whatever the
    model trained with — for an Oracle, this is the first time its own Z is optimised at all.
    Z_lr is patched onto the live optimizer as well as the config: the optimizer is built at
    construction and never re-reads config.Z_lr (docs/flanker_task.md, Gotcha 1).

    `latent` overrides any other latent-update setting for the test, e.g.
    exponential_increase_steepness=[0] or Z_decay=3e-4. Both are read live: the gradient
    filter is rebuilt from the copy's config when Z is reset below, and Z_decay is read on
    every step (Z_decay_mode='grad'). Two non-latent overrides are also honoured: `WU_lr`
    (patched onto the live weight optimizer, for a plastic-weight test) and
    `block_len_range` (the test stream's block lengths; the RNN baseline uses (250, 350)).

    Returns (logger, model_copy, test_cfg). The logger holds one phase, 'Inference only'.
    """
    tcfg = copy.deepcopy(cfg)
    if run_name is not None:
        tcfg.run_name = run_name
    if Z_lr is not None:
        tcfg.Z_lr = float(Z_lr)
    for k, v in latent.items():
        setattr(tcfg, k, v)
    if n_trials is not None:
        tcfg.n_test_trials = int(n_trials)
    if tcfg.test_no_of_steps_in_latent_space is None and tcfg.no_of_steps_in_latent_space == 0:
        tcfg.test_no_of_steps_in_latent_space = 1      # a test of Z inference needs LU on
    # A per-trial perturbation of the latent update (hier_switch_hooks). The spec is kept on
    # the config so it travels into the session's meta; the hook object is what runs.
    tcfg.perturb = perturb
    from hier_switch_hooks import TrialHook
    tcfg.trial_hook = TrialHook.from_spec(perturb)
    tcfg.reconfigure_for_prediction(tcfg.experiment_to_run)
    tcfg._allow_latent_updates = True

    m = copy.deepcopy(model)
    m.config = tcfg
    # Patch the live optimizer, which never re-reads the config. weight_decay matters when
    # Z_decay_mode is 'optimizer' (or 'both'): there the weight decay is the optimizer's own
    # weight_decay, fixed at construction. In 'grad' mode it is read from the config each step.
    opt_decay = float(tcfg.Z_decay or 0.0) if m._z_decay_mode() in ('optimizer', 'both') else 0.0
    for g in m.Z_optimizer.param_groups:
        g['lr'] = float(tcfg.Z_lr)
        g['weight_decay'] = opt_decay
        # SGD carries a momentum buffer; a hook may switch it on for a window of trials.
        if 'momentum' in g:
            g['momentum'] = float(getattr(tcfg, 'Z_momentum', 0.0) or 0.0)
    # The weight optimizer never re-reads config.WU_lr either. It matters only for a test
    # with plastic weights (the RNN baseline, test_no_of_steps_in_weight_space=1), whose
    # rate at test is a knob of its own: 3e-3 on 250-350-trial blocks, where 1e-3 only
    # half-recovers the output and 1e-2 thrashes (docs/hier_switch_task.md, v17).
    if 'WU_lr' in latent:
        for g in m.W_optimizer.param_groups:
            g['lr'] = float(tcfg.WU_lr)
    m.set_Z(z_init_like(m.Z, tcfg.Z_init))

    _, _, _, loader = create_datasets_and_loaders(tcfg)
    logger = Logger()
    logger.config = tcfg
    logger.log_phase('Inference only')
    predictive_learning(logger, tcfg, loader, m)
    return logger, m, tcfg


def report(logger, cfg, label=''):
    trials = extract_trials(logger, cfg)
    res_tr = summarize(trials, 'Learning and inference', last_frac=0.25)
    res_te = summarize(trials, 'Inference only')
    print_summary(res_tr, f'{label} train (last 25%)')
    print_summary(res_te, f'{label} test')
    return trials, res_tr, res_te


def block_window(trials, cfg, phase, n_blocks=8, from_end=True):
    """Timestep window [x1, x2) spanning n_blocks context blocks at one end of a phase."""
    idx = np.flatnonzero(trials['phase'] == phase)
    if len(idx) == 0:
        return None
    starts = idx[trials['since'][idx] == 1]
    if from_end:
        k0 = starts[-n_blocks] if len(starts) >= n_blocks else idx[0]
        k1 = idx[-1] + 1
    else:
        k0 = idx[0]
        k1 = starts[n_blocks] if len(starts) > n_blocks else idx[-1] + 1
    return int(k0 * cfg.trial_len), int(k1 * cfg.trial_len)


PANELS = ['behavior', 'corrects', 'latent_2d', 'gradients']


def plot_panels(logger, cfg, x1=0, x2=None, filename=None):
    w, h = FigSize.custom(4.0, 0.75)
    fig = plot_logger_panels(logger, cfg, PANELS, x1=x1, x2=x2, width=w, subplot_height=h,
                             dpi=150, annotate_phases='behavior', rasterize=True)
    if filename:
        path = cfg.export_path + filename
        fig.savefig(path, bbox_inches='tight')
        print(f'Exported: {path}')
    return fig


#%%
if __name__ == '__main__':
    logger, model, config = run()
    trials, res_tr, res_te = report(logger, config, label=f'{MODEL} seed {SEED}')

    plot_panels(logger, config, filename='panels_full.pdf')
    win = block_window(trials, config, 'Learning and inference', n_blocks=8, from_end=True)
    if win:
        plot_panels(logger, config, *win, filename='panels_train.pdf')
    win = block_window(trials, config, 'Inference only', n_blocks=8, from_end=False)
    if win:
        plot_panels(logger, config, *win, filename='panels_test.pdf')
    plt.show()
