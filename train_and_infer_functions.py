"""
train_and_infer_functions.py — Training loop for NeuraGEM.

Core algorithm (predictive_learning)
──────────────────────────────────────
For each batch the model performs two sequential gradient-descent steps:

    1. Weight Update (WU): standard BPTT.
       Minimise prediction loss by updating all RNN weights.

    2. Latent Update (LU): gradient descent on Z only.
       Minimise the same prediction loss by updating the latent context variable Z,
       without changing any weights.

Z is carried across batches (pass_previous_latent=True by default), so context
inferred in one batch warm-starts the next LU step. This lets the model adapt
to context shifts within a few timesteps.
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
import copy

from tqdm import tqdm
from functions_and_utils import Logger
from datasets import TaskDataset, TaskDataset2D, create_datasets_and_loaders

from models import RNN_with_latent
from functions_and_utils import *
from functions_and_utils import _log_effective_lr

from datasets import *


# ──────────────────────────────────────────────────────────────────────────────
# Loss masking
# ──────────────────────────────────────────────────────────────────────────────

def _mask_loss(loss, config):
    """Zero out output dimensions excluded by config.output_loss_mask.

    config.output_loss_mask : list of 0/1 of length output_size, or None (no masking).
    E.g. [0,0,0,0,0,1,1] keeps only the last two (xy) dims for the rotating-targets task.
    The returned tensor has the same shape as loss; only the backward signal is affected.
    """
    mask = getattr(config, 'output_loss_mask', None)
    if mask is None:
        return loss
    t = torch.tensor(mask, dtype=loss.dtype, device=loss.device)
    return loss * t


# ──────────────────────────────────────────────────────────────────────────────
# Batch preparation
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_batch_inputs(model, config, raw_inputs, batch_context_ids):
    """Move inputs to device, optionally add noise, and build combined input for add-gating."""
    inputs = raw_inputs.to(config.device)
    context_ids = batch_context_ids.to(config.device)

    gated_inputs = inputs
    if config.add_noise_to_input:
        gated_inputs = inputs + torch.randn_like(inputs) * config.noise_std
    input_feed_mask = getattr(config, 'input_feed_mask', None)
    if input_feed_mask is not None:
        m = torch.tensor(input_feed_mask, dtype=inputs.dtype, device=inputs.device)
        gated_inputs = gated_inputs * m
    model_inputs = gated_inputs

    if config.predict_first_frame:
        zero_frame = torch.zeros_like(model_inputs[:, :1, :])
        model_inputs = torch.cat((zero_frame, model_inputs[:, :-1, :]), dim=1)
        latent_inputs = context_ids[:, :]
    else:
        model_inputs = model_inputs[:, :-1, :]
        latent_inputs = context_ids[:, 1:]

    combined_input = model.combine_input_with_latent(
        model_inputs, what_latent=config.what_latent_to_use, taskID=latent_inputs,
    )


    return combined_input, model_inputs, inputs, latent_inputs, #context_ids



# ──────────────────────────────────────────────────────────────────────────────
# Weight Update step
# ──────────────────────────────────────────────────────────────────────────────

def _weight_update_step(model, config, combined_input, model_inputs, inputs, batch_hlcids, criterion):
    """
    One step of BPTT: forward pass → compute loss → backprop → update weights.

    Returns:
        outputs:   stacked predictions (batch, seq_len, output_size)
        full_loss: per-element MSE loss as numpy array (for logging)
        hidden_states: RNN hidden state tuple (for optional logging)
    """
    if config.pass_previous_latent:
        model.detach_Z()
    else:
        model.reset_Z(batch_size=model_inputs.shape[0], seq_len=model_inputs.shape[1])

    model.W_optimizer.zero_grad()
    model.Z_optimizer.zero_grad()

    if config.use_add_gating:
        outputs, hidden_states = model(combined_input, taskID=batch_hlcids,
                                       what_latent=config.what_latent_to_use)
    else:
        # mul_gating applies Z internally during the forward pass
        outputs, hidden_states = model(model_inputs, taskID=batch_hlcids,
                                       what_latent=config.what_latent_to_use)

    outputs = torch.stack(outputs, dim=1)  # (batch, seq_len-1, output_size)

    # Compute the loss # (batch, seq_len-1, output_size), unreduced
    loss = criterion(outputs, inputs) if config.predict_first_frame else criterion(outputs, inputs[:, 1:, :])
    full_loss = loss.detach().cpu().numpy()

    loss_scalar = _mask_loss(loss, config)
    loss_scalar = loss_scalar.sum() if config.loss_reduction_WU == "sum" else loss_scalar.mean()
    loss_scalar.backward()

    if config.no_of_steps_in_weight_space > 0:
        model.W_optimizer.step()

    return outputs, full_loss, hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Latent Update step
# ──────────────────────────────────────────────────────────────────────────────

def _latent_update_step(model, config, model_inputs, inputs, criterion, logger, batch_hlcids):
    """
    One round of LU: gradient descent on Z to minimise prediction loss.

    Each iteration: detach/reset Z → zero_grad → forward → backward → adjust_Z_grads → step.
    Returns the pre-LU loss array (or None if LU is skipped).
    """
    if not config._allow_latent_updates or config.no_of_steps_in_latent_space <= 0:
        return None

    B, seq_len_model, _ = model_inputs.shape
    model._ensure_Z_shape(B, seq_len_model)

    before_optim_loss = None

    for _ in range(config.no_of_steps_in_latent_space):
        if config.pass_previous_latent:
            model.detach_Z()
        else:
            model.reset_Z(batch_size=B, seq_len=seq_len_model)

        model.Z_optimizer.zero_grad()

        if model.use_add_gating:
            cur_input = model.combine_input_with_latent(model_inputs, what_latent="self", taskID=batch_hlcids)
        else:
            cur_input = model_inputs

        outputs, _ = model(cur_input, taskID=batch_hlcids, what_latent="self")
        outputs = torch.stack(outputs, dim=1)

        loss = criterion(outputs, inputs) if config.predict_first_frame else criterion(outputs, inputs[:, 1:, :])

        if before_optim_loss is None:
            before_optim_loss = loss.detach().cpu().numpy()

        loss_scalar = _mask_loss(loss, config)
        loss_scalar = loss_scalar.sum() if config.loss_reduction_LU == "sum" else loss_scalar.mean()
        loss_scalar.backward()

        model.adjust_Z_grads(config.latent_aggregation_op)
        model.Z_optimizer.step()

        if logger is not None:
            if hasattr(logger, "log_updating_loss"):
                 logger.log_updating_loss(loss.detach().cpu().numpy())
            if hasattr(logger, "log_updating_latent"):
                logger.log_updating_latent(model.Z.detach().cpu().numpy())
            if hasattr(logger, "log_updating_output"):
                logger.log_updating_output(outputs.detach().cpu().numpy())

    return before_optim_loss

# ──────────────────────────────────────────────────────────────────────────────
# Batch logging
# ──────────────────────────────────────────────────────────────────────────────

def _log_batch(logger, config, inputs, outputs, full_loss, first_full_loss,
               model, combined_input, context_ids, hlcids, hidden_states, bi):
    """Record all per-batch quantities to the logger."""
    stride = config.stride

    # Initial burn-in: log every timestep in the first batch
    if bi == 0 and config.log_initial_burn_in_timesteps:
        for i in range(config.seq_len):
            logger.log_input(np.expand_dims(inputs[:, i, :].cpu().detach().numpy(), axis=0))
            logger.log_predicted_output(np.expand_dims(outputs.cpu().detach().numpy()[:, i, :], axis=0))
    else:
        logger.log_input(inputs[:, -stride:, :].cpu().detach().numpy())
        logger.log_predicted_output(outputs.cpu().detach().numpy()[:, -stride:, :])

    logger.log_training_batch(combined_input.cpu().detach().numpy()[:, -stride:, :])
    logger.log_training_loss(full_loss[:, -stride:, :])
    logger.context_ids.append(context_ids[:, -stride:].cpu().detach().numpy())
    logger.hlcids.append(hlcids[:, -stride:].cpu().detach().numpy())

    if config.log_hidden_states:
        logger.log_hidden_states(hidden_states)

    if config.log_weights:
        logger.log_weights(model)

    if model.Z.grad is not None:
        logger.log_gradients_corrections(model.Z.grad.clone()[:, -stride:, :].cpu().detach().numpy())

    if first_full_loss is not None:
        logger.log_training_loss_before_latent_optimization(first_full_loss[:, -stride:, :])

    logger.log_latent_value(model.Z.clone()[:, -stride:, :].cpu().detach().numpy())

    _log_effective_lr(model, config, logger)

    if config.use_input_attention:
        logger.input_attention_weights.append(model.input_attention_weights.clone().cpu().detach().numpy())


# ──────────────────────────────────────────────────────────────────────────────
# Core training loop
# ──────────────────────────────────────────────────────────────────────────────

def predictive_learning(logger, config, dataloader, model,
                         criterion=nn.MSELoss(reduction='none'), epochs=None):
    """
    Run the predictive learning algorithm for `epochs` passes over `dataloader`.

    Each batch:
        1. Weight Update (WU): BPTT on all network weights.
        2. Latent Update (LU): gradient descent on Z only.
        3. Log everything to `logger`.

    Phase control:
        config._allow_latent_updates — set False during the passive learning phase.
        config.no_of_steps_in_weight_space — set 0 to freeze weights (test phase).
        config.no_of_steps_in_latent_space — set 0 to disable LU entirely.
    """
    # epochs = config.epochs if epochs is None else epochs # I only use 1 epoch

    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for bi, batch in pbar:
        inputs, batch_context_ids, batch_hlcids = batch
        batch_hlcids = batch_hlcids.to(config.device)

        # ── 1. Prepare inputs ─────────────────────────────────────────
        combined_input, core_inputs, inputs, context_ids = _prepare_batch_inputs(
            model, config, inputs, batch_context_ids
        )

        if config.update_latent_before_weights:
            first_full_loss = _latent_update_step(
                model, config, core_inputs, inputs, criterion, logger, context_ids
            )
        # ── 2. Weight Update (WU) ─────────────────────────────────────
        outputs, full_loss, hidden_states = _weight_update_step(
            model, config, combined_input, core_inputs, inputs, context_ids, criterion
        )

        # Log weight gradient norms for diagnostic use
        weight_grad_norms = [
            torch.norm(p.grad).item()
            for name, p in model.named_parameters()
            if p.grad is not None and name in (
                'input_layer.weight', 'input_layer.bias',
                'rnn_cell.weight_ih', 'rnn_cell.weight_hh',
                'rnn_cell.bias_ih', 'rnn_cell.bias_hh',
                'output_layer.weight', 'output_layer.bias',
            )
        ]
        logger.others['grad_norms'].append(np.mean(weight_grad_norms) if weight_grad_norms else 0.0)

        # ── 3. Latent Update (LU) ─────────────────────────────────────
        if not config.update_latent_before_weights:
            first_full_loss = _latent_update_step(
                model, config, core_inputs, inputs, criterion, logger, batch_hlcids
            )
        # ── 4. Log ────────────────────────────────────────────────────
        _log_batch(logger, config, inputs, outputs, full_loss, first_full_loss,
                    model, combined_input, context_ids, batch_hlcids, hidden_states, bi)

        running_loss += full_loss.sum()

        # Optional: snapshot Z space at regular intervals
        if config.eval_z_space_interval and (bi % config.eval_z_space_interval == 0):
            ezsi = config.eval_z_space_interval
            config.eval_z_space_interval = 0
            zlogger, _ = eval_z_space(model, config, bi)
            config.eval_z_space_interval = ezsi
            folder = f'{config.export_path}{config.dataset_name}/z_space_evals/'
            os.makedirs(folder, exist_ok=True)
            np.save(folder + f'zlogger_{bi}.pkl', zlogger)

# ──────────────────────────────────────────────────────────────────────────────
# Full training pipeline
# ──────────────────────────────────────────────────────────────────────────────

def train_model(config, seed=0, save_models=True, load_models=False, run_test_phase=True):
    """
    Orchestrate the full training pipeline across three phases:

        Phase 1 (optional): Passive learning — WU only, no LU.
        Phase 2: Active learning — both WU and LU, blocked context presentation.
        Phase 3 (optional): Test — weights frozen, LU runs to assess adaptation speed;
                            then a second pass with LU also disabled (pure feedforward baseline).

    Returns: (logger_train, model, config, figs)
    """
    figs = {}
    config.env_seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(config.export_path, exist_ok=True)

    logger_train = Logger()
    logger_train.config = config
    criterion = nn.MSELoss(reduction='none')

    model_folder = os.path.join(config.export_folder, 'models')
    model_name = (f'model_seed_{seed}_blocked_{config.blocked_phase_length}'
                  f'_exp_{config.experiment_to_run}.pt')
    model_path = os.path.join(model_folder, model_name)

    if load_models and os.path.exists(model_path):
        model = torch.load(model_path)
        print('Model loaded successfully.')
    else:
        model = RNN_with_latent(config).to(config.device)

        # ── Phase 1: Passive learning (WU only) ────────────────────────────
        config._allow_latent_updates = False
        if config.add_passive_learning_phase:
            logger_train.log_phase('no inference learning')
            config.no_of_blocks = int(config.passive_phase_length / config.block_size)
            _, _, dataloader, _ = create_datasets_and_loaders(config)
            print('Phase 1: Passive learning (weight update only, no latent update)')
            predictive_learning(logger_train, config, dataloader, model, criterion,
                                 epochs=getattr(config, 'passive_epochs', 1))
        logger_train.others['timestep_passive_learning_ended'] = len(logger_train.inputs)

        # ── Phase 2: Active learning (WU + LU) ─────────────────────────────
        config._allow_latent_updates = True
        config.no_of_blocks = int(config.blocked_phase_length / config.block_size)
        _, _, dataloader, _ = create_datasets_and_loaders(config)
        logger_train.log_phase('Learning and inference')
        print('Phase 2: Active learning (weight + latent updates)')
        predictive_learning(logger_train, config, dataloader, model, criterion)
        logger_train.others['timestep_learning_ended'] = len(logger_train.inputs)

        # ── Phase 3: Test — inference only ─────────────────────────────────
        if run_test_phase:
            config.reconfigure_for_prediction(config.experiment_to_run)
            _, _, _, dataloader_test = create_datasets_and_loaders(config)

            # 3a. Latent updates allowed, weights frozen
            logger_train.log_phase('Inference only')
            weight_state = 'weights frozen' if config.no_of_steps_in_weight_space == 0 else 'weights plastic'
            print(f'Phase 3a: Inference with latent adaptation ({weight_state})')
            predictive_learning(logger_train, config, dataloader_test, model, criterion)

            # 3b. No updates at all — pure feedforward baseline
            logger_train.log_phase('No inference nor learning')
            config.no_of_steps_in_latent_space = 0
            config.no_of_steps_in_weight_space = 0
            print('Phase 3b: Feedforward baseline (no weight or latent updates)')
            predictive_learning(logger_train, config, dataloader_test, model, criterion)

    if config.log_end_weights:
        if config.use_mul_gating:
            logger_train.others['P'] = model.P.detach().cpu().numpy()
        logger_train.others['input_layer_weights'] = model.input_layer.weight.detach().cpu().numpy()

    if save_models:
        os.makedirs(model_folder, exist_ok=True)
        torch.save(model, model_path)

    return logger_train, model, config, figs


# ──────────────────────────────────────────────────────────────────────────────
# Generalisation tests
# ──────────────────────────────────────────────────────────────────────────────

def run_generalized_tests(model, config, test_type='ood_means', weights_frozen=False):
    """
    Run inference with a sweep of parameter values (means, stds, block sizes) to test generalisation.

    test_type options: 'ood_means', 'training_means', 'ood_stds', 'block_size', 'input_sweep'
    Returns a dict mapping each tested value to its Logger.
    """
    testing_loggers_dict = {}
    criterion = nn.MSELoss(reduction='none')

    if test_type == 'ood_means':
        values_to_test = np.round(np.arange(-0.2, 1.3, 0.1), 1)
    elif test_type == 'training_means':
        values_to_test = config.training_data_means
    elif test_type == 'ood_stds':
        values_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    elif test_type == 'block_size':
        values_to_test = [10, 12, 13, 20, 30, 40, 50, 60, 70, 80]
    elif test_type == 'input_sweep':
        values_to_test = np.round(np.arange(-0.2, 1.3, 0.1), 1)
    else:
        raise ValueError(f"Unknown test_type '{test_type}'.")

    no_of_blocks = 2
    for value in values_to_test:
        logger = Logger()
        model_copy = copy.deepcopy(model)

        if test_type in ('ood_means', 'training_means'):
            saved_means = config.training_data_means
            config.training_data_means = [value]
        elif test_type == 'ood_stds':
            config.default_std = value
            no_of_blocks = 10
        elif test_type == 'block_size':
            config.block_size = value
            baseline = 40
            default_n = 20
            if value < baseline:
                no_of_blocks = int(default_n * (1 + 0.5 * ((baseline / value) - 1)))
            else:
                no_of_blocks = default_n
            config.block_duration_distribution = 'fixed_block_size'

        dataset = TaskDataset(config, no_of_blocks=no_of_blocks)

        if test_type in ('ood_means', 'block_size', 'ood_stds'):
            pad_length = config.seq_len + config.pre_window
            first_mean = dataset.llcid_sequence[0]
            other_means = [m for m in config.training_data_means if m != first_mean]
            pad_mean = np.random.choice(other_means) if other_means else 0.505
            dataset.llcid_sequence = [pad_mean] * pad_length + dataset.llcid_sequence
            dataset.hlcid_sequence = (dataset.hlcid_sequence[:pad_length]
                                       + dataset.hlcid_sequence)
            while len(dataset.hlcid_sequence) < len(dataset.llcid_sequence):
                dataset.hlcid_sequence.append(dataset.hlcid_sequence[-1])
            dataset.block_sizes.insert(0, pad_length)
            dataset.data_sequence, dataset.llcid_sequence, dataset.hlcid_sequence = dataset.generate_sequences()

        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        model_copy.config.no_of_steps_in_weight_space = 0 if weights_frozen else 1
        config.no_of_steps_in_weight_space = 0 if weights_frozen else 1
        predictive_learning(logger, config, dataloader, model_copy, criterion)

        logger.config = config
        testing_loggers_dict[value] = logger

        if test_type in ('ood_means', 'training_means'):
            config.training_data_means = saved_means

    return testing_loggers_dict


# ──────────────────────────────────────────────────────────────────────────────
# Z-space snapshot
# ──────────────────────────────────────────────────────────────────────────────

def eval_z_space(model_org, config_org, bi):
    """
    Freeze the model at the current training step and run a short inference pass
    to observe what representations appear in Z. Used to visualise how the latent
    space evolves during training.
    """
    model = copy.deepcopy(model_org)
    config = copy.deepcopy(config_org)

    if config.dataset_name == 'contextual_switching_task':
        dataset = TaskDataset(config, no_of_blocks=20)
    elif config.dataset_name == 'contextual_switching_task_2D':
        dataset = TaskDataset2D(config, no_of_blocks=20)
    else:
        raise ValueError(f"eval_z_space not supported for dataset '{config.dataset_name}'.")

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    config.no_of_steps_in_weight_space = 0
    config.no_of_steps_in_latent_space = 5
    config.pass_previous_latent = False
    config.seq_len = 4
    model.config = config

    logger = Logger()
    predictive_learning(logger, config, dataloader, model, nn.MSELoss(reduction='none'), epochs=1)
    return logger, dataset
