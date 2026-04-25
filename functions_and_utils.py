import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
import matplotlib.transforms as mtransforms
import pandas as pd
from collections import defaultdict
from models import RNN_with_latent

import plot_style
plot_style.set_plot_style()

def explore_data_container(data):
    """
    Explores a nested data container (list, numpy array, or PyTorch tensor).
    Args:
        data: The input data container.
    Returns:
        None
    """
    def print_info(layer, depth):
        if isinstance(layer, list):
            print(f"Layer {depth}: List, Length = {len(layer)}")
            for item in layer:
                print_info(item, depth + 1)
                break  # Only print the first item
        elif isinstance(layer, np.ndarray):
            print(f"Layer {depth}: Numpy Array, Shape = {layer.shape}")
        elif isinstance(layer, torch.Tensor):
            print(f"Layer {depth}: PyTorch Tensor, Shape = {tuple(layer.shape)}")
            return  # Stop exploring when a tensor is encountered
        elif isinstance(layer, dict):
            print(f"Layer {depth}: Dictionary, keys: {list(layer.keys())}")
            for key in layer.keys():
                print_info(layer[key], depth + 1)
        else:
            print(f"Layer {depth}: Unknown type")
    print_info(data, depth=0)

def stats(var, var_name=None):
    if type(var) == type([]): # if a list
        var = np.array(var)
    elif type(var) == type(np.array([])):
        pass #if already a numpy array, just keep going.
    else: #assume torch tensor
        pass
        # var = var.detach().cpu().numpy()
    if var_name:
        print(var_name, ':')   
    out = ('Mean, {:2.5f}, var {:2.5f}, min {:2.3f}, max {:2.3f}, norm {}'.format(var.mean(), var.var(), var.min(), var.max(),np.linalg.norm(var) ))
    print(out)
    return (out)
#%%
import matplotlib.pyplot as plt
import numpy as np

def plot_logger_panels(logger, config, panel_order, x1=0,x2=None, dpi=100, subplot_height=1.4, width=3, annotate_phases=None, rasterize=False):
    # Panel layout, adjust based on panel_order length
    fig, axes = plt.subplot_mosaic(
        [[panel] for panel in panel_order], 
        sharex=False, sharey=False,
        constrained_layout=False,
        figsize=[width, (subplot_height * len(panel_order))],  # Adjust figure size dynamically
        dpi=dpi
    )

    # Label axes
    for idx, (label, ax) in enumerate(axes.items(), start=65):  # ASCII value of 'A' is 65
        trans = mtransforms.ScaledTranslation(-27/72, 0/72, fig.dpi_scale_trans)
        ax.text(-0.02, 1.0, chr(idx), transform=ax.transAxes + trans,
                fontsize=12, va='bottom', fontfamily='sans-serif', )


    # Concatenate logger data
    ii = np.concatenate(logger.inputs, axis=0)
    ii = ii.reshape(-1, ii.shape[-1])
    li = np.concatenate(logger.latent_values, axis=0)
    li = li.reshape(-1, li.shape[-1])
    
    if logger.predicted_outputs:
        oi = np.concatenate(logger.predicted_outputs, axis=0)
        oi = oi.reshape(-1, oi.shape[-1])
    ll = np.concatenate(logger.llcids, axis=0)
    hh = np.concatenate(logger.hlcids, axis=0)

    if x2 is None:
        x1, x2 = x1, ii.shape[0]
    else:
        x1, x2 = x1, x2

    # Helper functions to plot specific panels
    def plot_behavior(ax):
        if ii.shape[-1] > 1:
            im = ax.imshow(ii[x1:x2, ].T, aspect='auto', cmap='viridis', interpolation='none')
            if rasterize:
                im.set_rasterized(True)
        else:
            line1 = ax.plot(ii[x1:x2, ], '.', alpha=0.6, markersize=3, linewidth=1, color=obs_color, label='Observed')
            if logger.predicted_outputs:
                line2 = ax.plot(oi[x1:x2, ], '.', alpha=0.7, markersize=3, linewidth=1, label='Predicted', color=preds_color)
            if rasterize:
                for line in line1 + line2:
                    line.set_rasterized(True)
            # legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
            legend = ax.legend(loc='upper center',bbox_to_anchor=(.4, 1.16))
            for lh in legend.legend_handles:
                lh.set_alpha(1.0)
        ax.set_ylabel('Observed value')

    def plot_latent(ax, force_2d=False, chunk_no=None):
        if li.shape[1] < 2 or force_2d:
            ax.plot(li[x1:x2, :2], alpha=0.9, linewidth=1)
            ax.plot(li[x1:x2, 2:], alpha=0.8, linewidth=1)
            ax.set_ylabel('Z')
        else:
            if chunk_no is None:
                im = ax.imshow(li[x1:x2, ].T, aspect='auto', cmap='viridis', interpolation='none', rasterized=True)
                ax.set_ylabel('Latent (Z)')
            else:
                total_chunks = config.latent_chunks
                chunk_size = li.shape[-1] // total_chunks
                chunk_start = chunk_no * chunk_size
                chunk_end = (chunk_no + 1) * chunk_size
                im = ax.imshow(li[x1:x2, chunk_start:chunk_end].T, aspect='auto', cmap='viridis', interpolation='none', rasterized=True)
                ax.set_ylabel(f'Z{chunk_no+1}')
            # add a color bar, but outside of the main plot, do not change the subplot size
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    def plot_loss(ax):
        if logger.prediction_losses:
            prediction_losses = np.concatenate(logger.prediction_losses, axis=0)
        else:
            prediction_losses = np.concatenate(logger.training_losses, axis=0)
        prediction_losses = prediction_losses.reshape(-1, prediction_losses.shape[-1])
        pl = prediction_losses.mean(axis=-1)[x1:x2]
        line1 = ax.plot(pl, linewidth=0.5, alpha=0.9, color='grey')
        line2 = ax.plot(np.convolve(pl, np.ones(10)/10, mode='full'), linewidth=1, color='black')
        if rasterize:
            for line in line1 + line2:
                line.set_rasterized(True)
        ax.set_ylabel('Prediction Loss')
        ax.set_xlabel('Time step')

    def plot_gradients(ax):
        gradients = np.stack(logger.gradients_corrections).squeeze()
        if gradients.shape[1] > 1: # stride is more than one
            gradients = gradients.reshape(-1, gradients.shape[-1])
        line = ax.plot(gradients[x1:x2], label='Corrections', alpha=0.9, linewidth=1)
        if rasterize:
            for l in line:
                l.set_rasterized(True)
        ax.set_ylabel('$\partial \epsilon/\partial Z$')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    def limit_y_values(ax, y_values):
        y_max = np.percentile(y_values, 95)
        ax.set_ylim(top=y_max)

    ll = ll.reshape(-1, 1)
    hh = hh.reshape(-1,1)
    unique_ll = np.unique(ll)
    ll_cmap = plt.get_cmap('Paired', 1+len(np.unique(ll)))
    unique_hh = np.unique(hh)
    hh_cmap = plt.get_cmap('viridis', len(unique_hh))

    def plot_task_illustration(ax):
        if config.dataset_name in ['contextual_switching_task', 'contextual_switching_task_hierarchical']:
            first_block_start_ts = (config.seq_len - config.stride ) % config.block_size
            most_recent_hh_value = 0
            for i in range(first_block_start_ts+1, x2, config.block_size,):
                if i < ll.shape[0]:
                    color = ll_cmap(np.where(unique_ll == ll[i][0])[0][0])
                    line1 = ax.axvline(i, 0.0, 0.4, alpha=0.6, color=color, linewidth=3)
                    if hh[i] != most_recent_hh_value:
                        hlcolor = hh_cmap(np.where(unique_hh == hh[i][0])[0][0])
                        line2 = ax.axvline(i, .6, 0.9, color=hlcolor, linestyle='-', alpha=0.6, linewidth=3)
                        most_recent_hh_value = hh[i]
                    if rasterize:
                        line1.set_rasterized(True)
                        line2.set_rasterized(True)
        else:
            scatter1 = ax.scatter(range(x1, x2), 0.2 * np.ones(x2-x1), c=hh[x1:x2], cmap=hh_cmap, s=3)
            scatter2 = ax.scatter(range(x1, x2), np.zeros(x2-x1), c=ll[x1:x2], cmap=ll_cmap, s=3)
            if rasterize:
                scatter1.set_rasterized(True)
                scatter2.set_rasterized(True)
        ax.set_axis_off()
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0, pos.width, pos.height * 0.3]
        ax.set_position(new_pos)
        ax.set_ylim(-0.1, 0.5)

    def plot_weights_grad_norm(ax, plot_smoothed=True):
        weights = np.stack(logger.others['grad_norms']).squeeze()
        x_weights = list(range(0, len(weights)*config.stride, config.stride))
        line = ax.plot(x_weights, weights, label='Weights grad norm', alpha=0.9, linewidth=1)
        
        # Apply causal smoothing using exponential moving average
        smoothed_weights = np.zeros_like(weights)
        smoothed_weights[0] = weights[0]
        alpha = 0.1  # Smoothing factor (lower = more smoothing)
        for i in range(1, len(weights)):
            smoothed_weights[i] = alpha * weights[i] + (1 - alpha) * smoothed_weights[i-1]
        if plot_smoothed:    
            smoothed_line = ax.plot(x_weights, smoothed_weights, label='Smoothed weights grad norm', alpha=0.9, linewidth=2, color='red')
        
        if rasterize:
            for l in line + smoothed_line:
                l.set_rasterized(True)
        ax.set_ylabel('$\|\partial \epsilon/\partial W\|$')
        if plot_smoothed:
            ax.legend()

    def plot_effective_lr(ax, plot_smoothed=True):
        # Plot the logged effective learning rate for the first latent dimension
        eff_list = logger.others.get('latent_effective_lr', [])
        if len(eff_list) == 0:
            ax.text(0.5, 0.5, 'No effective LR logged', ha='center', va='center')
            ax.set_ylabel('Effective LR')
            return
        # eff_list is a list of arrays with shape (batch, stride, 1). Concatenate into a time series.
        try:
            effs = np.concatenate(eff_list, axis=0)  # (n_samples, stride, 1)
            effs = effs.reshape(-1, effs.shape[-1])  # (timepoints, 1)
            series = effs[:, 0]
        except Exception:
            # if shapes unexpected, try flattening whatever is present
            series = np.concatenate([np.ravel(x) for x in eff_list])

        x_eff = np.arange(len(series))
        line = ax.plot(x_eff, series, label='Latent eff LR (dim0)', alpha=0.9, linewidth=1)
        ax.set_ylabel('Latent effective LR')
        ax.set_ylim([0, 3000])

    def plot_corrects(ax):
        if config.dataset_name == 'seq_learn':
            corrects, states, transitions, A_starts, B_starts, both_starts = get_corrects_and_trial_starts(logger)
            corrects_by_transition = {k:[] for k in transitions.keys()}
            for k, v in transitions.items():
                corrects_by_transition[k] = corrects[both_starts+v]

            focused_corrects = np.ones_like(corrects)*0.5
            focused_corrects[both_starts+transitions['T5/6']] = corrects[both_starts+transitions['T5/6']]
            focused_corrects[both_starts+transitions['T1/2']] = corrects[both_starts+transitions['T1/2']]
            alpha = 0.5
            running_avg_corrects = [0]
            for idx, val in enumerate(focused_corrects):
                if val == 0.5: # use the 0.5 value as a marker to skip the timesteps where the correctness is not of interest
                    running_avg_corrects.append(running_avg_corrects[-1])
                else:
                    updated_val = (1-alpha) * running_avg_corrects[-1] + alpha * val
                    running_avg_corrects.append(updated_val)
            running_avg_corrects.pop(0)

            ax.plot(running_avg_corrects, label='Running Average', color='k', alpha=0.9, linewidth=0.75)

            ax.legend()
            ax.set_ylabel('Correct')
            ax.set_xlabel('Time Step')
            ax.set_ylim(0, 1.1)

        elif logger.predicted_outputs and config.dataset_name in ['contextual_switching_task', 'contextual_switching_task_2D']:
            # abs distance of predictions from the underlying block mean (llcids store the true mean)
            abs_dist = np.abs(oi[x1:x2] - ll[x1:x2])
            smoothed = np.convolve(abs_dist[:, 0], np.ones(20) / 20, mode='same')
            # ax.plot(abs_dist[:, 0], '.', alpha=0.3, markersize=2, color='grey')
            ax.plot(smoothed, linewidth=1, color='k', label='|pred − mean|')
            ax.set_ylabel('|pred − mean|')

        else:
            print('No corrects to plot for this dataset')

    def label_phases(ax, subplot_height):
        for i, (phase_name, phase_start) in enumerate(logger.phases):
            local_start = phase_start - x1
            if local_start + 10 > x2 - x1:
                break
            if i < len(logger.phases) - 1:
                local_end = logger.phases[i + 1][1] - x1
            else:
                local_end = x2 - x1
            local_end = min(local_end, x2 - x1)
            phase_midpoint = (max(local_start, 0) + local_end) / 2
            phase_name = '\n'.join(phase_name.split())

            if 0 <= local_start <= x2 - x1:
                ax.axvline(local_start, color='tab:green', linestyle='-', linewidth=3, alpha=0.7)
            y_factor = 1.05 if subplot_height == 4 else 1.8
            ax.text(phase_midpoint, ax.get_ylim()[1] * y_factor, phase_name, rotation=0, verticalalignment='top', color='tab:green',
             fontsize=6, fontweight='bold', ha='center')
    def plot_latent_2d(ax):
        plot_latent(ax, force_2d=True)
    def plot_latent_chunk_1(ax):
        plot_latent(ax, chunk_no=0)
    def plot_latent_chunk_2(ax):
        plot_latent(ax, chunk_no=1)

    # Dictionary mapping panel names to functions
    panel_functions = {
        'behavior': plot_behavior,
        'latent': plot_latent,
        'latent_effective_lr': plot_effective_lr,
        'latent_2d': plot_latent_2d,
        'latent_2D': plot_latent_2d,
        'latent2D': plot_latent_2d,
        'latent_chunk_1': plot_latent_chunk_1,
        'latent_chunk_2': plot_latent_chunk_2,
        'loss': plot_loss,
        'gradients': plot_gradients,
        'task_illustration_and_hierarchies': plot_task_illustration,
        'weights_grad_norm': plot_weights_grad_norm,
        'corrects': plot_corrects,
    }

    # Execute the plot function for each panel in the order provided
    for panel in panel_order:
        if panel in panel_functions:
            panel_functions[panel](axes[panel])

    for panel in panel_order:
        ax = axes[panel]
        ax.set_xlim(0, x2 - x1)
        if config.dataset_name == 'seq_learn':
            # plot_switches(ax, states, both_starts)
            plot_switches_from_logger(ax, logger, config, use_ll=False)
        elif panel not in  ['task_illustration_and_hierarchies', 'latent', 'latent_chunk_1', 'latent_chunk_2']:
            plot_switches_from_logger(ax, logger, config)
        
        # annotate_training_phases(ax, config, logger=logger, add_text=True if ax==axes['A'] else False)
        # remove xtick labels except for bottom panel
        if ax != list(axes.values())[-1]:
            ax.set_xticklabels('')
    list(axes.values())[-1].set_xlabel('Time steps')
    # label_phases(list(axes.values())[-1])
    if annotate_phases is not None:
        ax=axes[annotate_phases]
        label_phases(ax, subplot_height)

    return fig


def extract_block_corrects(logger, last_ts_in_a_block=15,
                           phases_to_include='Learning and inference'):
    """Per-block mean |pred − ground-truth mean| using only the tail of each block.

    Returns (block_means, block_sems) as np.ndarrays of shape (n_blocks,),
    or (None, None) if no predicted_outputs are logged.
    """
    if not logger.predicted_outputs:
        return None, None

    ll_full = np.concatenate(logger.llcids, axis=0).reshape(-1, 1)
    oi_full = np.concatenate(logger.predicted_outputs, axis=0)
    oi_full = oi_full.reshape(-1, oi_full.shape[-1])

    if phases_to_include is None:
        ll, oi = ll_full, oi_full
    else:
        include = [phases_to_include] if isinstance(phases_to_include, str) else phases_to_include
        phase_windows = []
        for i, (name, ts) in enumerate(logger.phases):
            end = logger.phases[i + 1][1] if i + 1 < len(logger.phases) else len(ll_full)
            if name in include:
                phase_windows.append((ts, end))
        if phase_windows:
            ll = np.concatenate([ll_full[s:e] for s, e in phase_windows], axis=0)
            oi = np.concatenate([oi_full[s:e] for s, e in phase_windows], axis=0)
        else:
            ll, oi = ll_full, oi_full

    abs_dist = np.abs(oi[:, 0] - ll[:, 0])
    switches = [ts for ts in range(1, len(ll)) if ll[ts] != ll[ts - 1]]
    block_boundaries = list(zip([0] + switches, switches + [len(ll)]))

    block_means, block_sems = [], []
    for start, end in block_boundaries:
        tail_start = max(start, end - last_ts_in_a_block)
        tail = abs_dist[tail_start:end]
        if len(tail) > 0:
            block_means.append(tail.mean())
            block_sems.append(tail.std() / np.sqrt(len(tail)))
    return np.array(block_means), np.array(block_sems)


def plot_logger_analysis(logger, config, subplot_width=1.5, subplot_height=1.5, dpi=100,
                         last_ts_in_a_block=15, aggregate_blocks=None,
                         phases_to_include='Learning and inference'):
    """
    2×2 summary analysis of a training run (not time-series).

    phases_to_include : str or list of str, phase name(s) from logger.phases to include.
                        Defaults to 'Learning and inference'. Pass None to use all data.
    aggregate_blocks  : int, number of blocks to average into one data point.
                        E.g. no_of_blocks // 3 gives early/mid/late.
    """
    fig, axes = plt.subplot_mosaic(
        [['A', 'B'], ['C', 'D']],
        sharex=False, sharey=False,
        constrained_layout=True,
        figsize=(subplot_width * 2, subplot_height * 2),
        dpi=dpi,
    )

    for idx, (label, ax) in enumerate(axes.items(), start=65):
        trans = mtransforms.ScaledTranslation(-27/72, 0/72, fig.dpi_scale_trans)
        ax.text(-0.02, 1.0, chr(idx), transform=ax.transAxes + trans,
                fontsize=12, va='bottom', fontfamily='sans-serif')

    # ── Panel A: |pred − mean| per block (tail only), optionally aggregated ──
    def plot_block_corrects(ax):
        block_means, block_sems = extract_block_corrects(
            logger, last_ts_in_a_block=last_ts_in_a_block,
            phases_to_include=phases_to_include,
        )
        if block_means is None:
            ax.set_axis_off()
            return

        n_agg = int(aggregate_blocks) if aggregate_blocks is not None else 1
        if n_agg > 1:
            n_groups = len(block_means) // n_agg
            x_vals, grp_means, grp_sems = [], [], []
            for g in range(n_groups):
                grp = block_means[g * n_agg:(g + 1) * n_agg]
                x_vals.append(g + 0.5)
                grp_means.append(grp.mean())
                grp_sems.append(grp.std() / np.sqrt(len(grp)))
            ax.errorbar(x_vals, grp_means, yerr=grp_sems,
                        fmt='-o', capsize=3, linewidth=1, markersize=4)
            ax.set_xlabel(f'Block group (×{n_agg})')
            ax.set_xticks(x_vals)
            ax.set_xticklabels([f'{g*n_agg}–{(g+1)*n_agg}' for g in range(n_groups)], fontsize=6, rotation=30)
        else:
            ax.errorbar(range(len(block_means)), block_means, yerr=block_sems,
                        fmt='-o', capsize=2, markersize=3, linewidth=1, alpha=0.8)
            ax.set_xlabel('Block')

        phase_label = phases_to_include if isinstance(phases_to_include, str) else ', '.join(phases_to_include or [])
        ax.set_title(f'|pred − mean| tail={last_ts_in_a_block}\n{phase_label}', fontsize=7)
        ax.set_ylabel('|pred − mean|')

    plot_block_corrects(axes['A'])

    for key in ['B', 'C', 'D']:
        axes[key].set_axis_off()

    return fig


def plot_behavior_panel(ax, logger, config, x2=None):
    ii = np.concatenate(logger.inputs, axis=0)
    ii = ii.reshape(-1, ii.shape[-1])
    if x2 is None:
        x1, x2 = 0, ii.shape[0]
    else:
        x1, x2 = 0, x2

    if ii.shape[-1] > 1:
        ax.imshow(ii[x1:x2, ].T, aspect='auto', cmap='viridis', interpolation='none')
    else:
        ax.plot(ii[x1:x2, ], '.', alpha=0.7, markersize=3, linewidth=1, color='tab:grey', label='Observed')
        if logger.predicted_outputs:
            oi = np.concatenate(logger.predicted_outputs, axis=0)
            oi = oi.reshape(-1, oi.shape[-1])
            ax.plot(oi[x1:x2, ], '.', alpha=0.7, markersize=3, linewidth=1, label='Predicted', color='tab:red')
            ax.legend(loc='lower center')
    ax.set_ylabel('Data dim')
    ax.set_xlabel('Time steps')
    ax.set_xlim(x1, x2)
    plot_switches_from_logger(ax, logger, config)

def plot_task_and_hierarchies_illustration(logger,  config, x2=None, show_output=False):
        
    obs_color = 'tab:grey'
    preds_color = 'tab:red'

    fig, axes = plt.subplot_mosaic([['A'], ['B'], ['B']], sharex=True,
                                    constrained_layout=False, figsize = [16/2.53, 4.5/2.53], dpi=300)
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        # ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
        #     fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    # merge the batches into one sequence
    ci = np.concatenate(logger.training_batches, axis=0)
    ci = ci.reshape(-1, ci.shape[-1])
    #li = ci[:, -config.latent_dims[0]:]  # latent
    li = np.concatenate(logger.latent_values, axis=0)
    li = li.reshape(-1, li.shape[-1])

    ii = np.concatenate(logger.inputs, axis=0)
    ii = ii.reshape(-1, ii.shape[-1])
    if logger.predicted_outputs != []:
        oi = np.concatenate(logger.predicted_outputs, axis=0)
        oi = oi.reshape(-1, oi.shape[-1])
    ll = np.concatenate(logger.llcids, axis=0)
    ll = ll.reshape(-1, 1)
    hh = np.concatenate(logger.hlcids, axis=0)
    hh = hh.reshape(-1,1)


    unique_ll = np.unique(ll)
    ll_cmap = plt.get_cmap('Set1', int(1.0 / 0.1) + 1)
    ll_colors = ll_cmap(np.arange(0, 1.1, 0.1))

    unique_hh = np.unique(hh)
    hh_cmap = plt.get_cmap('viridis', len(unique_hh))
    # hh_cmap = plt.get_cmap('winter', len(unique_hh))

    # x1, x2 = 0, min(5000, ci.shape[0])
    if x2 is None:
        x1, x2 = 0, ii.shape[0]
    else:
        x1, x2 = 0, x2
    
    ax = axes['B'] 
    if (ii.shape[-1] ) > 1: # if there are more than one features
        ax.imshow(ii[x1:x2,].T, aspect='auto', cmap='viridis', interpolation='none')
        ax.set_ylabel('Feature')
    else: # if input is 1D
        ax.plot(ii[x1:x2,], 'o',  markersize =0.5, color=obs_color)
        if show_output:
            ax.plot(oi[x1:x2,], 'o',   markersize =0.5,  label='Predicted Output', color=preds_color)
        ax.set_ylabel('Feature')
    ax.set_xlabel('Time steps')
    # shade the background alternatively using ax span for each block
    if config.dataset_name == 'contextual_switching_task':
        first_block_start_ts = (config.seq_len - config.stride ) % config.block_size
        for i in range(first_block_start_ts+1, x2, config.block_size,):
            if i < ll.shape[0]:
                ax.axvspan(i, i+config.block_size, color=ll_cmap(ll[i][0]), alpha=0.04)

        #easier and more reliable way to plot hh
    ax.set_xlim(x1, x2)
    
    ax = axes['A']
    if config.dataset_name == 'contextual_switching_task':
        first_block_start_ts = (config.seq_len - config.stride ) % config.block_size
        most_recent_hh_value = 0
        for i in range(first_block_start_ts+1, x2, config.block_size,):
            # ax.axvline(c, .6, .9 , color=color, linestyle='-', alpha=0.6, linewidth=3)
            if i < ll.shape[0]:
                color = ll_cmap(ll[i][0])
                ax.axvline(i, 0.0,0.4,  alpha=0.6, color=color, linewidth=3)
                if hh[i] != most_recent_hh_value:
                    ax.axvline(i, .6, 0.9, color=hh_cmap(hh[i][0]-1), linestyle='-', alpha=0.6, linewidth=3)
                    most_recent_hh_value = hh[i]
    else:
        ax.scatter(range(x1, x2), np.zeros(x2-x1), c=hh[x1:x2], cmap=hh_cmap, s=3)
        ax.scatter(range(x1, x2), np.ones(x2-x1), c=ll[x1:x2], cmap=ll_cmap, s=3)


    ax.set_axis_off()
    ax.set_xlim(x1, x2)
# plot_task_and_hierarchies_illustration(logger, config, x2=1500)
# plt.savefig(f'{config.export_path}_task_and_hierarchies_illustration.pdf', dpi=300, bbox_inches='tight')
# print('figure saved to: ', f'{config.export_path}_task_and_hierarchies_illustration.pdf')

def plot_seq_learn_behavior_and_overall_corrects(logger, config, include_gradients=False):
    if include_gradients:
        fig, axes = plt.subplot_mosaic([['A', ], ['B',], ['C', ] ,['D'],['E'] ], sharex=False, sharey=False,
                                    constrained_layout=False, figsize = [18/2.53, (11+5)/2.53])
    else:
        fig, axes = plt.subplot_mosaic([['A', ], ['B',], ['C', ] ], sharex=False, sharey=False,
                                    constrained_layout=False, figsize = [18/2.53, (11)/2.53])

    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        # ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            # fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    li = np.concatenate(logger.latent_values, axis=0)
    li = li.reshape(-1, li.shape[-1])

    ii = np.concatenate(logger.inputs, axis=0)
    ii = ii.reshape(-1, ii.shape[-1])

    x1 =0
    x2 =len(li)

    ax = axes['C']

    if li.shape[1] > 4:
        ax.imshow(li[x1:x2, ].T, aspect='auto', cmap='viridis', interpolation='none')
    else: # if latent  is 2D
        ax.plot(li[x1:x2,], alpha = 0.7, linewidth=1,)
    ax.set_xlim(x1, x2)
    ax.set_ylabel('Latent')

    ax = axes['A']
    # plot predicted losses
    if logger.prediction_losses != []:
        prediction_losses = np.concatenate(logger.prediction_losses, axis = 0)
        prediction_losses = prediction_losses.reshape(-1, prediction_losses.shape[-1])
        prediction_losses = prediction_losses.mean(axis=-1) # average over the output dimensions
    else:
        prediction_losses = np.concatenate(logger.training_losses, axis = 0)
        prediction_losses = prediction_losses.reshape(-1, prediction_losses.shape[-1])
        prediction_losses = prediction_losses.mean(axis=-1) # average over the output dimensions
    ax.plot(prediction_losses, linewidth=1, alpha = 0.6, color='grey', label='Loss')
    # convolve with a window = 10 to smooth the plot
    ax.plot(np.convolve(prediction_losses.squeeze(), np.ones(10)/10, mode='valid',), linewidth=1, color='black', label='Loss Smoothed')
    ax.set_ylabel('Prediction Loss')
    # ax.set_xlabel('Time step')
    ax.set_xlim(x1, x2)
    ax.legend()

    ax = axes['B']
    if config.dataset_name == 'seq_learn':
        corrects, states, transitions, A_starts, B_starts, both_starts = get_corrects_and_trial_starts(logger)
        corrects_by_transition = {k:[] for k in transitions.keys()}
        for k, v in transitions.items():
            corrects_by_transition[k] = corrects[both_starts+v]

        focused_corrects = np.ones_like(corrects)*0.5
        focused_corrects[both_starts+transitions['T5/6']] = corrects[both_starts+transitions['T5/6']]
        focused_corrects[both_starts+transitions['T1/2']] = corrects[both_starts+transitions['T1/2']]

        alpha = 0.5

        running_avg_corrects = [0]
        for idx, val in enumerate(focused_corrects):
            if val == 0.5:
                running_avg_corrects.append(running_avg_corrects[-1])
            else:
                updated_val = (1-alpha) * running_avg_corrects[-1] + alpha * val
                running_avg_corrects.append(updated_val)
        running_avg_corrects.pop(0)

        ax.plot(running_avg_corrects, label='Running Average', color='k', alpha=0.9, linewidth=0.75)

        ax.legend(fontsize=8)
        ax.set_ylabel('Correct')
        ax.set_xlabel('Time Step')
        ax.set_ylim(0, 1.1)
    else: # plot behavior instead
        ii = np.concatenate(logger.inputs, axis=0)
        ii = ii.reshape(-1, ii.shape[-1])
        oi = np.concatenate(logger.predicted_outputs, axis=0)
        oi = oi.reshape(-1, oi.shape[-1])

        if (ii.shape[-1] ) > 1: # if there are more than one features
            ax.imshow(ii[x1:x2,].T, aspect='auto', cmap='viridis', interpolation='none')
            ax.set_ylabel('Feature')
        else: # if input is 1D
            ax.plot(ii[x1:x2,], '.', alpha = 0.7, markersize =3, linewidth=1, color=obs_color)
            if logger.predicted_outputs != []:
                ax.plot(oi[x1:x2,], '.', alpha = 0.7, markersize =3, linewidth=1, label='Predicted Output', color=preds_color)
            ax.set_ylabel('Feature')

    if include_gradients:
        ax = axes['E']
        # gmaxe = np.stack(logger.gradients_max_entropy).squeeze()
        # no_of_missing_grads = len(states) - gmaxe.shape[0]
        # gmaxe = np.pad(gmaxe, (no_of_missing_grads, 0), 'constant', constant_values=(0, 0))
        # ax.plot(gmaxe, label='Max Entropy', alpha=0.9)
        # ax.set_ylabel('Gradient Max Entropy')
        dw_norm = np.stack(logger.others['grad_norms'])
        if config.stride > 1:
            dw_norm = dw_norm.reshape(-1, dw_norm.shape[-1])
        ax.plot(dw_norm, label='Grad Norms', alpha=0.9)
        ax.set_ylabel('$\delta$ Weight', )
        
        ax = axes['D']
        gcorr = np.stack(logger.gradients_corrections).squeeze()
        if config.stride > 1:
            gcorr = gcorr.reshape(-1, gcorr.shape[-1]) 
        ax.plot(gcorr, label='Corrections', alpha=0.9)
        ax.set_ylabel('Latent Gradients')

    for ax in axes.values():
        ax.set_xlim(x1, x2)
    if config.dataset_name == 'seq_learn':
            plot_switches(ax, states, both_starts)
    else:
            plot_switches_from_logger(ax, logger, config)
        
    annotate_training_phases(ax, config, logger=logger, add_text=True if ax==axes['A'] else False)
    if ax != list(axes.values())[-1]: ax.set_xticklabels('')
    list(axes.values())[-1].set_xlabel('Time steps')

    return(fig)


#%%
# seq_learn behavior
def get_corrects_and_trial_starts(logger):
    inputs = np.vstack(logger.inputs)
    # inputs has shape (no of batches, batch_size, seq_len, features)
    inputs = inputs.reshape(-1, inputs.shape[-1])
    states = np.argmax(inputs, axis=-1)
    states = states.squeeze()

    outputs = np.vstack(logger.predicted_outputs)
    outputs = outputs.reshape(-1, outputs.shape[-1])
    preds = np.argmax(outputs, axis=-1)
    preds = preds.squeeze()

    corrects = (preds == states)*1.

    transitions = {'T0':0, 'T1/2':1, 'T3/4':2, 'T5/6':3, 'T7/8':4, 'T9':5}
    A_starts = np.where((states == 1))
    B_starts = np.where((states == 2))
    both_starts = np.where((states == 0))


    # removce the last element, and remove the tuple added by np.where
    A_starts = A_starts[0][:-1]
    B_starts = B_starts[0][:-1]
    both_starts = both_starts[0][:-1]
    return corrects, states, transitions, A_starts, B_starts, both_starts


def annotate_training_phases(ax, config, logger = None, add_text=True):
    if logger is not None:
        if hasattr(logger, 'time_step_shuffle_ended'):
            time_step_shuffle_ended = logger.time_step_shuffle_ended
        else:
            time_step_shuffle_ended = 0

        if hasattr(logger, 'time_step_training_ends'):
            time_step_training_ends = logger.time_step_training_ends
        else:
            time_step_training_ends = 0
    ax.axvline(time_step_training_ends, color='r', linestyle='-', alpha= 0.4, linewidth=4)
    if time_step_shuffle_ended > 0:
        ax.axvline(time_step_shuffle_ended, color='g', linestyle='-', alpha= 0.4, linewidth=4)
    if add_text:
        axis_2_max = 0.8*(ax.get_ylim()[1])
        axis_2_fontsize = 10
        ax.text(1.02*(time_step_training_ends), axis_2_max, 'Testing', rotation=0, fontsize=axis_2_fontsize, color='r', transform=ax.transData) 
        if config.add_blocked_phase:
            ax.text(int(0.3*(time_step_training_ends-time_step_shuffle_ended)+time_step_shuffle_ended), axis_2_max, 'Blocked Training', rotation=0, fontsize=axis_2_fontsize, color='r', transform=ax.transData) 
        if time_step_shuffle_ended>0:   
            ax.text(int(0.2*(time_step_shuffle_ended)), axis_2_max, 'Interleaved Training', rotation=0, fontsize=axis_2_fontsize, color='g', transform=ax.transData)
def plot_switches(ax, states, both_starts):
    TIS_time_steps = both_starts + 1
    switches = [ts for ts in TIS_time_steps if states[ts] != states[ts-6]] + [len(states)]
    for isw, switch in enumerate(switches):
        c = 'tab:red' if isw%2==0 else 'tab:blue'
        if isw < len(switches) -1 :
            # print(switch, switches[isw+1])
            alpha = 0.1 if (switches[isw+1] - switch) > 7 else 0.03 # make it much lighter for shuffled or interleaved
            ax.axvspan(switch, switches[isw+1], color=c, alpha=alpha)
        # ax.axvline(switch, color=c, linestyle=':', alpha=0.3, linewidth=2)
        # if isw % 2 == 0 and len(switches) > isw:
        #     ax.axvspan(switch, switches[isw+1], color='b', alpha=0.1)

def plot_switches_from_logger(ax, logger, config, use_ll=True, alpha =0.1, alpha_interleaved=0.03):
    import plot_style
    cs = plot_style.Color_scheme()
    ll = np.concatenate(logger.llcids, axis=0)
    ll = ll.reshape(-1, ll.shape[-1])
    if not use_ll:
        hh = np.concatenate(logger.hlcids, axis=0)
        hh = hh.reshape(-1, hh.shape[-1])
        ll = hh # use high level instead

    unique_ll = np.unique(ll)
    # ll_cmap = plt.get_cmap('Pastel', len(unique_ll))
    ll_cmap = plt.get_cmap('Paired', 1+len(np.unique(ll)))

    # check at which time steps the task ll changes
    switches = [ts for ts in range(1, len(ll)) if ll[ts] != ll[ts-1]]
    switches = [0] + switches + [len(ll)]
    for isw, switch in enumerate(switches):
        c = cs.contextA if isw%2==0 else cs.contextB

        # cmap = plt.get_cmap('Paired')    
        # c = cmap(1) if isw % 2 == 0 else cmap(5)
         
        if isw < len(switches) -1 :
            # c = ll_cmap(ll[switch][0])
            # print(switch, switches[isw+1])
            alpha = alpha if (switches[isw+1] - switch) > 7 else alpha_interleaved # make it much lighter for shuffled or interleaved
            ax.axvspan(switch, switches[isw+1], color=c, alpha=alpha)

def plot_corrects_by_transition(logger, get_corrects_and_trial_starts):
    
    corrects, states, transitions, A_starts, B_starts, both_starts = get_corrects_and_trial_starts(logger)
    corrects_by_transition = {k:[] for k in transitions.keys()}
    for k, v in transitions.items():
        corrects_by_transition[k] = corrects[both_starts+v]

    TIS_time_steps = both_starts + 1
    switches = [ts for ts in TIS_time_steps if states[ts] != states[ts-6]]
    # swtiches = TIS_time_steps[switches]-1

    fig, axes = plt.subplots(6,1  , figsize=(7, 4))
    for i, (k, v) in enumerate(corrects_by_transition.items()):
        ax = axes[i]
        ax.plot(both_starts+v, v, label=k, color = 'C'+str(i))
        ax.set_ylabel('Correct')
        ax.set_xlabel('Time step')
        ax.legend()
        ax.set_ylim(0, 1)
        # if len(switches) < 20: # takes too long to plot if interleaved
        for switch in switches:
            ax.axvline(switch, color='b', linestyle=':', alpha=0.5)
    return fig
#%%
def plot_corrects_t12_adaptation(logger, get_corrects_and_trial_starts):
    '''
    So both_starts has the indices or timesteps at which each trial started.
    
    v is the index from the trial start to the transition type of interest.
    So transition T5/6 is the 4th timestep after the trial started. 3 = transitions['T5/6']


    '''
    corrects, states, transitions, A_starts, B_starts, both_starts = get_corrects_and_trial_starts(logger)
    corrects_by_transition = {k:[] for k in transitions.keys()}
    for k, v in transitions.items():
        corrects_by_transition[k] = corrects[both_starts+v]

    TIS_time_steps = both_starts + 1 # plus one to get the T1/2 task identifying state.
    switches = [ts for ts in TIS_time_steps if states[ts] != states[ts-6]]
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    pre_window = 5 *6 
    post_window = 15 *6
    corrects_t12 = corrects[both_starts+transitions['T1/2']]
    peri_switch_corects_t12 = []
    cmap = plt.cm.viridis
    num_switches = len(switches)
    colors = [cmap(i/num_switches) for i in range(num_switches)]
    for i, switch in enumerate(switches):
        if (switch > pre_window) and (switch < (len(corrects) - post_window)):
            x = corrects[switch-(pre_window):switch+post_window:6]
            ax.plot(x, color=colors[i], alpha=0.7, label=f'Switch {i}' if i in [1, 10, 20, 30] else None)
            peri_switch_corects_t12.append(x)
    peri_switch_corects_t12 = np.array(peri_switch_corects_t12)
    ax.plot(peri_switch_corects_t12.mean(axis=0), label='Mean Corrects T1/2', alpha=0.3, linewidth=3)    
    ax.set_ylabel('Correct')
    ax.set_xlabel('Trials around switch')
    ax.set_ylim(0, 1)
    ax.axvline(pre_window//6, color='r', linestyle=':', alpha=0.5, linewidth=4)
    ax.set_xticks([0, pre_window//6, post_window//6])
    ax.set_xticklabels([-pre_window//6, 0, post_window//6])
    # add a legend to the right outside of the plot, only include every fifth line
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    return fig


#%%

def extract_trial_starts_by_type(logger, config):
    # convert logger.inputs to integer states
 
    inputs = np.vstack(logger.inputs)
    # inputs has shape (no of batches, batch_size, seq_len, features)
    inputs = inputs.reshape(-1, inputs.shape[-1])
    states = np.argmax(inputs, axis=-1)
    states = states.squeeze()

    outputs = np.vstack(logger.predicted_outputs)
    outputs = outputs.reshape(-1, outputs.shape[-1])
    preds = np.argmax(outputs, axis=-1)
    preds = preds.squeeze()

    corrects = (preds == states)*1.

    # NOTE calling the context indicative signal (State 1 vs. 2) as the trial 'start'
    # find all the indices where the state is 1
    A_starts = np.where(states == 1)
    B_starts = np.where(states == 2)

    # print('A_starts: ', A_starts)
    # print('B_starts: ', B_starts)

    # removce the last element, and remove the tuple added by np.where
    A_starts = A_starts[0][:-1]
    B_starts = B_starts[0][:-1]

    A3_starts = [s for s in A_starts if states[s+1] == 3]
    A4_starts = [s for s in A_starts if states[s+1] == 4]
    B3_starts = [s for s in B_starts if states[s+1] == 3]
    B4_starts = [s for s in B_starts if states[s+1] == 4]

    # convert to numpy arrays
    A3_starts = np.array(A3_starts)
    A4_starts = np.array(A4_starts)
    B3_starts = np.array(B3_starts)
    B4_starts = np.array(B4_starts)

    # print('A3_starts: ', A3_starts)

    # print all the inputs that are incexed in A3_starts
    representation_A3 = [(inputs[s+1]) for s in A3_starts]
    representation_A4 = [(inputs[s+1]) for s in A4_starts]
    rep_A3 = np.array(representation_A3).squeeze()

    # print('rep_A3: ', rep_A3.shape)
    # print('rep_A3 mean: ', rep_A3.mean(axis=0))

    
    return A3_starts, A4_starts, B3_starts, B4_starts, inputs, outputs, 

def plot_corrects_seq_learn(logger, config):
    # convert logger.inputs to integer states
    corrects_fig, axes = plt.subplot_mosaic([['A','A', 'C'],['B', 'B', 'D']],
                                constrained_layout=False, figsize = [22/2.53, 6/2.53])
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, corrects_fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')


    inputs = np.vstack(logger.inputs)
    # inputs has shape (no of batches, batch_size, seq_len, features)
    inputs = inputs.reshape(-1, inputs.shape[-1])
    states = np.argmax(inputs, axis=-1)
    states = states.squeeze()

    outputs = np.vstack(logger.predicted_outputs)
    outputs = outputs.reshape(-1, outputs.shape[-1])
    preds = np.argmax(outputs, axis=-1)
    preds = preds.squeeze()

    corrects = (preds == states)*1.

    ax = axes['A']
    # ax.plot(corrects)
    # add some noise to the corrects to make it easier to see
    ax.plot(corrects + np.random.normal(0, 0.1, len(corrects)), 'o', alpha=0.5, markersize=1)
    #plot again with a moving average

    ll = np.concatenate(logger.llcids, axis=0)
    ll = ll.reshape(-1, 1)
    hh = np.concatenate(logger.hlcids, axis=0)
    hh = hh.reshape(-1,1)

    x1, x2 = 0, len(corrects)
    x1 = int(x1)
    x2 = int(x2)
    
    unique_ll = np.unique(ll)
    ll_cmap = plt.get_cmap('Set1', int(1.0 / 0.1) + 1)

    unique_hh = np.unique(hh)
    hh_cmap = plt.get_cmap('viridis', len(unique_hh))
    ax.scatter(range(x1, x2), 0.3 + np.zeros(x2-x1), c=hh[x1:x2], cmap=hh_cmap, s=3)
    ax.scatter(range(x1, x2), 0.7* np.ones(x2-x1), c=ll[x1:x2], cmap=ll_cmap, s=3)

    ax = axes['B']

    ax.plot(pd.Series(corrects).rolling(window=10).mean())

    ax.set_title('Correct predictions')
    ax.set_ylabel('Correct')
    ax.set_xlabel('Batch')
    bs = config.block_size
    ax.set_xticks(np.arange(0, len(corrects), bs))
    
    # ax.axvline(x=bs, color='r', linestyle='--', alpha=0.2)
    for switch in np.arange(0, len(corrects), bs):
        ax.axvline(x=switch, color='r', linestyle='--', alpha=0.2)
    ax.set_ylim([0,1])
    ax = axes['C']
    cmap = matplotlib.cm.get_cmap('viridis')
    window_pre, window_post = 5, 20
    switch_centered_corrects = []
    for i, t_switch in enumerate(np.arange(0, len(corrects), bs)):
        if t_switch > window_pre and t_switch < ((states.shape[-1])-window_post):
            switch_centered_corrects.append(corrects[t_switch-window_pre:t_switch+window_post])

    switch_corrects  = np.array(switch_centered_corrects)

    for i, loss in enumerate(switch_corrects):
        ax.plot(range(-window_pre, window_post),  loss, color=cmap(i/len(switch_corrects)),
                alpha=0.5, label=f'switch {i+1}')# if i in [0, 9, 19, len(switch_corrects)-1] else None)
        ax.axvline(x=0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Timesteps around switch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Switch related accuracy {""}', fontsize=8)
    # ax.legend()

    ax = axes['D']
    losses = np.vstack(logger.prediction_losses if logger.prediction_losses else logger.training_losses)
    losses = losses.reshape(-1, losses.shape[-1]).mean(axis=-1)
    switch_centered_loss = []
    for i, t_switch in enumerate(np.arange(0, len(losses), bs)):
        if t_switch > window_pre and t_switch < ((states.shape[-1])-window_post):
            switch_centered_loss.append(losses[t_switch-window_pre:t_switch+window_post])

    switch_losses  = np.array(switch_centered_loss)
    for i, loss in enumerate(switch_losses):
        ax.plot(range(-window_pre, window_post),  loss, color=cmap(i/len(switch_losses)),
                alpha=0.5, label=f'switch {i+1}')# if i in [0, 9, 19, len(switch_losses)-1] else None)
        ax.axvline(x=0, color='grey', linestyle='--', alpha=0.5)

    ax.set_xlabel('Timesteps around switch')
    ax.set_ylabel('Loss')

    return corrects_fig


class Logger:
    """
    Logger for tracking model training, inference, and latent optimization.
    
    DATA STORAGE FORMAT:
    All logged data follows the pattern: List of numpy arrays with shape (batch_dim, seq_len, var_dim)
    - batch_dim: Always 1 in current experiments
    - seq_len: Typically config.stride timesteps per logged entry (except first batch with burn-in)
    - var_dim: Dimensionality of the logged variable
    
    After concatenation (e.g., np.concatenate(logger.inputs, axis=0)), arrays become:
    Shape: (total_timesteps, var_dim) after reshaping with .reshape(-1, var_dim.shape[-1])
    
    LOGGED ATTRIBUTES:
    
    Training & Testing:
    - training_batches: List[(batch, stride, input_size)] - Combined input+latent sent to model
    - training_losses: List[(batch, stride, output_size)] - MSE loss per timestep and output dim
    - training_losses_before_latent_optimization: List[(batch, stride, output_size)] - Loss before LU step
    - testing_batches: List[(batch, stride, input_size)] - Same as training_batches for test data
    - testing_losses: List[(batch, stride, output_size)] - Same as training_losses for test data
    
    Predictions & Targets:
    - inputs: List[(batch, stride, input_size)] - Ground truth observations/targets
             First batch may have shape (batch, seq_len, input_size) if log_initial_burn_in_timesteps=True
    - predicted_outputs: List[(batch, stride, output_size)] - Model predictions
             Logged via: logger.log_predicted_output(outputs.cpu().detach().numpy()[:, -config.stride:, :])
    - prediction_losses: List[(batch, stride, output_size)] - Prediction error per output dimension
    
    Latent Variables (Z):
    - latent_values: List[(batch, stride, latent_dims)] - Z values after optimization
             Logged via: logger.log_latent_value(model.latent.clone()[:, -config.stride:, :].cpu().detach())
    - latent_gradients: List[(batch, stride, latent_dims)] - dL/dZ gradients
    - gradients_corrections: List[(batch, stride, latent_dims)] - Gradient of loss w.r.t. latents
             Logged via: logger.log_gradients_corrections(model.latent.grad.clone()[:, -config.stride:, :])
    - gradients_max_entropy: List[(batch, stride, latent_dims)] - Entropy-related gradients
    
    Latent Optimization Steps (within-batch Z updates):
    - latent_updating_losses: List[(batch, seq_len, output_size)] - Loss at each LU optimization step
    - latent_updating_latents: List[(batch, seq_len, latent_dims)] - Z values during optimization
    - latent_updating_outputs: List[(batch, seq_len, output_size)] - Model outputs during LU steps
    - latent_updating_combined_inputs: List - Combined input during latent updates
    - latent_updating_grad_model_outputs: List - Model output gradients during LU
    
    Task Context IDs:
    - llcids: List[(batch, stride, 1)] - Low-level context IDs (block-level latent indicators)
    - hlcids: List[(batch, stride, 1)] - High-level context IDs (hierarchical latent indicators)
    
    Model Internal States:
    - hidden_states: List[(batch, seq_len, hidden_size)] - RNN/LSTM hidden states (h, c)
             Only logged if config.log_hidden_states=True
             For LSTM: tuple of (h, c), Logger extracts h via hidden_states[0]
    - input_attention_weights: List - Attention weights if using input attention
    
    Training Phases:
    - phases: List[(phase_name: str, timestep: int)] - Marks phase transitions
             Common phases: 'no inference learning', 'Learning and inference', 'Inference only'
             Timestep is cumulative count from len(logger.inputs) when phase starts
    
    Miscellaneous:
    - others: defaultdict(list) - Flexible storage for experiment-specific data
             Common keys:
             - 'grad_norms': List[float] - Mean gradient norms of model weights
             - 'task_latents': List or array - Ground truth latent states from dataset
             - 'timestep_passive_learning_ended': int - End of passive phase
             - 'timestep_learning_ended': int - End of active learning phase
             - 'P': array - Randomly chosen vectors for mul_gating models
             - 'input_layer_weights': array - Input layer weights at end of training
             - 'rnn_cell.weight_hh': List[array] - LSTM weight matrices over time
    
    USAGE EXAMPLES:
    # Concatenate inputs (observations)
    ii = np.concatenate(logger.inputs, axis=0)  # Shape: (n_batches, stride, input_size)
    ii = ii.reshape(-1, ii.shape[-1])  # Shape: (total_timesteps, input_size)
    
    # Concatenate predictions
    oi = np.concatenate(logger.predicted_outputs, axis=0)
    oi = oi.reshape(-1, oi.shape[-1])  # Shape: (total_timesteps, output_size)
    
    # Concatenate latents
    li = np.concatenate(logger.latent_values, axis=0)
    li = li.reshape(-1, li.shape[-1])  # Shape: (total_timesteps, latent_dims)
    
    # Compute per-dimension MSE
    mse_per_dim = ((oi - ii) ** 2).mean(axis=0)  # Shape: (output_size,)
    
    # Extract specific latent chunks
    if config.latent_chunks > 1:
        chunk_size = li.shape[-1] // config.latent_chunks
        z1 = li[:, :chunk_size]  # First chunk
        z2 = li[:, chunk_size:2*chunk_size]  # Second chunk
    """
    def __init__(self):
        self.training_batches = []
        self.training_losses = []
        self.training_losses_before_latent_optimization = []
        self.testing_batches = []
        self.testing_losses = []
        self.latent_values = []
        self.latent_gradients = []
        self.latent_updating_losses = [] # to store the loss at each optimization round
        self.latent_updating_latents = [] # to store the latent at each optimization round
        self.latent_updating_combined_inputs = []
        self.latent_updating_outputs = [] 
        self.latent_updating_grad_model_outputs = []
        self.predicted_outputs = []
        self.prediction_losses = []
        self.inputs = []
        self.llcids = []
        self.hlcids = []
        self.hidden_states = []
        self.gradients_max_entropy = []
        self.gradients_corrections = []
        self.input_attention_weights = []
        self.phases = [] # tracks training and testing phases with a tuple, phase name, and time step it started
        self.others = defaultdict(list) # a blank dict to store all odds and ends


    def log_weights(self, model : RNN_with_latent):
        # log the weights of the model
        for name, param in model.named_parameters():
            if 'rnn_cell.weight_hh' in name:
                w_hh = param
                W_ig, W_fg, W_gg, W_og = w_hh.chunk(4, dim=0)
                self.others[name].append(W_gg.clone().cpu().detach().numpy()[:64, :64])  # LSTMs store weights for all operations in one W, so the first chunk of it is 

    def log_hidden_states(self, hidden_states):
        # check if two or one hidden states (e.g. LSTM)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        self.hidden_states.append(hidden_states.cpu().detach().numpy())

    def log_phase(self, phase_name):
        if self.inputs:
            linn = np.stack(self.inputs)

            inputs_len = len(linn.reshape(-1, linn.shape[-1]))
        else:
            inputs_len = 0
        time_step = inputs_len
        self.phases.append((phase_name, time_step))

    def log_updating_combined_input(self, combined_input):
        self.latent_updating_combined_inputs.append(combined_input)

    def log_updating_output(self, output):
        self.latent_updating_outputs.append(output)
        
    def log_updating_loss(self, loss):
        self.latent_updating_losses.append(loss)

    def log_training_batch(self, batch):
        self.training_batches.append(batch)

    def log_training_loss(self, loss):
        self.training_losses.append(loss)

    def log_testing_batch(self, batch):
        self.testing_batches.append(batch)

    def log_testing_loss(self, loss):
        self.testing_losses.append(loss)

    def log_latent_value(self, value):
        self.latent_values.append(value)

    def log_latent_gradient(self, gradient):
        self.latent_gradients.append(gradient)
    
    def log_training_loss_before_latent_optimization(self, loss):
        self.training_losses_before_latent_optimization.append(loss)

    def log_updating_latent(self, latent):
        self.latent_updating_latents.append(latent)

    def log_predicted_output(self, output):
        self.predicted_outputs.append(output)

    def log_prediction_loss(self, loss):
        self.prediction_losses.append(loss)
    
    def log_input(self, input):
        self.inputs.append(input)
    def log_gradients_max_entropy(self, gradient):
        self.gradients_max_entropy.append(gradient)
    def log_gradients_corrections(self, gradient):
        self.gradients_corrections.append(gradient)
    def log_updating_grad_model_outputs(self, grad_output):
        self.latent_updating_grad_model_outputs.append(grad_output)



#%%
obs_color = 'black'
preds_color = 'tab:red'


def get_matching_loggers( parameters_to_match, parameter_values_to_match, parameter_combinations, parameters_to_sweep, data_folder):
    files = os.listdir(data_folder)
    matching_exp_idxs = []
    for exp_idx, experiment in enumerate(parameter_combinations):
        matched = True
        for param_name, param_value in zip(parameters_to_match, parameter_values_to_match):
            if experiment[parameters_to_sweep.index(param_name)] != param_value:
                matched = False
        if matched:
            matching_exp_idxs.append(exp_idx)


    # print(f'Found {len(matching_exp_idxs)} experiments that match the parameters')

    # Loading the logger_train and logger_test for the matching experiments
    not_found_idxs = []
    loggers_train_that_match = []
    loggers_test_that_match = []
    for exp_idx in matching_exp_idxs:
        param_values = parameter_combinations[exp_idx][:len(parameters_to_sweep)]
        file_name = '_'.join([f'{param_name}_{param_value}' for param_name, param_value in zip(parameters_to_sweep, param_values)])
        if file_name + '_logger_train.npy' in files and file_name + '_logger_test.npy' in files:
            logger_train = np.load(data_folder + f'{file_name}_logger_train.npy', allow_pickle=True).item()
            logger_test = np.load(data_folder + f'{file_name}_logger_test.npy', allow_pickle=True).item()
            loggers_train_that_match.append(logger_train)
            loggers_test_that_match.append(logger_test)
            # print(f'Files found for experiment {exp_idx + 1} with parameters {param_values}')
        else:
            not_found_idxs.append(exp_idx)
            # pri nt(f'Files not found for experiment {exp_idx + 1}')
    if len(not_found_idxs) > 0:
        print(f'Files not found for {len(not_found_idxs)} out of {len(matching_exp_idxs)} experiments')
        print(f'Not found indexes: {not_found_idxs}')
    if len(loggers_train_that_match) == 0:
        print('No matching loggers found')
        raise ValueError(f'No matching loggers found for parameters \n{parameters_to_match} \nand values \n{parameter_values_to_match}')
    return loggers_train_that_match,loggers_test_that_match

def plot_corrects_by_transition_many(loggers, get_corrects_and_trial_starts):
    all_corrects_by_transition = []
    for logger in loggers:    
        corrects, states, transitions, A_starts, B_starts, both_starts = get_corrects_and_trial_starts(logger)
        corrects_by_transition = {k:[] for k in transitions.keys()}
        for k, v in transitions.items():
            corrects_by_transition[k] = corrects[both_starts+v]
        all_corrects_by_transition.append(corrects_by_transition)

    fig, axes = plt.subplots(6,1  , figsize=(7, 4))
    # Assuming the same switches for all loggers
    TIS_time_steps = both_starts + 1
    switches = [ts for ts in TIS_time_steps if states[ts] != states[ts-6]]
    
    averaged_corrects_by_transition = {k: np.mean([d[k] for d in all_corrects_by_transition], axis=0) for k in transitions.keys()}
    std_corrects_by_transition = {k: np.std([d[k] for d in all_corrects_by_transition], axis=0) for k in transitions.keys()}
    for i, (k, v) in enumerate(averaged_corrects_by_transition.items()):
    
        ax = axes[i]
        ax.plot(both_starts+v, v, label=k, color = 'C'+str(i))
        # also plot the individual runs with a lower alpha
        for d in all_corrects_by_transition:
            ax.plot(both_starts+v, d[k], color = 'C'+str(i), alpha=0.3)

        # ax.f ill_between(both_starts+v, v - std_corrects_by_transition[k], v + std_corrects_by_transition[k], alpha=0.3, color = 'C'+str(i))
        ax.set_ylabel('Correct')
        ax.set_xlabel('Time step')
        ax.legend()
        ax.set_ylim(0, 1)
        if len(switches) < 20: # takes too long to plot if interleaved
            for switch in switches:
                ax.axvline(switch, color='r', linestyle=':', alpha=0.7)
    over_all_accuracy = [np.mean(averaged_corrects_by_transition[k]) for k in averaged_corrects_by_transition.keys() if k in ['T1/2', 'T5/6', 'T7/8']]
    return fig, np.mean(over_all_accuracy)

def get_accuracy(loggers,get_corrects_and_trial_starts,
                 transitions_to_average = ['T1/2', 'T5/6', 'T7/8']):
    '''
    This function returns the accuracy for the transitions_to_average averaged across seeds
    BIG assumption. This function assumes that the switches are the same for all loggers
    So the corrects for each transition are stacked into a big matrix of corrects
    The rows are individual seeds, and the col is the time steps.

    '''
    all_corrects_by_transition = []
    for logger in loggers:    
        corrects, states, transitions, A_starts, B_starts, both_starts = get_corrects_and_trial_starts(logger)
        # both_starts are the starting time steps for all trial types A or B.
        corrects_by_transition = {k:[] for k in transitions.keys()}
        for k, v in transitions.items(): # for each type of transition, get all the corrects across time.
            corrects_by_transition[k] = corrects[both_starts+v]
        all_corrects_by_transition.append(corrects_by_transition)

    # Assuming the same switches for all loggers
    TIS_time_steps = both_starts + 1 # get the indices of all the task identifying state.
    switches = [ts for ts in TIS_time_steps if states[ts] != states[ts-6]]
    
    corrects_by_tranistion_averaged_over_seeds = {k: np.mean([d[k] for d in all_corrects_by_transition], axis=0) for k in transitions.keys()}
    # std_corrects_by_transition = {k: np.std([d[k] for d in all_corrects_by_transition], axis=0) for k in transitions.keys()}
     
    accuracies = [np.mean(corrects_by_tranistion_averaged_over_seeds[k]) for k in corrects_by_tranistion_averaged_over_seeds.keys() if k in transitions_to_average]
    stds = [np.std(corrects_by_tranistion_averaged_over_seeds[k]) for k in corrects_by_tranistion_averaged_over_seeds.keys() if k in transitions_to_average]

    return accuracies, stds

def get_accuracies_averaged_across_time(loggers, get_corrects_and_trial_starts,
                 transitions_to_average = ['T1/2', 'T5/6', 'T7/8'],
                   return_correct_t56_ema_thresh_crossing=False, alpha =0.25, thresh=0.95):
    '''
    This function takes in loggers and returns accuracy averaged across time
    returns a list of accuracies one for each logger (usually each logger is a different seed)
    
    As opposed to get_accuracies which averages across seeds but keeps the time dim

    More recently, I added the option to return the time at which the EMA of the corrects for T5/6 crosses a threshold
    
    '''
    all_corrects_by_transition = []
    cross_indices = []
    for logger in loggers:    
        corrects, states, transitions, A_starts, B_starts, both_starts = get_corrects_and_trial_starts(logger)
        corrects_by_transition = {k:[] for k in transitions.keys()}
        for k, v in transitions.items():
            corrects_by_transition[k] = corrects[both_starts+v]
        all_corrects_by_transition.append(corrects_by_transition)

        if return_correct_t56_ema_thresh_crossing:
            corrects_t56 = corrects_by_transition['T5/6']
            corrects_t56_moving_average = np.zeros_like(corrects_t56)
            corrects_t56_moving_average[0] = 0.5  # Starting value for EMA
            
            # Calculate the EMA
            for i in range(1, len(corrects_t56)):
                corrects_t56_moving_average[i] = corrects_t56_moving_average[i-1] + alpha * (corrects_t56[i] - corrects_t56_moving_average[i-1])
            cross_index = np.argmax(corrects_t56_moving_average > thresh)  # Returns 0 if never crosses
            cross_indices.append(cross_index)
    # penalize models that never cross the threshold
    if return_correct_t56_ema_thresh_crossing:
        cross_indices = [ci if ci > 0 else len(corrects_t56) for ci in cross_indices]

    # average across seeds, but keeps the time dim
    averaged_across_time_by_transition = {}
    for k in transitions.keys():
        averaged_across_time_by_transition[k] = np.mean([d[k] for d in all_corrects_by_transition], axis=1)
    # you get a value per logger/seed
    # now group the data for the transitions you want to average
    accuracies = [acc for k, acc in averaged_across_time_by_transition.items() if k in transitions_to_average]

    if return_correct_t56_ema_thresh_crossing:
        return accuracies, cross_indices
    else:
        return accuracies
    
def criterion_self_fullfilling_prophecy(output, _):
    ''' This loss function is used to capture the COIN task where humans are given motor feedback 
    that exactly matches their predictions. The loss is the difference between the predicted output and the most recent output. ''' 
    return torch.nn.functional.mse_loss(output[:, 1:], output[:, :-1].detach(), reduction='none')

# A function to rasterize components of a matplotlib figure while keeping
# axes, labels, etc as vector components
# https://brushingupscience.wordpress.com/2017/05/09/vector-and-raster-in-one-with-matplotlib/
from inspect import getmembers, isclass

def rasterize_and_save(fname, rasterize_list=None, fig=None, dpi=None,
                       savefig_kw={}):
    """Save a figure with raster and vector components

    This function lets you specify which objects to rasterize at the export
    stage, rather than within each plotting call. Rasterizing certain
    components of a complex figure can significantly reduce file size.

    Inputs
    ------
    fname : str
        Output filename with extension
    rasterize_list : list (or object)
        List of objects to rasterize (or a single object to rasterize)
    fig : matplotlib figure object
        Defaults to current figure
    dpi : int
        Resolution (dots per inch) for rasterizing
    savefig_kw : dict
        Extra keywords to pass to matplotlib.pyplot.savefig

    If rasterize_list is not specified, then all contour, pcolor, and
    collects objects (e.g., ``scatter, fill_between`` etc) will be
    rasterized

    Note: does not work correctly with round=True in Basemap

    Example
    -------
    Rasterize the contour, pcolor, and scatter plots, but not the line

    >>> import matplotlib.pyplot as plt
    >>> from numpy.random import random
    >>> X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
    >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    >>> cax1 = ax1.contourf(Z)
    >>> cax2 = ax2.scatter(X, Y, s=Z)
    >>> cax3 = ax3.pcolormesh(Z)
    >>> cax4 = ax4.plot(Z[:, 0])
    >>> rasterize_list = [cax1, cax2, cax3]
    >>> rasterize_and_save('out.svg', rasterize_list, fig=fig, dpi=300)
    """

    # Behave like pyplot and act on current figure if no figure is specified
    fig = plt.gcf() if fig is None else fig

    # Need to set_rasterization_zorder in order for rasterizing to work
    zorder = -5  # Somewhat arbitrary, just ensuring less than 0

    if rasterize_list is None:
        # Have a guess at stuff that should be rasterised
        types_to_raster = ['QuadMesh', 'Contour', 'collections']
        rasterize_list = []

        print("""
        No rasterize_list specified, so the following objects will
        be rasterized: """)
        # Get all axes, and then get objects within axes
        for ax in fig.get_axes():
            for item in ax.get_children():
                if any(x in str(item) for x in types_to_raster):
                    rasterize_list.append(item)
        print('\n'.join([str(x) for x in rasterize_list]))
    else:
        # Allow rasterize_list to be input as an object to rasterize
        if type(rasterize_list) != list:
            rasterize_list = [rasterize_list]

    for item in rasterize_list:

        # Whether or not plot is a contour plot is important
        is_contour = (isinstance(item, matplotlib.contour.QuadContourSet) or
                      isinstance(item, matplotlib.tri.TriContourSet))

        # Whether or not collection of lines
        # This is commented as we seldom want to rasterize lines
        # is_lines = isinstance(item, matplotlib.collections.LineCollection)

        # Whether or not current item is list of patches
        all_patch_types = tuple(
            x[1] for x in getmembers(matplotlib.patches, isclass))
        try:
            is_patch_list = isinstance(item[0], all_patch_types)
        except TypeError:
            is_patch_list = False

        # Convert to rasterized mode and then change zorder properties
        if is_contour:
            curr_ax = item.ax.axes
            curr_ax.set_rasterization_zorder(zorder)
            # For contour plots, need to set each part of the contour
            # collection individually
            for contour_level in item.collections:
                contour_level.set_zorder(zorder - 1)
                contour_level.set_rasterized(True)
        elif is_patch_list:
            # For list of patches, need to set zorder for each patch
            for patch in item:
                curr_ax = patch.axes
                curr_ax.set_rasterization_zorder(zorder)
                patch.set_zorder(zorder - 1)
                patch.set_rasterized(True)
        else:
            # For all other objects, we can just do it all at once
            curr_ax = item.axes
            curr_ax.set_rasterization_zorder(zorder)
            item.set_rasterized(True)
            item.set_zorder(zorder - 1)

    # dpi is a savefig keyword argument, but treat it as special since it is
    # important to this function
    if dpi is not None:
        savefig_kw['dpi'] = dpi

    # Save resulting figure
    fig.savefig(fname, **savefig_kw)


# ──────────────────────────────────────────────────────────────────────────────
# Effective LR logging (Adam-specific)
# ──────────────────────────────────────────────────────────────────────────────

def _log_effective_lr(model, config, logger):
    """
    Compute and log the effective per-parameter learning rate for the Z optimizer.

    For Adam: effective_lr[i] = lr / (sqrt(v_hat[i]) + eps)
    where v_hat is the bias-corrected second moment estimate.
    This gives a sense of how aggressively each Z dimension is being updated.
    """
    try:
        opt = model.Z_optimizer
        group = opt.param_groups[0]
        lr = float(group.get("lr", 0.0))
        betas = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        params = group.get("params", [])
        if not params:
            return
        state = opt.state.get(params[0], None)
        if state is None or "exp_avg_sq" not in state or "step" not in state:
            return
        v = state["exp_avg_sq"]
        step = state["step"]
        beta2 = betas[1]
        bias_correction2 = 1.0 - beta2 ** step if step > 0 else 1.0
        v_hat = v / bias_correction2
        eff_lr = lr / (v_hat.sqrt() + eps)
        eff_slice = eff_lr[:, -config.stride:, 0:1].cpu().detach().numpy()
        logger.others.setdefault("latent_effective_lr", []).append(eff_slice)
    except Exception:
        pass  # non-fatal; skip if optimizer state isn't available yet

