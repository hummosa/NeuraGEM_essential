"""
Debug visualization for a single trial.
Drop this into the debugger after model_inputs, outputs, context_ids are in scope.

Shapes expected:
  model_inputs  : (1, T, 6)  — 5 arrow slots + true direction (masked in practice)
  outputs       : (1, T, 6)  — model predictions
  context_ids   : (1, T, 1)  — target slot or congruency flag
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plot_style
plot_style.set_plot_style()
plt.rcParams['figure.dpi'] = 200
labelpad= 15
# ── extract numpy ──────────────────────────────────────────────────────────────
def _np(t):
    return t.detach().cpu().numpy().squeeze(0)  # drop batch dim → (T, D)

ci  = _np(model_inputs)   # (T, 6) — dim -1 is masked to 0 (model never sees true direction)
out = _np(outputs)        # (T, 6)
ctx = _np(context_ids)    # (T, 1) or (T,)

# Ground truth direction lives in the unmasked 'inputs' tensor (also in LU scope)
try:
    gt_vals = _np(inputs)[:, -1]   # noqa: F821 — inputs expected in debugger scope
except Exception:
    gt_vals = None   # not available; GT subplot will be skipped

T = ci.shape[0]
t_axis = np.arange(T)

n_arrow_slots = 5
slot_colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
slot_labels = ['Far_L', 'Near_L', 'Center', 'Near_R', 'Far_R']

# Temporal loss weights — read from model config if available
try:
    tw = list(model.config.temporal_loss_weights)  # noqa: F821 — model expected in debugger scope
except Exception:
    tw = None

# ── layout ─────────────────────────────────────────────────────────────────────
# rows: 5 slots | ground truth | context | [time pressure] | output
extra_heights = [0.5, 0.8] + ([0.5] if tw is not None else []) + [1.0]
n_rows = n_arrow_slots + len(extra_heights)

fig = plt.figure(figsize=(3.5, n_arrow_slots * 0.65 + sum(extra_heights) + 0.3))
gs  = gridspec.GridSpec(
    n_rows, 1,
    height_ratios=[1.0] * n_arrow_slots + extra_heights,
    hspace=0.18,
)

# ── arrow slot subplots ────────────────────────────────────────────────────────
arrow_axs = [fig.add_subplot(gs[i]) for i in range(n_arrow_slots)]
for i, ax in enumerate(arrow_axs):
    ax.plot(t_axis, ci[:, i], color=slot_colors[i], linewidth=1.75, solid_capstyle='round')
    ax.axhline(0, color='k', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_ylabel(slot_labels[i], rotation=0, labelpad=labelpad, va='center')
    ax.set_xlim(-0.5, T - 0.5)
    # ax.set_ylim(-2.5, 2.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # share y lims across all arrow slots
    y_min, y_max = ax.get_ylim()
    y_lim_max = max(abs(y_min), abs(y_max))
    if i == 0:
        mmax = y_lim_max
    else:
        mmax = max(mmax, y_lim_max)
for ax in arrow_axs:
    ax.set_ylim(-mmax * 1.1, mmax * 1.1)

# Label 'left' / 'right' directions on the first slot only
arrow_axs[0].text(T - 0.4, 0.4, 'right', va='bottom', ha='right', color='k', alpha=0.55)
arrow_axs[0].text(T - 0.4, -0.4, 'left',  va='top',    ha='right', color='k', alpha=0.55)

# ── ground truth action ────────────────────────────────────────────────────────
row = n_arrow_slots
ax_gt = fig.add_subplot(gs[row])
if gt_vals is not None:
    gt_val = float(gt_vals[0])   # constant within trial: ±1
    ax_gt.hlines(gt_val, 0., T - 1 , colors='k', linewidth=6, zorder=3)
else:
    ax_gt.text(0.5, 0.5, 'inputs not in scope', ha='center', va='center',
               transform=ax_gt.transAxes, alpha=0.4)
ax_gt.axhline(0, color='k', linewidth=0.4, linestyle=':', alpha=0.5)
ax_gt.set_ylabel('GT dir', rotation=0, va='center')
ax_gt.yaxis.set_label_coords(-0.1, 0.5)
ax_gt.set_xlim(-0.5, T - 0.5)
ax_gt.set_ylim(-1.5, 1.5)
ax_gt.set_yticks([-1, 1])
ax_gt.set_xticks([])
ax_gt.spines['bottom'].set_visible(False)
ax_gt.spines['top'].set_visible(False)
ax_gt.spines['right'].set_visible(False)

# ── context ids ────────────────────────────────────────────────────────────────
row += 1
ax_ctx = fig.add_subplot(gs[row])
ctx_vals = ctx.reshape(T)
ax_ctx.plot(t_axis, ctx_vals, color='#444444', linewidth=7, solid_capstyle='butt')
for boundary in [0.5, 1.5, 2.5, 3.5]:
    ax_ctx.axhline(boundary, color='k', linewidth=0.8, linestyle='--', alpha=0.6)
ax_ctx.set_ylabel('ctx', rotation=0, va='center')
ax_ctx.yaxis.set_label_coords(-0.1, 0.5)
ax_ctx.set_xlim(-0.5, T - 0.5)
ax_ctx.set_ylim(-0.5, 4.5)
ax_ctx.set_yticks(range(5))
ax_ctx.set_xticks([])
ax_ctx.spines['bottom'].set_visible(False)
ax_ctx.spines['top'].set_visible(False)
ax_ctx.spines['right'].set_visible(False)
# flip y-axis so higher context IDs are visually higher
ax_ctx.invert_yaxis()

# ── temporal loss weights (time pressure) ─────────────────────────────────────
if tw is not None:
    row += 1
    ax_tw = fig.add_subplot(gs[row])
    tw_padded = tw[:T] + [0.0] * max(0, T - len(tw))
    ax_tw.bar(t_axis[:len(tw_padded)], tw_padded, color='#555555', alpha=0.6, width=0.7)
    ax_tw.set_ylabel('loss\nwt', rotation=0, labelpad=labelpad, va='center')
    ax_tw.set_xlim(-0.5, T - 0.5)
    ax_tw.set_yticks([])
    ax_tw.set_xticks([])
    ax_tw.spines['bottom'].set_visible(False)
    ax_tw.spines['top'].set_visible(False)
    ax_tw.spines['right'].set_visible(False)

# ── output ─────────────────────────────────────────────────────────────────────
row += 1
ax_out = fig.add_subplot(gs[row])
output_colors = ['#aec7e8'] * (out.shape[1] - 1) + ['#1f77b4']
for d in range(out.shape[1]):
    alpha = 1.0 if d == out.shape[1] - 1 else 0.35
    lw    = 2.0 if d == out.shape[1] - 1 else 0.8
    ax_out.plot(t_axis, out[:, d], color=output_colors[d], alpha=alpha, linewidth=lw,
                label=f'out[{d}]' if d == out.shape[1] - 1 else None)
ax_out.axhline(0,  color='k', linewidth=0.4, linestyle=':')
ax_out.axhline( 1, color='k', linewidth=0.6, linestyle='--', alpha=0.35)
ax_out.axhline(-1, color='k', linewidth=0.6, linestyle='--', alpha=0.35)
ax_out.set_ylim(-1.3, 1.3)
ax_out.set_xlim(-0.5, T - 0.5)
ax_out.set_xticks(t_axis)
ax_out.set_xticklabels([f't{j}' for j in t_axis])
ax_out.set_ylabel('output', rotation=0, va='center')
ax_out.yaxis.set_label_coords(-0.1, 0.5)
ax_out.legend(loc='upper right')
ax_out.spines['top'].set_visible(False)
ax_out.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# ── Correlation structure ──────────────────────────────────────────────────────
try:
    p_corr = list(model.config.p_corr_by_distance)  # noqa: F821
except Exception:
    p_corr = [1.0, 0.60, 0.40, 0.25, 0.1]

distances = np.arange(len(p_corr))

# Exponential fit: log(p) ~ a + b*d  →  p ~ exp(a) * exp(b*d)
coeffs = np.polyfit(distances, np.log(np.array(p_corr, dtype=float)), 1)
d_fine = np.linspace(0, len(p_corr) - 1, 300)
p_fit  = np.exp(np.polyval(coeffs, d_fine))

fig_corr, ax_corr = plt.subplots(figsize=(2.2, 1.8))
ax_corr.plot(d_fine, p_fit, color='#4393c3', linewidth=1.5)
ax_corr.scatter(distances, p_corr, color='k', zorder=3, s=18)
ax_corr.set_xlabel('Slot distance from target')
ax_corr.set_ylabel('P(congruent)')
ax_corr.set_ylim(0, 1.05)
ax_corr.set_xticks(distances)
ax_corr.spines['top'].set_visible(False)
ax_corr.spines['right'].set_visible(False)
fig_corr.tight_layout()
plt.show()
