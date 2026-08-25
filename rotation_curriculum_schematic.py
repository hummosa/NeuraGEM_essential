"""Curriculum schematic — pure-diagram companion to Fig F (`plot_s1_vs_s3_by_zlr` in
`rotation_curriculum_analysis.py`).

Draws the three-stage curriculum (S1 uncued self-inference -> S2 cued -> S3 uncued, weights
frozen) as a 2-row x 3-column schematic: an illustrative blocked-context bar per stage on top, a
two-node latent-context diagram per stage below. No dependency on trained models, `cache`, or
`AnalysisParams` — everything here is descriptive, not measured, which is why it lives in its own
module rather than inside the data-analysis one.

Companion to, and deliberately NOT merged into, `plot_s1_vs_s3_by_zlr`'s own figure object — a
separate file/function/PDF, sized to the same width so the two stack cleanly in a document.

## Why S1 and S3 draw the identical node/arrow construct

Labeling the reciprocal arrows between the two latent nodes with `alpha_z` alone risks reading as
an HMM transition *probability* rather than a latent *update rate*. Resolved by drawing S1 and S3
with the literal same construct — reciprocal curved arrows, a small rate glyph, an `alpha_z`
label — because the mechanism (Z updates via self-inference at rate `alpha_z`) really is identical
in both stages; only the outcome differs (S1: the weights don't yet know how to use it, drawn as
unfilled '?' nodes; S3: they do, drawn as filled colored nodes). Repeating the identical construct
is itself the argument that it's a rate, not a fixed per-transition probability — a real
transition probability wouldn't appear unchanged in a stage where the model manifestly fails to
use it. S2 gets a different construct entirely: two nodes, both already colored (identities
fixed), no reciprocal arrows, no rate glyph, no `alpha_z` label — one external "cue" arrow
instead, since S2 hands Z the answer directly (`no_of_steps_in_latent_space=0`) rather than
inferring it.

No connecting arrows between the three stage-columns — deliberately minimal; left-to-right order
plus the repeated S1/S3 construct already implies the same model is carried through.

Usage:
    from rotation_curriculum_schematic import plot_curriculum_schematic
    plot_curriculum_schematic()
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

import plot_style
from plot_style import Color_scheme, FigSize
from rotation_curriculum_config import EXPORT_ROOT

plot_style.set_plot_style()


# Illustrative only — NOT the real nominal block counts (~14/29/21 at block_size=140; see
# S1_LENGTH/S2_LENGTH/S3_LENGTH in rotation_curriculum_config.py). Chosen so S1 < S3 < S2 in
# roughly the real 1 : 2.0 : 1.5 ratio; fixed block width/spacing at a growing count naturally
# renders S2 denser, which is itself the "S2 is the longest stage" cue.
S1_N_BLOCKS, S2_N_BLOCKS, S3_N_BLOCKS = 4, 7, 5

COLUMN_TITLES = ('S1 — uncued', 'S2 — cued', 'S3 — uncued, weights frozen')

# Node layout shared by the S1/S3 and S2 constructs, in each row-2 axis's [0,1]x[0,1] coordinates.
_NODE_XY = ((0.22, 0.36), (0.78, 0.36))
_NODE_R = 0.13


# ---------------------------------------------------------------------------
# Row 1: blocked-context bar
# ---------------------------------------------------------------------------

def _draw_block_bar(ax, n_blocks: int, color1, color2, *, active_index: int | None = None) -> None:
    """Alternating context-A/context-B blocks. `n_blocks` is a small illustrative count, not the
    real nominal block count. `active_index`, if given, outlines that block in black to mark it
    as the "current" block — used only for S2, so the cue arrow below has something to visually
    sync with.
    """
    block_width, block_height, spacing, y_base = 0.7, 0.33, 0.08, 0.13
    for i in range(n_blocks):
        x = i * (block_width + spacing)
        color = color1 if i % 2 == 0 else color2
        is_active = i == active_index
        rect = Rectangle((x, y_base), block_width, block_height, facecolor=color,
                         edgecolor='k' if is_active else 'none',
                         linewidth=0.8 if is_active else 0, alpha=0.7)
        ax.add_patch(rect)
    ax.set_xlim(-0.1, n_blocks * (block_width + spacing))
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')


# ---------------------------------------------------------------------------
# Row 2: latent-context node diagrams
# ---------------------------------------------------------------------------

def _draw_node(ax, cx: float, cy: float, r: float, *, filled: bool, color=None,
               mark: str | None = None) -> Circle:
    """One latent-context node. Unfilled + dashed grey outline + '?' mark for S1; filled + solid
    colored + black outline for S2/S3. Returns the Circle patch so callers can pass it to
    FancyArrowPatch's patchA/patchB, which clips the arrow to the node boundary.
    """
    if filled:
        circ = Circle((cx, cy), r, facecolor=color, edgecolor='k', linewidth=0.6,
                      alpha=0.85, zorder=2)
    else:
        circ = Circle((cx, cy), r, facecolor='none', edgecolor='0.55', linewidth=1.0,
                      linestyle='--', zorder=2)
    ax.add_patch(circ)
    if mark:
        ax.text(cx, cy, mark, ha='center', va='center', color='0.45', fontsize=8, zorder=3)
    return circ


def _draw_reciprocal_arrows(ax, circ_left: Circle, circ_right: Circle) -> None:
    """The classic two-arrow HMM look: one FancyArrowPatch each direction, arced opposite ways so
    they don't overlap. patchA/patchB let matplotlib clip the arrow to the node boundary.
    """
    kw = dict(arrowstyle='-|>', mutation_scale=6, linewidth=0.9, color='0.25', zorder=1)
    ax.add_patch(FancyArrowPatch(circ_left.center, circ_right.center,
                                 connectionstyle='arc3,rad=0.4',
                                 patchA=circ_left, patchB=circ_right,
                                 shrinkA=1, shrinkB=1, **kw))
    ax.add_patch(FancyArrowPatch(circ_right.center, circ_left.center,
                                 connectionstyle='arc3,rad=-0.4',
                                 patchA=circ_right, patchB=circ_left,
                                 shrinkA=1, shrinkB=1, **kw))


def _draw_rate_glyph(ax, cx: float, cy: float, r: float, *, color: str = '0.25') -> None:
    """Minimal clock glyph: a circle + two hands, set to a non-noon angle so it reads as
    "ticking"/rate rather than a stopped clock. No icon library — a Circle outline, two Line2D
    hands, and a small filled Circle pivot. Signals "rate", not "probability", alongside the
    alpha_z label.
    """
    ax.add_patch(Circle((cx, cy), r, facecolor='none', edgecolor=color, linewidth=1.2, zorder=4))
    for angle_deg, length_frac, lw in ((75, 0.55, 1.4), (-35, 0.85, 1.0)):
        t = np.deg2rad(angle_deg)
        ax.add_line(Line2D([cx, cx + length_frac * r * np.cos(t)],
                           [cy, cy + length_frac * r * np.sin(t)],
                           color=color, linewidth=lw, solid_capstyle='round', zorder=6))
    ax.add_patch(Circle((cx, cy), 0.14 * r, facecolor=color, edgecolor='none', zorder=7))


def _draw_self_infer_pair(ax, *, filled: bool) -> None:
    """S1/S3 construct: two nodes + reciprocal curved arrows + rate glyph + alpha_z label —
    literally the same drawing call for S1 (filled=False) and S3 (filled=True). The mechanism
    (self-inference, Z updates at rate alpha_z) is identical in both; only the fill/outline
    (unfilled dashed '?' vs filled solid color) shows whether the weights have learned to use it.
    """
    cs = Color_scheme()
    (lx, ly), (rx, ry) = _NODE_XY
    color_l = cs.contextA if filled else None
    color_r = cs.contextB if filled else None
    circ_l = _draw_node(ax, lx, ly, _NODE_R, filled=filled, color=color_l, mark=None if filled else '?')
    circ_r = _draw_node(ax, rx, ry, _NODE_R, filled=filled, color=color_r, mark=None if filled else '?')
    _draw_reciprocal_arrows(ax, circ_l, circ_r)
    _draw_rate_glyph(ax, cx=0.5, cy=0.84, r=_NODE_R * 0.55)
    ax.text(0.5, 0.98, r'$\alpha_z$', ha='center', va='bottom', color='0.25', fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def _draw_cued_pair(ax, *, active: str = 'A') -> None:
    """S2 construct: two filled, colored nodes — both identities known/fixed. No reciprocal
    arrows, no rate glyph, no alpha_z label (LU is off in S2 — Z is handed the answer, not
    inferred). One external 'cue' arrow points down into whichever node `active` names.
    """
    cs = Color_scheme()
    (lx, ly), (rx, ry) = _NODE_XY
    circ_l = _draw_node(ax, lx, ly, _NODE_R, filled=True, color=cs.contextA)
    circ_r = _draw_node(ax, rx, ry, _NODE_R, filled=True, color=cs.contextB)
    target = circ_l if active == 'A' else circ_r
    tx, ty = target.center
    start = (tx, 0.93)
    ax.add_patch(FancyArrowPatch(start, target.center, arrowstyle='-|>', mutation_scale=5,
                                 linewidth=0.9, color='k', patchB=target, shrinkA=0, shrinkB=2))
    ax.text(start[0], start[1] + 0.02, 'cue', ha='center', va='bottom', color='0.15', fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _save_schematic(fig, export_dir: Path, name: str, save_plots: bool, show_plots: bool) -> None:
    """Mirrors rotation_curriculum_analysis._save's save/show pattern, but takes plain bools
    instead of an AnalysisParams object, since this module has none.
    """
    if save_plots:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / name
        fig.savefig(out, bbox_inches='tight')
        print(f'  Saved -> {out}')
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_curriculum_schematic(export_dir: Path = EXPORT_ROOT / 'figures', save_plots: bool = True,
                              show_plots: bool = True, dpi: int = 160) -> plt.Figure:
    """Companion schematic drawn ABOVE Fig F: the three-stage curriculum, drawn not measured.

    2x3 grid. Row 1: an illustrative blocked-context bar per stage. Row 2: a two-node
    latent-context diagram per stage — S1 and S3 draw the identical self-inference construct
    (reciprocal arrows, rate glyph, alpha_z label), differing only in node fill; S2 has no such
    mechanism drawn at all, just two known colored identities and one external cue arrow. No
    connecting arrows between columns (deliberate: left-to-right order plus S1/S3's repeated
    construct already implies continuity).

    No dependency on trained-model data/cache/AnalysisParams — pure diagram, independently
    callable/testable.
    """
    cs = Color_scheme()
    fig, axes = plt.subplots(2, 3, figsize=FigSize.custom(4.0, 1.5), dpi=dpi,
                             height_ratios=[0.4, 1], layout='constrained')

    s2_active_idx = S2_N_BLOCKS - 1                              # "current" block = rightmost
    s2_active_color = 'A' if s2_active_idx % 2 == 0 else 'B'     # single source of truth, keeps
                                                                  # bar highlight and cue arrow synced

    bar_specs = ((S1_N_BLOCKS, None), (S2_N_BLOCKS, s2_active_idx), (S3_N_BLOCKS, None))
    for col, (n_blocks, active_idx) in enumerate(bar_specs):
        _draw_block_bar(axes[0, col], n_blocks, cs.contextA, cs.contextB, active_index=active_idx)
        axes[0, col].set_title(COLUMN_TITLES[col], fontsize=6)

    _draw_self_infer_pair(axes[1, 0], filled=False)
    _draw_cued_pair(axes[1, 1], active=s2_active_color)
    _draw_self_infer_pair(axes[1, 2], filled=True)

    _save_schematic(fig, export_dir, 'schematic_curriculum.pdf', save_plots, show_plots)
    return fig


if __name__ == '__main__':
    plot_curriculum_schematic()
