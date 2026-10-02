"""Task schematic — the two-context ("context inference") version of the rotating-targets task.

Companion to `rotation_curriculum_schematic.plot_curriculum_schematic`, which draws the training
*protocol*; this one draws the *task* the protocol trains on. Same conventions: no dependency on
trained models, `cache` or `AnalysisParams` — everything here is descriptive, drawn, not measured.

Layout: the participant's view of one trial across the top (three screens: the shield cue, the
move, the attack), then a blocked-context strip (the two contexts alternate, switches unsignalled),
then one circular-arena panel per context. In each arena a shield cue appears
somewhere near the middle of the canvas, the agent moves it to where it believes that colour's
target is, and the attack lands as a draw from N(target, noise_std**2) — hence a cloud of outcomes
per colour rather than a point.

The first row and the arena row are deliberately different views of the same trial. The arenas are
the experimenter's: ring, five targets, a block's worth of outcomes, which context is active. The
top row is everything the participant actually gets — one shield, one placement, one dot — and it
is drawn from the same sampled trial as the context-A panel so the two rows are readable as one
event rather than two unrelated cartoons.

## What the two panels are contrasting, and what they are deliberately NOT

The experiment this figure fronts (`rotation_slips_perseveration*`, `rotation_curriculum*`) uses
TWO contexts alternating in blocks — not the paper's "rotate to an arbitrary novel angle" transfer
test. So the panels show two target *layouts* the agent must tell apart from noisy outcomes, and
the figure's subject is context ambiguity, not rotational generalisation. That is why:

* the relation between the two layouts is drawn at low emphasis — a thin grey arc and a small
  angle label on ONE colour, nothing more — and why `CONTEXT_GEOMETRY` can switch the second
  context to independently sampled target locations. If the schematic only reads correctly under
  `'rotated'`, it has smuggled in a claim (structure transfer) that this experiment never tests.
* the cue colour's target in the OTHER context is ghosted into each panel. That single ghost is
  the ambiguity: one outcome is weak evidence for which context is active, because the colour's
  two candidate targets are only a chord 2*r*sin(60/2) = 0.50 apart — a few sigma. (Ghosting all
  five would draw a second ring 12 deg off the first, since 60 deg of rotation against 72 deg
  target spacing nearly maps the layout onto itself as a *set*; that near-coincidence is a fact
  about 5 colours at 60 deg, not the point being made, and it reads as clutter.) See the
  ambiguity table in rotation_slips_perseveration_config.

Geometry comes from the real configs (`RotatingTargetsConfig`, `TRAIN_ROTATIONS`, `NOISE_LEVELS`)
rather than from numbers typed in here, so the drawn arena cannot drift away from the arena the
models are actually trained on. Two things are illustrative and labelled as such: the outcome
count per colour (a dozen-odd dots read as a cloud; a block's worth would fill the panel), and the
noise level — see NOISE_STD_DRAWN.

Usage:
    from rotation_task_schematic import plot_task_schematic
    plot_task_schematic()                                # three arenas: 0 deg, 60 deg, ?? deg
    plot_task_schematic(context_geometry='independent')  # two arenas, targets redrawn on the circle
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import plot_style
from configs import RotatingTargetsConfig
from plot_style import Color_scheme, FigSize
from rotation_curriculum_config import EXPORT_ROOT
from rotation_curriculum_schematic import draw_block_bar
from rotation_slips_perseveration_config import NOISE_LEVELS, TRAIN_ROTATIONS

plot_style.set_plot_style()


# ── Task geometry, taken from the real config ────────────────────────────────

_CFG = RotatingTargetsConfig()
N_COLORS      = _CFG.n_colors            # 5 shield colours
TARGET_RADIUS = _CFG.target_radius       # 0.5
ROTATION_DEG  = float(max(TRAIN_ROTATIONS) - min(TRAIN_ROTATIONS))   # 60 deg

#: Paper inches. Width is quoted PER ARENA PANEL, because the arena row holds two panels under
#: 'independent' and three under 'rotated' — a third context should widen the figure, not shrink
#: every panel in it. Height covers all three rows (participant view / block strip / arenas).
fig_x_per_arena = 1.15
fig_y = 2.20

#: How context B's targets relate to context A's. 'rotated' = every target turned by the same
#: ROTATION_DEG (what the task actually does); 'independent' = each colour's target redrawn
#: uniformly on the circle, i.e. two unrelated layouts. The experiment is about telling two
#: contexts apart, which is true either way — flip this to check that the figure still makes its
#: point when no shared rotation is available to exploit.
# CONTEXT_GEOMETRY = 'rotated'   # 'rotated' | 'independent'
CONTEXT_GEOMETRY = 'independent'

#: Third arena under 'rotated' only: a layout turned by an angle the model never trained on. Its
#: panel is titled with a question mark rather than the number, because that is the model's
#: position — it sees noisy outcomes from a layout it has never been in and has to work out how far
#: round it has been turned. Under 'independent' there is no shared rotation to be uncertain about,
#: so the panel does not exist and the figure stays at two arenas.
NOVEL_ROTATION_DEG = 170.0

#: The novel rotation is not one of the blocked training contexts, so it gets neither context
#: colour — grey says "not a block you have been in".
NOVEL_CONTEXT_COLOR = '0.45'

#: A real sweep level (NOISE_LEVELS = [0.04, 0.12, 0.20, 0.30]), but NOT the headline operating
#: point the experiments run at (HEADLINE_NOISE = 0.20). At 0.20 the clouds swamp the ring and the
#: figure stops showing what the task is; 0.12 keeps five distinguishable targets on a circle
#: while still making the point that an outcome is a draw, not a location. Pass noise_std=0.20 to
#: see the operating point — the clouds then reach well past the rival target.
NOISE_STD_DRAWN = NOISE_LEVELS[1]

#: Illustrative only. A real state-block is ~14 mini-blocks x 5 colours of outcomes per context.
N_OUTCOMES_PER_COLOR = 16
N_BLOCKS_IN_STRIP    = 6

#: Fixed so the figure is byte-identical run to run (the clouds, the cue's starting position and,
#: under 'independent', the target layout are all sampled).
SEED = 3

#: The shield does not appear at the centre. It shows up somewhere in the middle of the canvas and
#: the participant drags it out, so a cue pinned to (0, 0) in every panel draws a regularity the
#: task does not have. Draws are uniform in RADIUS over [0, CUE_START_MAX_R * target_radius] — not
#: uniform in area, which would pile them against the outer edge of the allowed disc and overstate
#: the wiggle. At 0.4 the cue stays well inside the ring, so it never reads as a sixth target.
CUE_START_MAX_R = 0.4

_SHIELD_CMAP = plt.get_cmap('tab10', N_COLORS)   # same colour code as plot_arena_trials


def _shield_colors():
    return [_SHIELD_CMAP(c) for c in range(N_COLORS)]


# ── Target layouts ────────────────────────────────────────────────────────────

def _base_angles() -> np.ndarray:
    """Context A: `n_colors` points evenly spaced on the circle — the dataset's 0 deg layout."""
    return 2 * np.pi * np.arange(N_COLORS) / N_COLORS


def _independent_angles(rng: np.random.Generator, min_gap: float, reference: np.ndarray,
                        cue_index: int, min_cue_shift: float = np.deg2rad(45)) -> np.ndarray:
    """Context B under `CONTEXT_GEOMETRY='independent'`: one angle per colour drawn uniformly and
    independently — not sorted, so the colours' cyclic order is scrambled too, which is the whole
    point of this variant (nothing about context A's layout carries over).

    Two rejection conditions, both legibility and both stated rather than hidden, since they make
    the draw not-quite-uniform: no two targets closer than `min_gap` (overlapping clouds read as
    one target), and the cue colour at least `min_cue_shift` from where it sits in context A (its
    context-A position is ghosted into the panel; a draw that lands on top of it hides the one
    marker carrying the ambiguity).
    """
    for _ in range(2000):
        angles = rng.uniform(0, 2 * np.pi, N_COLORS)
        ordered = np.sort(angles)
        gaps = np.diff(np.concatenate([ordered, ordered[:1] + 2 * np.pi]))
        cue_shift = abs((angles[cue_index] - reference[cue_index] + np.pi) % (2 * np.pi) - np.pi)
        if gaps.min() > min_gap and cue_shift > min_cue_shift:
            return angles
    return angles   # fell through: draw what we have rather than fail a schematic


def _targets(angles: np.ndarray) -> np.ndarray:
    """(n_colors, 2) target coordinates on the arena circle."""
    return TARGET_RADIUS * np.column_stack([np.cos(angles), np.sin(angles)])


def _context_layouts(context_geometry: str, rng: np.random.Generator, noise_std: float,
                     cue_index: int) -> list[np.ndarray]:
    """Target coordinates per arena panel — two under 'independent', three under 'rotated'.

    'rotated' gets the extra panel because it is the only variant in which the third arena means
    anything: the two trained layouts differ by a known angle, so a third one differing by an
    untrained angle poses the transfer question. Under 'independent' the layouts share no angle to
    extrapolate, and a third unrelated layout would add a panel and no claim.
    """
    angles_a = _base_angles()
    if context_geometry == 'rotated':
        return [_targets(angles_a),
                _targets(angles_a + np.deg2rad(ROTATION_DEG)),
                _targets(angles_a + np.deg2rad(NOVEL_ROTATION_DEG))]
    if context_geometry == 'independent':
        angles_b = _independent_angles(rng, 3.0 * noise_std / TARGET_RADIUS, angles_a, cue_index)
        return [_targets(angles_a), _targets(angles_b)]
    raise ValueError(f"context_geometry must be 'rotated' or 'independent', got "
                     f"{context_geometry!r}")


def _cue_start(rng: np.random.Generator) -> np.ndarray:
    """Where the shield appears on this trial — see CUE_START_MAX_R."""
    radius = CUE_START_MAX_R * TARGET_RADIUS * rng.uniform()
    angle = rng.uniform(0, 2 * np.pi)
    return radius * np.array([np.cos(angle), np.sin(angle)])


def _visible_attack(rng: np.random.Generator, target: np.ndarray, noise_std: float) -> np.ndarray:
    """The one attack the participant-view row shows: a draw from N(target, noise_std**2), rejected
    and redrawn until it is 1.3-2.2 sigma from the target.

    The clouds in the arenas are sampled honestly, and must be — a lopsided handful of draws is
    what a real sample looks like. This single dot is not a sample, it is a legend entry with one
    instance: a typical draw lands within a shield's width of the placement, and at frame scale the
    dot then hides under the shield and the last screen shows nothing happening. The band keeps it
    visible without inflating it into a miss.
    """
    for _ in range(200):
        attack = target + noise_std * rng.standard_normal(2)
        if 1.3 * noise_std < np.hypot(*(attack - target)) < 2.2 * noise_std:
            return attack
    return attack   # fell through: draw what we have rather than fail a schematic


# ── One arena panel ───────────────────────────────────────────────────────────

def _draw_arena_ring(ax, color) -> None:
    """The circle the targets live on, tinted with the context colour so panel identity is
    readable without reading the title.
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(TARGET_RADIUS * np.cos(theta), TARGET_RADIUS * np.sin(theta),
            color=color, linewidth=0.7, alpha=0.45, zorder=0)


def _draw_ghost_target(ax, target: np.ndarray, color) -> None:
    """Where the cue colour's target sits in the OTHER context — the rival explanation for this
    colour's outcomes, and the whole reason a single observation leaves the context uncertain.
    Drawn in the cue colour but faded and dashed, so it reads as "same colour, elsewhere" rather
    than as a sixth target.
    """
    ax.add_patch(Circle(tuple(target), 0.034, facecolor='none', edgecolor=color, alpha=0.7,
                        linewidth=0.9, linestyle=(0, (1.4, 1.4)), zorder=4))


def _draw_outcomes(ax, targets: np.ndarray, colors, noise_std: float,
                   rng: np.random.Generator) -> None:
    """A cloud of attack outcomes per colour: attack = target + noise_std * N(0, I2), exactly the
    dataset's sampling rule, sampled honestly (no re-centring — a real handful of draws is lopsided,
    which is the point). Open circles mark the (never displayed) target the cloud is centred on.

    The 1-sigma disc is tinted under each cloud. Without it a handful of draws at sigma=0.12 on a
    radius-0.5 arena read as scattered specks that the eye does not attach to any target; the tint
    supplies the grouping the sample alone does not, and states the "outcomes are drawn around the
    target" rule directly instead of leaving it to be inferred from the scatter.
    """
    for (x, y), c in zip(targets, colors):
        pts = np.array([x, y]) + noise_std * rng.standard_normal((N_OUTCOMES_PER_COLOR, 2))
        ax.add_patch(Circle((x, y), noise_std, facecolor=c, edgecolor='none', alpha=0.10,
                            zorder=1))
        ax.scatter(pts[:, 0], pts[:, 1], s=3.2, color=c, alpha=0.6, linewidths=0, zorder=2)
        ax.add_patch(Circle((x, y), 0.030, facecolor='none', edgecolor=c,
                            linewidth=0.8, alpha=0.95, zorder=3))


def _draw_cue_trial(ax, start: np.ndarray, target: np.ndarray, color, noise_std: float, *,
                    annotate: bool) -> None:
    """The trial being depicted: a shield of one colour appears somewhere near the middle of the
    canvas (`start`, sampled — see CUE_START_MAX_R), the agent moves it to where it believes that
    colour's target is (dashed arrow), and the attack lands as a draw around that target. Only the
    cue colour's 1-sigma disc gets an outline and the sigma label; the other four are tinted by
    _draw_outcomes and left unlabelled.
    """
    ax.add_patch(FancyArrowPatch(tuple(start), tuple(target), arrowstyle='-|>', mutation_scale=5,
                                 linewidth=0.8, color='0.35', linestyle=(0, (2.2, 1.6)),
                                 shrinkA=6, shrinkB=5, zorder=4))
    ax.add_patch(Circle(tuple(start), 0.055, facecolor=color, edgecolor='k', linewidth=0.6,
                        zorder=5))
    ax.add_patch(Circle(tuple(target), noise_std, facecolor='none', edgecolor='0.45',
                        linewidth=0.6, linestyle=(0, (1.2, 1.6)), zorder=3))
    if annotate:
        # Put the label on the far side of the shield from the arrow, whatever direction the
        # sampled start sent the arrow off in — a fixed "below the cue" offset lands on the arrow
        # for any start that sits below its target.
        away = start - target
        away = away / max(np.hypot(*away), 1e-9)
        ax.text(start[0] + 0.085 * away[0], start[1] + 0.085 * away[1], 'Shield cue',
                ha='center' if abs(away[0]) < 0.6 else ('left' if away[0] > 0 else 'right'),
                va='center' if abs(away[0]) >= 0.6 else ('bottom' if away[1] > 0 else 'top'),
                fontsize=5, color='0.05')
        tx, ty = target
        r = np.hypot(tx, ty)
        ax.text(tx + noise_std * tx / r, ty + noise_std * ty / r + 0.02, r'$\sigma$',
                ha='center', va='bottom', fontsize=5, color='0.45')


def _draw_rotation_hint(ax, ghost: np.ndarray, target: np.ndarray,
                        label: str | None = None) -> None:
    """Low-emphasis only: a thin grey arc just outside the arena, running from where the cue
    colour sits in the other context to where it sits here, with the angle in small grey type.
    Deliberately one colour, one hairline, drawn OUTSIDE the arena — this experiment tests context
    inference, not rotational transfer, and a bold "everything rotates by 60 deg" annotation would
    advertise the wrong claim. Drawn as a true circular arc (not a chord bow) so it reads as the
    layout turning, and swept the short way round via the wrapped angle difference.

    `label` overrides the printed angle. The novel-rotation panel passes a question mark: it is
    turned by NOVEL_ROTATION_DEG, but printing that number would answer the question the panel is
    there to ask.
    """
    a0 = np.arctan2(ghost[1], ghost[0])
    da = (np.arctan2(target[1], target[0]) - a0 + np.pi) % (2 * np.pi) - np.pi
    r = TARGET_RADIUS * 1.20
    t = a0 + da * np.linspace(0, 1, 40)
    # Low emphasis is about WHERE this is drawn (outside the arena, one colour) and how much of it
    # there is, not about being hard to see. At 0.6 pt in 0.68 grey the arc disappeared at panel
    # size; doubled and darkened it is legible and still subordinate to everything inside the ring.
    ax.plot(r * np.cos(t), r * np.sin(t), color='0.40', linewidth=1.2, zorder=2)
    ax.annotate('', xy=(r * np.cos(t[-1]), r * np.sin(t[-1])),
                xytext=(r * np.cos(t[-4]), r * np.sin(t[-4])),
                arrowprops=dict(arrowstyle='-|>', linewidth=1.2, color='0.40', mutation_scale=6))
    t_mid = a0 + da / 2
    text = label if label is not None else f'{abs(np.rad2deg(da)):.0f}' + r'$\degree$'
    ax.text(r * 1.18 * np.cos(t_mid), r * 1.18 * np.sin(t_mid), text,
            ha='center', va='center', fontsize=5, color='0.35')


def _arena_limit(noise_std: float) -> float:
    """Half-width of an arena panel: the ring, plus enough room for the tails of a cloud."""
    return TARGET_RADIUS + 2.6 * noise_std + 0.06


def _draw_context_panel(ax, targets: np.ndarray, ghosts: np.ndarray, *, ring_color,
                        noise_std: float, rng: np.random.Generator, cue_index: int,
                        cue_start: np.ndarray, rotation_hint: str | None,
                        annotate: bool) -> None:
    colors = _shield_colors()
    _draw_arena_ring(ax, ring_color)
    _draw_ghost_target(ax, ghosts[cue_index], colors[cue_index])
    if rotation_hint is not None:
        _draw_rotation_hint(ax, ghosts[cue_index], targets[cue_index],
                            label=rotation_hint or None)
    _draw_outcomes(ax, targets, colors, noise_std, rng)
    _draw_cue_trial(ax, cue_start, targets[cue_index], colors[cue_index], noise_std,
                    annotate=annotate)

    lim = _arena_limit(noise_std)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


# ── What the participant sees ─────────────────────────────────────────────────

#: The screens of one trial, in order. This is the whole of the participant's evidence: a coloured
#: shield, wherever they dragged it, and one dot. No ring, no targets, no block boundary, no
#: context label — everything else in this figure is the experimenter's view. The blank canvas the
#: trial starts from is not drawn: an empty rectangle spends a panel saying nothing, and the first
#: frame already shows the canvas, empty apart from the cue.
FRAME_LABELS = ('Shield cue', 'Move the shield', 'Attack outcome')

_FRAME_W = 1.0   # frame side, in the sequence axes' own units

#: Room for the arrow between two frames, in frame widths, by arena count. The frame row is
#: aspect-locked and limited by its ROW HEIGHT, not its width, so widening the gap spreads the
#: frames across more of the figure without shrinking them — which is the only reason this is tied
#: to the arena count: three arenas make the figure half again as wide, and a gap sized for a
#: two-arena figure leaves the row stranded in the middle of it. Not widened all the way to the
#: full width: past about 1.2 the arrows are longer than the screens they join and the three frames
#: stop reading as one sequence.
_FRAME_GAP_BY_ARENAS = {2: 0.70, 3: 1.20}


def _frame_xy(point: np.ndarray, cx: float) -> tuple[float, float]:
    """Arena coordinates -> coordinates inside the frame centred at `cx`.

    Positions are scaled (a target on the ring lands 0.40 from the frame centre, comfortably inside
    the 1.0-wide frame) but marker sizes are NOT: a frame is roughly a third of an arena panel, so a
    faithfully scaled shield would be a speck. The trial is the same one the first arena panel
    draws, so the two rows show one event from two points of view.
    """
    scale = 0.40 / TARGET_RADIUS
    return cx + scale * point[0], 0.5 + scale * point[1]


def _draw_shield(ax, xy: tuple[float, float], color, *, ghost: bool = False) -> None:
    if ghost:
        ax.add_patch(Circle(xy, 0.075, facecolor='none', edgecolor=color, alpha=0.45,
                            linewidth=0.7, linestyle=(0, (1.4, 1.4)), zorder=3))
    else:
        ax.add_patch(Circle(xy, 0.075, facecolor=color, edgecolor='k', linewidth=0.6, zorder=4))


def _draw_trial_frames(ax, start: np.ndarray, target: np.ndarray, attack: np.ndarray,
                       color, gap: float) -> None:
    """One trial as three screens, with the participant's own actions in between."""
    n = len(FRAME_LABELS)
    for i, label in enumerate(FRAME_LABELS):
        cx = 0.5 + i * (_FRAME_W + gap)
        ax.add_patch(FancyBboxPatch((cx - 0.5 * _FRAME_W, 0.0), _FRAME_W, _FRAME_W,
                                    boxstyle='round,pad=0,rounding_size=0.07',
                                    facecolor='0.965', edgecolor='0.78', linewidth=0.6, zorder=0))
        if i == 0:
            _draw_shield(ax, _frame_xy(start, cx), color)
        elif i == 1:
            _draw_shield(ax, _frame_xy(start, cx), color, ghost=True)
            ax.add_patch(FancyArrowPatch(_frame_xy(start, cx), _frame_xy(target, cx),
                                         arrowstyle='-|>', mutation_scale=4, linewidth=0.7,
                                         color='0.4', linestyle=(0, (2.0, 1.5)),
                                         shrinkA=4, shrinkB=4, zorder=2))
            _draw_shield(ax, _frame_xy(target, cx), color)
        elif i == 2:
            _draw_shield(ax, _frame_xy(target, cx), color)
            ax.scatter(*_frame_xy(attack, cx), s=7, color=color, linewidths=0, zorder=5)
        ax.text(cx, -0.11, label, ha='center', va='top', fontsize=5, color='0.15')
        if i < n - 1:
            ax.annotate('', xy=(cx + 0.5 * _FRAME_W + gap - 0.09, 0.5),
                        xytext=(cx + 0.5 * _FRAME_W + 0.09, 0.5),
                        arrowprops=dict(arrowstyle='-|>', linewidth=0.6, color='0.55',
                                        mutation_scale=4))

    total = n * _FRAME_W + (n - 1) * gap
    ax.set_xlim(-0.04, total + 0.04)
    ax.set_ylim(-0.34, 1.04)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


# ── Orchestration ─────────────────────────────────────────────────────────────

def _arena_specs(context_geometry: str, layouts: list[np.ndarray], cs) -> list[tuple]:
    """Per-arena (ghost layout, ring colour, title, rotation-hint label) — everything that differs
    between panels, kept out of the drawing loop.

    Ghosting rule: every panel ghosts the cue colour's target in the panel it is defined against.
    Under 'rotated' that is the 0 deg layout for both turned panels — the angle in the title is
    measured from there, so the arc has to start there. Panel 1 ghosts the 60 deg layout, since it
    has no earlier panel to be measured against and the rival explanation for its outcomes is the
    other trained context.

    Hint label: `''` means "print the arc's own angle", a string means print that instead, `None`
    means draw no arc. The novel panel prints a question mark — it is turned by
    NOVEL_ROTATION_DEG, but the panel exists to ask how far, not to answer.
    """
    if context_geometry == 'rotated':
        return [(layouts[1], cs.contextA, r'Rotation 0$\degree$', None),
                (layouts[0], cs.contextB, f'Rotation {ROTATION_DEG:.0f}' + r'$\degree$', ''),
                (layouts[0], NOVEL_CONTEXT_COLOR, r'Rotation ??$\degree$', r'??$\degree$')]
    return [(layouts[1], cs.contextA, 'Context A', None),
            (layouts[0], cs.contextB, 'Context B', None)]

def _draw_legend(fig, noise_std: float) -> None:
    """One shared marker key under all the arenas, so no panel has to carry its own labels.

    Three entries, not four: the ghost (the cue colour's target in the other context) is drawn in
    the panels but has no key. A fourth open circle differing from the third only by a dash pattern
    is not something a reader decodes from a legend at 5 pt — the marker has to be read in place,
    where it sits next to the arrow and the cue colour's cloud.
    """
    handles = [
        Line2D([], [], marker='o', linestyle='none', markersize=3.2, markerfacecolor='0.35',
               markeredgecolor='k', markeredgewidth=0.5, label='Shield cue'),
        Line2D([], [], marker='o', linestyle='none', markersize=1.8, markerfacecolor='0.35',
               markeredgecolor='none', label=f'Attack outcomes ($\\sigma$={noise_std:g})'),
        Line2D([], [], marker='o', linestyle='none', markersize=3.4, markerfacecolor='none',
               markeredgecolor='0.35', markeredgewidth=0.8, label='Target, this context'),
    ]
    fig.legend(handles=handles, loc='lower center', ncols=3, fontsize=5, frameon=False,
               handletextpad=0.3, columnspacing=1.0, borderpad=0.1)


def _save_schematic(fig, export_dir: Path, name: str, save_plots: bool, show_plots: bool) -> None:
    """Same save/show contract as rotation_curriculum_schematic._save_schematic — plain bools, no
    AnalysisParams, because this module measures nothing.
    """
    if save_plots:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / name
        fig.savefig(out, bbox_inches='tight')
        print(f'  Saved -> ./{out}')
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_task_schematic(export_dir: Path = EXPORT_ROOT / 'figures',
                        context_geometry: str = CONTEXT_GEOMETRY,
                        noise_std: float = NOISE_STD_DRAWN,
                        cue_index: int = 1,
                        save_plots: bool = True, show_plots: bool = True,
                        dpi: int = 160, seed: int = SEED) -> plt.Figure:
    """The task, drawn not measured.

    Row 1 — what the participant sees: one trial as four screens (blank canvas, shield cue,
    the move, the attack), and nothing else. Row 2 — the block strip: contexts alternate,
    switches unsignalled. Row 3 — the experimenter's arenas: ring, five targets, a cloud of
    outcomes per colour, the sampled shield start with a dashed arrow to that colour's target, and
    the cue colour's target in the other context ghosted in, which is the ambiguity a single
    outcome leaves. Row 1 draws the same trial as the first arena, so the two rows are one event
    seen twice rather than two cartoons.

    `context_geometry` decides how many arenas there are:

    * `'independent'` — two, 'Context A' and 'Context B', each colour's target redrawn uniformly
      on the circle so nothing about A's layout carries over to B.
    * `'rotated'` — three, the layout turned by 0 deg, ROTATION_DEG (the two trained contexts) and
      NOVEL_ROTATION_DEG. The third is titled with a question mark, not its angle: it is the
      transfer question, an untrained rotation the model has to identify from noisy outcomes.

    `noise_std` defaults to NOISE_STD_DRAWN (a real sweep level, chosen for legibility over the
    headline 0.20 — see the constant); `cue_index` picks which shield colour is shown mid-trial.
    """
    cs = Color_scheme()
    rng = np.random.default_rng(seed)
    layouts = _context_layouts(context_geometry, rng, noise_std, cue_index)
    n_arenas = len(layouts)
    cue_colour = _shield_colors()[cue_index]

    # One sampled start per arena, so the cue is not pinned to the centre and is not in the same
    # spot twice. The first one is also the trial the participant-view row replays.
    starts = [_cue_start(rng) for _ in range(n_arenas)]
    attack = _visible_attack(rng, layouts[0][cue_index], noise_std)

    # Width scales with the arena count so a third context does not shrink the other two; the
    # aspect-locked panels then saturate the width they are given.
    fig = plt.figure(figsize=FigSize.custom(fig_x_per_arena * n_arenas, fig_y), dpi=dpi,
                     layout='constrained')
    # Tight pads: the arena axes are aspect-locked, so any slack the layout engine leaves shows up
    # as a band of white between the rows rather than as bigger arenas.
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01)
    # The frame row is aspect-locked and much wider than it is tall, so its height ratio — not its
    # width — is what sets how big the canvases render. 0.68 is 0.52 (three frames at the old size)
    # plus the 30% that makes them read as screens rather than thumbnails.
    gs = fig.add_gridspec(3, n_arenas, height_ratios=[0.68, 0.18, 1.0])

    # No row title. The frame labels say what each screen is, and a heading over the row would be
    # the only one in the figure that names a point of view rather than a thing being drawn.
    ax_seq = fig.add_subplot(gs[0, :])
    _draw_trial_frames(ax_seq, starts[0], layouts[0][cue_index], attack, cue_colour,
                       gap=_FRAME_GAP_BY_ARENAS.get(n_arenas, 0.70))

    ax_strip = fig.add_subplot(gs[1, :])
    draw_block_bar(ax_strip, N_BLOCKS_IN_STRIP, cs.contextA, cs.contextB)
    # draw_block_bar leaves ylim=(0,1) with the bar in 0.13-0.46; crop to just the bar plus the
    # time arrow, or the empty upper half pushes the title off into whitespace.
    ax_strip.set_ylim(0.0, 0.50)
    ax_strip.set_title('Contexts alternate in blocks — switches unsignalled', fontsize=6, pad=2)
    x_end = N_BLOCKS_IN_STRIP * 0.78 - 0.08
    ax_strip.annotate('', xy=(x_end, 0.07), xytext=(-0.05, 0.07),
                      arrowprops=dict(arrowstyle='-|>', linewidth=0.6, color='0.45',
                                      mutation_scale=5))
    ax_strip.text(x_end, 0.05, 'Trials', ha='right', va='top', fontsize=5, color='0.45')

    for col, (ghosts, color, title, hint) in enumerate(_arena_specs(context_geometry, layouts, cs)):
        ax = fig.add_subplot(gs[2, col])
        _draw_context_panel(ax, layouts[col], ghosts, ring_color=color, noise_std=noise_std,
                            rng=rng, cue_index=cue_index, cue_start=starts[col],
                            rotation_hint=hint, annotate=col == 0)
        ax.set_title(title, fontsize=6, color=color)

    _draw_legend(fig, noise_std)

    name = f'schematic_task_{context_geometry}.pdf'
    _save_schematic(fig, export_dir, name, save_plots, show_plots)
    return fig


if __name__ == '__main__':
    plot_task_schematic()
