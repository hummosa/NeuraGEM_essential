# Figure Style Guide

## Standard figure sizes

All figure sizing goes through `plot_style.FigSize`.  Import it at the top of any plotting file:

```python
from plot_style import FigSize
```

### Font sizes
**Never pass `fontsize=` to standard elements** (axis labels, tick labels, legends, titles).  Let `plot_style.set_plot_style()` set all of those via `rcParams`.  The only acceptable exception is a one-off annotation that deliberately deviates from the global style (e.g., a single data label overlaid on a panel).

### Titles
**Default: no title.**  x/y labels carry the message.  A title is only justified when the content literally cannot be read without it — e.g., a panel in a grid where both axes are shared and the panel identity is otherwise ambiguous.  Never use titles to describe the method, the model, or the parameters — those belong in the caption or the legend.  `ax.set_title(...)` and `fig.suptitle(...)` should be rare exceptions, not the default.

### Figure sizes — the most common mistake
Figures look large in Jupyter because the notebook scales them up.  Do not increase `figsize` to make text look bigger — **zoom the notebook instead**.  The sizes in `FigSize` are calibrated for print/PDF at the actual panel dimensions.

A telltale sign you've made the figure too big: the text looks tiny relative to the panel.  That happens when you use large `figsize` values (e.g. 6×4) and the font, set to 6–7 pt by `plot_style`, becomes microscopic at the rendered DPI.  Use the presets and trust them.  If you're tempted to write `figsize=(6, 4)` or larger, stop and use `FigSize.large` or `FigSize.wide` instead.

### Single-panel presets (width × height, inches)

These are **paper-ready** dimensions — one panel of a multi-panel figure. They are
deliberately tiny.

| Name | Size | Use when |
|---|---|---|
| `FigSize.small` | 1.5 × 1.5 | compact summary, small insets |
| `FigSize.large` | 2.0 × 2.0 | main square result panels |
| `FigSize.wide` | 2.0 × 1.5 | time-series, learning curves |
| `FigSize.narrow` | 1.5 × 2.0 | distributions, bar charts |
| `FigSize.tall` | 2.0 × 1.5 | portrait with a bit more width |

### Development scale — `FigSize.dev()`

Paper sizes are hard to read while exploring. Call `FigSize.dev()` once at the top of a
script to double every preset; layout is identical, just legible on screen.

```python
import plot_style
plot_style.set_plot_style()
plot_style.FigSize.dev()      # 2x — everything scales, including row()/grid()/custom()
...
plot_style.FigSize.paper()    # back to paper-ready before exporting
```

The presets are metaclass properties, so the scale applies to *every* access — a direct
`figsize=FigSize.small` scales just as `FigSize.row(...)` does. `FigSize.DEV_SCALE`
controls the multiplier (default 2.0).

### Judge size in inches, not pixels

A figure rendered inline in VS Code / Jupyter is drawn at `figure.dpi` (300, set by
`set_plot_style()`) and then **again at 2× for retina displays**. A correctly-sized
6.0 × 1.5 in figure therefore arrives as a 3599 × 889 px image, which looks enormous while
being exactly right. Always check with:

```python
fig.get_size_inches()      # the number that matters
```

If a figure is genuinely too large, that call will say so. Pixel dimensions will not.

### The real symptom to watch: text-to-ink ratio

If the text looks *large* relative to the plotted data, the figure is too **small** — or,
more often, the drawing area has been squeezed by something else: long rotated tick
labels, an `aspect='equal'` constraint, or a legend anchored outside the axes. Fix the
squeeze, do not enlarge the figure.

### When no preset fits — `FigSize.custom()`

```python
# Bar chart whose width must grow with the number of groups
fig, ax = plt.subplots(figsize=FigSize.custom(max(1.5, 0.30 * n_bars + 0.6), 1.5))
```

`custom()` takes paper-ready inches and applies the current scale. Use it instead of a
literal `figsize=` so the dev/paper switch keeps working.

### Multi-panel helpers

```python
# 1 × N row of panels (default panel = wide)
fig, axes = plt.subplots(1, 4, figsize=FigSize.row(4))
fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3, FigSize.large))

# R × C grid (default panel = large)
fig, axes = plt.subplots(2, 3, figsize=FigSize.grid(2, 3))
fig, axes = plt.subplots(3, 2, figsize=FigSize.grid(3, 2, FigSize.wide))
```

### Bar rows, legends and shared scales — `flanker_figure_utils`

The flanker figures build on three helpers that live next to the panel primitives in
`flanker_figure_utils.py`, because they encode layout decisions this guide asks for:

```python
from flanker_figure_utils import bar_row, compact_legend, share_ylim

# One row of bar panels; each panel gets width in proportion to the tick labels it
# carries, so a two-bar panel is not padded out to the width of a four-bar one.
fig, axes = bar_row([(groups_a, dict(ylabel='Accuracy', baseline=0.5)),
                     (groups_b, dict(ylabel='RT (timesteps)'))])

compact_legend(ax, loc='lower center', ncol=3)   # frameless, tight, rcParams font size
share_ylim(axes[1], axes[2])                     # one scale for panels that measure the same thing
```

`share_ylim` is applied after plotting rather than through `sharey=` at subplot creation,
so a row can share a scale within a pair of panels without dragging along neighbours that
measure something else. It drops the repeated y-label and tick labels from all but the
first panel, which only reads correctly when the panels are side by side.

Use it wherever two panels answer the same question — RT after each history cell,
accuracy in stage 1 vs stage 2 — and *not* when the two quantities differ by an order of
magnitude, since the smaller panel then flattens into its baseline.

### Figures in the interactive window

`save()` writes the PDF, then displays the figure when it detects a Jupyter / VS Code
kernel, so running a figure script in the interactive window shows the panels instead of
only printing export paths. Batch runs (`python flanker_sweep_figures.py`) are unchanged.
Paper sizes still look small on screen — that is what `FigSize.dev()` is for.

## Rules of thumb

- **Never hard-code `figsize`** — always use a `FigSize` preset (or `FigSize.custom`) so that resizing is a one-line change.
- Keep figures small by default.  Figures that look right in a Jupyter notebook at 100% zoom are usually too big for a paper panel.  If something looks too small in the notebook, zoom in — or call `FigSize.dev()` — rather than enlarging the figure.
- Match panel aspect to content: time-series → `wide`, scatter / 2D → `large` or `small`, bar / distribution → `narrow`.
- **Avoid one-panel-per-model layouts.**  `row(n)`/`grid(...)` grow linearly, so four models becomes a 6-inch figure that fits neither a paper column nor a slide.  Prefer a single panel with one line per model, distinguished by colour.  Reach for a per-model panel only when the content genuinely cannot overlay (scatter clouds, arenas) — and then use `small`.
- **Watch legends anchored outside the axes.**  With `bbox_inches='tight'` they inflate the *saved* file well beyond the nominal `figsize`, which is a common reason a figure is larger than its preset suggests.  Prefer a compact in-axes legend (`frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.25`), or one horizontal row above the axes via `bbox_to_anchor=(0, 1, 1, 0.12), mode='expand', ncol=n`.
- Encode a second factor as **fill opacity or line style**, not a second colour dimension.  It keeps the legend to a couple of greyscale swatches instead of one entry per model × factor.
- A figure that genuinely summarises a lot of data may be bigger — but that is the exception, and worth a comment saying why.

## Colors — model identity

Models keep the same colour in every figure, so a reader can track a condition across a
multi-panel figure. The registry is `plot_style.MODEL_COLORS`, keyed by *normalised* label
(lowercased, punctuation stripped, whitespace collapsed), resolved by:

```python
from plot_style import get_model_color
get_model_color('Oracle Z (one-hot)')   # -> 'tab:orange'
get_model_color('NeuraGEM_10')          # -> 'tab:blue'  (longest registered prefix)
get_model_color('Brand New Condition')  # -> stable fallback colour, never raises
```

Resolution order: exact normalised match → longest registered prefix → a stable fallback
from a small palette. It never raises, so an unregistered condition cannot stop a figure
from being drawn.

**When you add a model condition, add a row to `MODEL_COLORS`.** Every figure that goes
through `get_model_color` then picks it up with no further change.

`Color_scheme.get_model_color(name)` delegates here, so existing call sites keep working.

Other `Color_scheme` fields remain for non-model roles:

```python
from plot_style import Color_scheme
cs = Color_scheme()
# cs.contextA, cs.contextB, cs.ood_data, cs.iid_data, cs.linewidth, …
```

The panel size fields (`panel_small_size`, etc.) that used to live in `Color_scheme` have been removed — use `FigSize` instead.

## Colors — flanker task conditions

The flanker figures use one scheme, defined in `plot_style.FLANKER_COLORS`, so a panel can
be read without a legend. Three orthogonal channels:

| channel | factor | encoding |
|---|---|---|
| hue | congruency | blue = congruent, red = incongruent |
| shade | distance | dark = near flankers, light = far flankers |
| fill | outcome | filled bar / solid line = correct, hollow bar / dashed line = error |

```python
from plot_style import FLANKER_COLORS, FLANKER_CELLS, flanker_color, outcome_style

flanker_color(cong=False, near=True)        # -> near-incongruent, dark red
flanker_color(cong=True,  near=None)        # -> congruent pooled over distance
outcome_style(correct=False, kind='bar', color=c)   # hollow bar kwargs
outcome_style(correct=False, kind='line')           # dashed
```

Putting outcome on **fill** rather than on a third hue is what makes the scheme work
across mark types: a dashed line and a hollow bar say the same thing, neither costs a
colour, and both survive greyscale printing since shade is already doing work. The bar
primitives take a `hollow=[bool, ...]` list, one entry per group —
`flanker_analyses.plot_scalar_bars` and `flanker_figure_utils.bars_with_seeds`.

The hues are ColorBrewer RdBu. Pooled congruent/incongruent are the mid-tones and the
near/far cells bracket each one darker/lighter, so a pooled line sits visually between its
own two cells.

**Cell order.** `FLANKER_CELLS` fixes the order for every congruency × distance panel:
congruent pair first, then incongruent, near before far within each. Grouping by
congruency puts the contrast the figure is about side by side and leaves distance as the
within-pair step. Use it rather than re-listing the cells per figure.

Where a panel has no distance factor, shade is free to carry something else — the
trial-history panels use it for the *previous* trial's congruency. That is fine as long as
the panel says so in a comment; the rule is one meaning per panel, not one meaning forever.

## Font sizes set globally

`set_plot_style()` covers axis labels (6pt), ticks (6pt), legend (6pt), base font (7pt),
**and** the figure-level labels that are easy to miss: `figure.labelsize` (6pt, used by
`fig.supxlabel`/`supylabel`), `axes.titlesize` (7pt), `figure.titlesize` (8pt). Without
those three, super-labels fall back to the matplotlib default of ~12pt and tower over a
6pt panel. Never pass `fontsize=` to work around it — fix the rcParam.

