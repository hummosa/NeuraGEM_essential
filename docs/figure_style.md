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

| Name | Size | Use when |
|---|---|---|
| `FigSize.small` | 1.5 × 1.5 | compact summary, small insets |
| `FigSize.large` | 3.0 × 3.0 | main square result panels |
| `FigSize.wide` | 3.0 × 1.5 | time-series, learning curves |
| `FigSize.narrow` | 1.5 × 3.0 | distributions, bar charts |
| `FigSize.tall` | 2.0 × 3.0 | portrait with a bit more width |

### Multi-panel helpers

```python
# 1 × N row of panels (default panel = wide)
fig, axes = plt.subplots(1, 4, figsize=FigSize.row(4))
fig, axes = plt.subplots(1, 3, figsize=FigSize.row(3, FigSize.large))

# R × C grid (default panel = large)
fig, axes = plt.subplots(2, 3, figsize=FigSize.grid(2, 3))
fig, axes = plt.subplots(3, 2, figsize=FigSize.grid(3, 2, FigSize.wide))
```

## Rules of thumb

- **Never hard-code `figsize`** — always use a `FigSize` preset so that resizing is a one-line change.
- Keep figures small by default.  Figures that look right in a Jupyter notebook at 100% zoom are usually too big for a paper panel.  If something looks too small in the notebook, zoom in rather than enlarging the figure.
- Match panel aspect to content: time-series → `wide`, scatter / 2D → `large` or `small`, bar / distribution → `narrow`.

## Colors and line styles

Use `plot_style.Color_scheme` for model colors.  Instantiate once per script:

```python
from plot_style import Color_scheme
cs = Color_scheme()
# cs.neuragem, cs.rnn, cs.ood_data, …
```

The panel size fields (`panel_small_size`, etc.) that used to live in `Color_scheme` have been removed — use `FigSize` instead.

