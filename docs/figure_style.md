# Figure Style Guide

## Standard figure sizes

All figure sizing goes through `plot_style.FigSize`.  Import it at the top of any plotting file:

```python
from plot_style import FigSize
```

### Font sizes
Stop specifying fonesize for standard elements like titles axes labels and legends so those can be standarized by the plot_style file. 

### Titles
You have a tendency to add titles to everything. Mostly uncessary. Most of the time the x and y labels tell the story like it should. Only need titles for things than cannot be disambuguated otherwise. Definitely not to add methods details or stuff like that. 

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

