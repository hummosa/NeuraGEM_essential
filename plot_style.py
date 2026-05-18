# mystyle.py
# import seaborn as sns
import matplotlib as mpl


# ── Figure / subplot size presets ─────────────────────────────────────────────
# All sizes are in inches.  These are single-panel dimensions; multiply the
# relevant axis by the number of panels when calling plt.subplots().
# See docs/figure_style.md for guidance.

class FigSize:
    """Standard single-panel figure sizes (width × height, inches)."""
    small  = (1.5, 1.5)   # compact square: summary stats, small insets
    large  = (2.0, 2.0)   # full square: main result panels
    wide   = (2.0, 1.5)   # landscape: time-series, learning curves
    narrow = (1.5, 2.0)   # portrait: distributions, bar charts
    tall   = (2.0, 1.5)   # alias for narrow-ish portrait
    # add 0.5 inch to height to make room for titles:
    # small = (small[0], small[1] + 1)
    # large = (large[0], large[1] + 1)
    # wide  = (wide[0], wide[1] + 1)
    # narrow = (narrow[0], narrow[1] + 1)
    # tall   = (tall[0], tall[1] + 1)

    @staticmethod
    def row(n, panel=None):
        """Return figsize for a 1×n row of panels (default panel = wide)."""
        pw, ph = panel or FigSize.wide
        return (pw * n, ph)

    @staticmethod
    def grid(rows, cols, panel=None):
        """Return figsize for a rows×cols grid (default panel = large)."""
        pw, ph = panel or FigSize.large
        return (pw * cols, ph * rows)


class Color_scheme:
    def __init__(self):
        self.short_horizon_rnn = 'tab:green'
        self.rnn = 'tab:green'
        self.long_horizon_rnn = 'tab:red'
        self.mrnn = 'tab:red'
        self.neuragem = 'tab:blue'
        self.neuragem_additive = 'tab:cyan'
        self.ood_data = 'tab:orange'
        self.iid_data = 'tab:purple'
        self.bayesian = 'tab:brown'
        self.naive = 'tab:purple'

        self.contextA = '#d9a528'
        self.contextB = '#a62b2a'

        self.violin_plot_width = 0.5
        self.linewidth = 0.7
        self.marker_size = 2
        self.marker = 'o'
        self.alpha_shaded_regions = 0.3

    def get_model_color(self, model_name):
        if model_name in ['short_horizon_rnn', 'long_horizon_rnn', 'neuragem']:
            return getattr(self, model_name)
        elif model_name in ['rnn', 'mrnn']:
            converted_name = 'short_horizon_rnn' if 'rnn' in model_name else 'long_horizon_rnn'
            return getattr(self, converted_name)
        else:
            print(f'ERROR: Model name {model_name} not found in color scheme.')
            return 'tab:gray'
def set_plot_style():
    # sns.set(font_scale=0.8)  # Adjust font scale
    # sns.set_style('white', {'axes.linewidth': 0.5})  # Remove grid
    
    mpl.rcParams['xtick.bottom'] = True
    mpl.rcParams['ytick.left'] = True
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['xtick.major.width'] = 0.85
    mpl.rcParams['ytick.major.size'] = 3
    mpl.rcParams['ytick.major.width'] = 0.9

    # Remove spines on right and top
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    # Make axis lines thinner
    mpl.rcParams['axes.linewidth'] = 0.7

    # Set the font
    mpl.rcParams['font.family'] = 'sans-serif'

    # Set the font size. Make fonts smaller for paper
    mpl.rcParams['font.size'] = 7

    # Set the font size for the legend
    mpl.rcParams['legend.fontsize'] = 6

    # Set the font size for x and y axis labels
    mpl.rcParams['axes.labelsize'] = 6

    # Set the fone size for the x and y tick labels
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    
    # Makes text appear as text and not as paths
    mpl.rcParams['pdf.fonttype'] = 42

    # Set the line width
    mpl.rcParams['lines.linewidth'] = 0.7
