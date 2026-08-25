# mystyle.py
# import seaborn as sns
import matplotlib as mpl


# ── Figure / subplot size presets ─────────────────────────────────────────────
# All sizes are in inches.  These are single-panel dimensions; multiply the
# relevant axis by the number of panels when calling plt.subplots().
# See docs/figure_style.md for guidance.

class _FigSizeMeta(type):
    """
    Lets the FigSize presets respond to `FigSize.set_scale()` with no call-site changes.

    The presets are metaclass properties rather than plain class attributes so that
    `FigSize.small` is recomputed on every access. Without this, a scale switch would only
    affect `row()`/`grid()` and silently miss every direct `figsize=FigSize.small`.
    """
    _scale = 1.0
    _BASE = {
        'small':  (1.5, 1.5),   # compact square: summary stats, small insets
        'large':  (2.0, 2.0),   # full square: main result panels
        'wide':   (2.0, 1.5),   # landscape: time-series, learning curves
        'narrow': (1.5, 2.0),   # portrait: distributions, bar charts
        'tall':   (2.0, 1.5),   # alias for narrow-ish portrait
    }

    def __getattr__(cls, name):
        # Only reached when normal attribute lookup fails, so the presets must NOT also
        # be defined as class attributes on FigSize.
        if name in _FigSizeMeta._BASE:
            w, h = _FigSizeMeta._BASE[name]
            return (w * _FigSizeMeta._scale, h * _FigSizeMeta._scale)
        raise AttributeError(name)


class FigSize(metaclass=_FigSizeMeta):
    """
    Standard single-panel figure sizes (width × height, inches).

    Presets — `small`, `large`, `wide`, `narrow`, `tall` — are PAPER-READY dimensions,
    sized for a panel in a multi-panel figure. They are deliberately tiny. Do not enlarge
    a figure because the text looks small in a notebook; zoom the notebook instead.

    For development and exploration, call `FigSize.dev()` once at the top of a script to
    double every preset. Everything stays laid out identically, just legible on screen.
    Switch back with `FigSize.paper()` before exporting.

        import plot_style
        plot_style.set_plot_style()
        plot_style.FigSize.dev()      # 2x, for working on screen

    A figure that genuinely summarises a lot of data may be larger, but reach for
    `row()`/`grid()` first — and note that both grow linearly with panel count, so a
    per-model panel is usually the wrong structure. Prefer one panel with one line per
    model (coloured via `Color_scheme.get_model_color`) over N panels side by side.
    """

    #: Scale applied to every preset. 1.0 = paper-ready; 2.0 = comfortable on screen.
    DEV_SCALE = 2.0

    @staticmethod
    def set_scale(scale):
        """Set the global multiplier applied to every preset."""
        _FigSizeMeta._scale = float(scale)

    @staticmethod
    def get_scale():
        return _FigSizeMeta._scale

    @staticmethod
    def dev():
        """Enlarge all presets for on-screen exploration (does not change layout)."""
        FigSize.set_scale(FigSize.DEV_SCALE)

    @staticmethod
    def paper():
        """Restore paper-ready preset sizes."""
        FigSize.set_scale(1.0)

    @staticmethod
    def custom(width, height):
        """
        Scale-aware arbitrary size, in paper-ready inches.

        Use only when no preset fits — typically a width that must grow with the number of
        groups in a bar chart. Going through this instead of a literal `figsize=` keeps the
        dev/paper switch working.
        """
        s = _FigSizeMeta._scale
        return (width * s, height * s)

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


# ── Model → colour registry ───────────────────────────────────────────────────
# Keyed by normalised label (lowercased, punctuation stripped, whitespace collapsed) so
# that the display names used in run scripts — 'Oracle Z (one-hot)', 'NeuraGEM' — resolve
# without the caller having to know about normalisation. Add a row here when you add a
# model condition; every figure then picks the colour up automatically.
MODEL_COLORS = {
    'neuragem':            'tab:blue',
    'neuragem additive':   'tab:cyan',
    'rnn':                 'tab:green',
    'short horizon rnn':   'tab:green',
    'long horizon rnn':    'tab:red',
    'mrnn':                'tab:red',
    # Oracle family — distinct hues, since they are compared against each other directly.
    'oracle z':            'tab:orange',
    'oracle z one hot':    'tab:orange',
    'oracle z norm':       'tab:red',
    'oracle z cont':       'tab:purple',
    'bayesian':            'tab:brown',
    'naive':               'tab:purple',
}

# Assigned in order to labels not in MODEL_COLORS, so an unregistered condition still gets
# a stable colour instead of an error.
_FALLBACK_COLORS = ['tab:gray', 'tab:olive', 'tab:pink', 'tab:brown', 'tab:cyan']
_assigned_fallbacks = {}


def normalize_model_name(name):
    """'Oracle Z (one-hot)' → 'oracle z one hot'. Lowercase, depunctuate, collapse spaces."""
    s = ''.join(c if (c.isalnum() or c.isspace()) else ' ' for c in str(name).lower())
    return ' '.join(s.split())


def get_model_color(name):
    """
    Resolve a model label to its colour.

    Exact match on the normalised name first, then the longest registered prefix (so
    'NeuraGEM_10' inherits NeuraGEM's colour), then a stable fallback. Never raises — a
    missing entry should not stop a figure from being drawn.
    """
    key = normalize_model_name(name)
    if key in MODEL_COLORS:
        return MODEL_COLORS[key]

    prefixes = [k for k in MODEL_COLORS if key.startswith(k)]
    if prefixes:
        return MODEL_COLORS[max(prefixes, key=len)]

    if key not in _assigned_fallbacks:
        _assigned_fallbacks[key] = _FALLBACK_COLORS[
            len(_assigned_fallbacks) % len(_FALLBACK_COLORS)]
    return _assigned_fallbacks[key]


# ── Flanker condition palette ─────────────────────────────────────────────────
#
# Three orthogonal visual channels, so any flanker panel can be read without a legend:
#
#     hue          congruency   blue = congruent, red = incongruent
#     shade        distance     dark = near flankers, light = far flankers
#     fill/dash    outcome      solid/filled = correct, dashed/hollow = error
#
# Keeping outcome on fill rather than on hue is what makes the scheme work across mark
# types: a dashed line and a hollow bar carry the same meaning, and neither costs a
# colour. It also survives greyscale printing, since shade already varies.
#
# The hues are ColorBrewer RdBu — the pooled congruent/incongruent colours are the
# mid-tones, and near/far bracket each one darker/lighter, so a pooled line sits visually
# between its own two cells rather than beside them.

FLANKER_COLORS = {
    # pooled by congruency
    'cong':        '#4393c3',
    'incong':      '#d6604d',
    # congruency × distance
    'near_cong':   '#2166ac',   # dark blue
    'far_cong':    '#92c5de',   # light blue
    'near_incong': '#b2182b',   # dark red
    'far_incong':  '#f4a582',   # light red
    # outcome, when a panel pools over congruency and only contrasts correct vs error
    'correct':     '#4393c3',
    'error':       '#b2182b',
    # non-condition roles
    'neutral':     '#777777',
    'pass_':       '#2166ac',
    'fail':        '#b2182b',
    'null':        '#999999',
}

#: Canonical cell order for every congruency × distance panel: congruent pair first,
#: then incongruent, near before far within each. Grouping by congruency (rather than
#: interleaving near/far) puts the comparison the figure is about — the congruency
#: effect — side by side, and leaves the distance contrast as the within-pair step.
FLANKER_CELLS = [('near_cong', 'near\ncong'), ('far_cong', 'far\ncong'),
                 ('near_incong', 'near\nincong'), ('far_incong', 'far\nincong')]


def flanker_color(cong, near=None):
    """
    Colour for a flanker condition.

    cong : True congruent / False incongruent
    near : True near flankers / False far / None to pool over distance
    """
    stem = 'cong' if cong else 'incong'
    if near is None:
        return FLANKER_COLORS[stem]
    return FLANKER_COLORS[f'{"near" if near else "far"}_{stem}']


def outcome_style(correct, kind='bar', color=None):
    """
    Matplotlib kwargs marking a group as a correct-response or error cell.

    Outcome rides on fill, never on hue, so the same call works for every mark type:

        kind='bar'     filled vs hollow (coloured edge, no face)
        kind='line'    solid vs dashed
        kind='marker'  filled vs open marker face

    `color` is the condition colour from `flanker_color`; it is only needed for 'bar'
    and 'marker', which have to keep the hue on the edge when the face goes away.
    """
    if kind == 'line':
        return dict(linestyle='-' if correct else '--')
    if kind == 'marker':
        return dict(markerfacecolor=color if correct else 'none',
                    markeredgecolor=color)
    if kind == 'bar':
        if correct:
            return dict(color=color, alpha=0.65, linewidth=0)
        # Hollow: the face is dropped entirely rather than lightened, so an error bar can
        # never be mistaken for a far-flanker (lighter) cell of the same hue.
        return dict(facecolor='none', edgecolor=color, alpha=1.0, linewidth=1.2,
                    hatch=None)
    raise ValueError(f'unknown kind {kind!r}')


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
        """Resolve a model label to its colour — see the module-level MODEL_COLORS."""
        return get_model_color(model_name)
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

    # Shared figure-level labels (fig.supxlabel / fig.supylabel) and titles. Without
    # these they fall back to the matplotlib defaults (~12pt) and tower over a 6pt panel.
    mpl.rcParams['figure.labelsize'] = 6
    mpl.rcParams['axes.titlesize'] = 7
    mpl.rcParams['figure.titlesize'] = 8
    
    # Makes text appear as text and not as paths
    mpl.rcParams['pdf.fonttype'] = 42

    # Set the line width
    mpl.rcParams['lines.linewidth'] = 0.7

    # Set the dpi 
    mpl.rcParams['figure.dpi'] = 300
