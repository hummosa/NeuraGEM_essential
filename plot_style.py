# mystyle.py
# import seaborn as sns
import matplotlib as mpl

class Color_scheme:
    def __init__(self):
        self.short_horizon_rnn = 'tab:green'
        self.rnn = 'tab:green'
        self.long_horizon_rnn = 'tab:red'
        self.mrnn = 'tab:red'
        self.neuragem = 'tab:blue'
        self.neuragem_additive = 'tab:cyan'  # For additive NeuraGem
        self.ood_data = 'tab:orange'
        self.iid_data = 'tab:purple'
        self.bayesian = 'tab:brown'
        self.naive = 'tab:purple'

        # self.contextA = '#4424D6' # purple
        # self.contextB = '#FCCB1A' # yellow gold. # picked from https://www.w3schools.com/colors/colors_complementary.asp
        self.contextA = '#d9a528' # 
        self.contextB = '#a62b2a' # 

        self.violin_plot_width = 0.5

        self.linewidth = 0.7
        self.marker_size = 2
        self.marker = 'o'
        self.alpha_shaded_regions = 0.3

        self.panel_small_size = [1.2, 1.2]
        self.panel_large_size = [1.9, 1.9]
        self.panel_wide_size = [1.9, 1.2]
        self.panel_tall_size = [1.2, 1.9]

    def get_model_color(self, model_name):
        if model_name in ['short_horizon_rnn', 'long_horizon_rnn', 'neuragem']:
            return getattr(self, model_name)
        elif model_name in ['rnn', 'mrnn']:
            converted_name = 'short_horizon_rnn' if 'rnn' in model_name else 'long_horizon_rnn'
            return getattr(self, converted_name)
        else:
            print(f'ERROR: Model name {model_name} not found in color scheme. Valid options are: short_horizon_rnn or rnn, long_horizon_rnn or mrnn, and neuragem.')
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
