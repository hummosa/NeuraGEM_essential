# %%
"""Run the four behavioral conditions and export compact figures.
Plots early training behavior for:
- NeuraGEM
- NeuraGEM with 10 latent Z units
- NeuraGEM with post-RNN gating .. actually more performant.
- Short-horizon RNN baseline
- Long-horizon RNN baseline
- NeuraGEM without latent decay
"""

if 'get_ipython' in globals():
    from IPython import get_ipython
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    
# supress all warnings and messages from matplotlib
import logging
logging.getLogger("matplotlib.font_manager").disabled = True

import plot_style
plot_style.set_plot_style()

from configs import *
from train_and_infer_functions import train_model
from functions_and_utils import plot_logger_panels, rasterize_and_save
cs = plot_style.Color_scheme()

COMPACT_PANELS = ["behavior", "latent_2d", "gradients",]
COMPACT_PANELS_BASELINE = ["behavior"]

def build_base_config():
    config = ContextualSwitchingTaskConfig(experiment_to_run="figure")
    config.default_std = 0.1
    config.log_weights = False
    config.save_model = False
    config.load_saved_model = False
    return config


def slugify(name):
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    return "_".join(part for part in slug.split("_") if part)


def run_condition(model_name, overrides, panel_order, title_color):
    config = build_base_config()
    for key, value in overrides.items():
        setattr(config, key, value)

    config.run_name = f"behavior_{slugify(model_name)}"

    print("Running:", model_name, "seed:", config.env_seed)
    logger, model, config, _ = train_model(
        config, seed=config.env_seed, save_models=False, load_models=False
    )

    fig = plot_logger_panels(
        logger,
        config,
        panel_order,
        x2=260,
        dpi=300,
        subplot_height=0.8,
        width=2.5,
        rasterize=True,
    )
    fig.suptitle(model_name, color=title_color, fontsize=8, y=0.995)

    out_path = f"{config.export_path}{config.dataset_name}_{slugify(model_name)}_behavior_rasterized.pdf"
    rasterize_and_save(
        fname=out_path,
        rasterize_list=None,
        fig=fig,
        dpi=300,
        savefig_kw={"bbox_inches": "tight", "transparent": True},
    )
    print("figure saved to:", out_path)
    return fig


if __name__ == "__main__":
    conditions = [
        {
            "model_name": "NeuraGEM",
            "overrides": {},
            "panel_order": COMPACT_PANELS + ['weights_grad_norm'],
            "title_color": cs.neuragem,
        },
        {
            "model_name": "NeuraGEM Post-RNN gating",
            "overrides": {"post_gating": True, "pre_gating": False},
            "panel_order": COMPACT_PANELS + ['weights_grad_norm'],
            "title_color": cs.neuragem,
        },
        {
            "model_name": "NeuraGEM 10 Z units",
            "overrides": {"latent_dims": [10], "input_size": 1},
            "panel_order": COMPACT_PANELS + ['weights_grad_norm'],
            "title_color": cs.neuragem,
        },
        {
            "model_name": "RNN^{short}",
            "overrides": {"LU_lr": 0.0},
            "panel_order": COMPACT_PANELS_BASELINE,
            "title_color": cs.short_horizon_rnn,
        },
        {
            "model_name": "RNN^{long}",
            "overrides": {"LU_lr": 0.0, "seq_len": 50},
            "panel_order": COMPACT_PANELS_BASELINE,
            "title_color": cs.long_horizon_rnn,
        },
        {
            "model_name": "NeuraGEM (no Z decay)",
            "overrides": {"l2_loss": 0.0},
            "panel_order": COMPACT_PANELS,
            "title_color": cs.neuragem,
        },
    ]

    for condition in conditions:
        run_condition(**condition)
