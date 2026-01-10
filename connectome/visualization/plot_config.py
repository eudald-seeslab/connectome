"""Configuration for randomization plots."""

import matplotlib as mpl
import seaborn as sns


RANDOMIZATION_NAMES = {
    "biological": "Biological",
    "neuron_binned": "Neuron binned",
    "random_binned": "Binned",
    "unconstrained": "Unconstrained",
    "random_pruned": "Random pruned",
    "connection_pruned": "Connection-pruned",
}

RANDOMIZATION_COLORS = {
    # Singular, destaca sobre grocs i vermells
    "biological": "#4c6ef5",        # blau porpra profund

    # Binned – paleta groguenca
    "neuron_binned": "#e9c46a",     # groc mostassa clar
    "random_binned": "#cfae3b",     # mostassa més fosc / daurat

    # Unconstrained / pruned – paleta vermellosa
    "unconstrained": "#8b1e1e",     # vermell fosc sobri
    "random_pruned": "#b23a48",     # vermell gerd apagat
    "connection_pruned": "#e07a5f", # vermell-terracota clar
}


def get_randomization_colors(randomization_name: str) -> str:
    """Get the color for a randomization strategy by its display name."""
    reverse_randomization_names = {v: k for k, v in RANDOMIZATION_NAMES.items()}
    return RANDOMIZATION_COLORS[reverse_randomization_names[randomization_name]]


PLOT_STYLE_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 1,
    "grid.linewidth": 0.5,
    "lines.linewidth": 2,
    "lines.markersize": 5,
}


def apply_plot_style(overrides: dict | None = None):
    """Apply the centralized matplotlib rcParams style."""
    sns.set_theme(style="white", font_scale=1.4)
    style = PLOT_STYLE_PARAMS.copy()
    if overrides:
        style.update(overrides)
    mpl.rcParams.update(style)


def split_title(title: str, max_length: int = 15) -> str:
    """Split long titles for better display in plots."""
    if len(title) > max_length:
        return title.replace("-", "\n").replace(" ", "\n")
    return title

