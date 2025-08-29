RANDOMIZATION_NAMES = {
    "biological": "Biological",
    "neuron_binned": "Neuron binned",
    "random_binned": "Random bin-wise",
    "unconstrained": "Unconstrained",
    "random_pruned": "Random pruned",
    "connection_pruned": "Connection-pruned",
} 

RANDOMIZATION_COLORS = {
    "biological": "black",
    "neuron_binned": "#2ca02c",
    "random_binned": "#d62728",
    "unconstrained": "#1f77b4",
    "random_pruned": "#ff7f0e",
    "connection_pruned": "#9467bd",
}

def get_randomization_colors(randomization_name: str):
    # reverse the randomization_names dictionary
    reverse_randomization_names = {v: k for k, v in RANDOMIZATION_NAMES.items()}
    return RANDOMIZATION_COLORS[reverse_randomization_names[randomization_name]]


# -----------------------------------------------------------------------------
# Central plotting style (Nature-like). Apply via apply_plot_style() in plots.
# -----------------------------------------------------------------------------

PLOT_STYLE_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
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
    """Apply the centralized matplotlib rcParams style.

    Parameters
    ----------
    overrides : dict, optional
        Dictionary with rcParams that should override the defaults.
    """
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(style="white", font_scale=1.4)

    style = PLOT_STYLE_PARAMS.copy()
    if overrides:
        style.update(overrides)
    mpl.rcParams.update(style)