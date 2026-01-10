"""Re-export plot configuration from the main visualization module."""

from connectome.visualization.plot_config import (
    RANDOMIZATION_NAMES,
    RANDOMIZATION_COLORS,
    PLOT_STYLE_PARAMS,
    apply_plot_style,
    get_randomization_colors,
    split_title,
)

__all__ = [
    "RANDOMIZATION_NAMES",
    "RANDOMIZATION_COLORS",
    "PLOT_STYLE_PARAMS",
    "apply_plot_style",
    "get_randomization_colors",
    "split_title",
]

