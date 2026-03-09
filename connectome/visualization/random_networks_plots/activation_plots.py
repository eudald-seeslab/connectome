"""Re-export activation plots from the main visualization module."""

from connectome.visualization.activation_plots import (
    plot_activation_statistics,
    get_active_neuron_bounds,
    visualize_steps_separated_compact,
    plot_3d_activation_compact,
)

__all__ = [
    "plot_activation_statistics",
    "get_active_neuron_bounds",
    "visualize_steps_separated_compact",
    "plot_3d_activation_compact",
]

