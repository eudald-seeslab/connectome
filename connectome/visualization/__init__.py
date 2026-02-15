"""Visualization module for connectome analysis."""

# Basic plot functions re-exported from trainyourfly
from trainyourfly.plots.plots import (
    plot_weber_fraction,
    plot_accuracy_per_value,
    plot_accuracy_per_colour,
    plot_contingency_table,
    plot_results,
    guess_your_plots,
)

# Research-specific visualization
from .plot_config import (
    RANDOMIZATION_NAMES,
    RANDOMIZATION_COLORS,
    PLOT_STYLE_PARAMS,
    apply_plot_style,
    get_randomization_colors,
    split_title,
)

from .synapse_distributions import (
    plot_synapse_length_distributions,
    plot_synapse_counts_histogram,
)

from .activation_plots import (
    plot_activation_statistics,
    get_active_neuron_bounds,
    visualize_steps_separated_compact,
    plot_3d_activation_compact,
)

from .models_accuracy import (
    grouped_accuracy_comparison,
    grouped_accuracy_comparison_4groups,
    task_accuracy_comparison,
)

__all__ = [
    # plot_config
    "RANDOMIZATION_NAMES",
    "RANDOMIZATION_COLORS",
    "PLOT_STYLE_PARAMS",
    "apply_plot_style",
    "get_randomization_colors",
    "split_title",
    # synapse_distributions
    "plot_synapse_length_distributions",
    "plot_synapse_counts_histogram",
    # activation_plots
    "plot_activation_statistics",
    "get_active_neuron_bounds",
    "visualize_steps_separated_compact",
    "plot_3d_activation_compact",
    # models_accuracy
    "grouped_accuracy_comparison",
    "grouped_accuracy_comparison_4groups",
    "task_accuracy_comparison",
    # plots (from trainyourfly)
    "plot_weber_fraction",
    "plot_accuracy_per_value",
    "plot_accuracy_per_colour",
    "plot_contingency_table",
    "plot_results",
    "guess_your_plots",
]
