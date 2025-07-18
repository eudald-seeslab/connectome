import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patheffects import withStroke
import matplotlib.pyplot as plt
import numpy as np

from notebooks.visualization.activations_funcs import split_title
from utils.randomizers.randomizers_helpers import compute_total_synapse_length


def accuracy_comparison():
    # Task types (now on x-axis)
    tasks = [
        "Color\nDiscrimination",
        "Numerical\nDiscrimination",
        "Shape\nRecognition",
    ]

    # Network configurations (now as colors/legend)
    networks = [
        "Biological",
        "Random\nunconstrained",
        "Random\npruned",
        "Random\nbin-wise",
    ]

    # Made-up performance data for three tasks (percentage accuracy)
    # Reorganized: each row is a task, each column is a network
    task_data = {
        "Color\nDiscrimination": [100, 100, 100, 100],
        "Numerical\nDiscrimination": [84, 92, 91, 82],
        "Shape\nRecognition": [64, 69, 69, 60]
    }

    # Made-up error bars (standard error) - reorganized to match
    task_errors = {
        "Color\nDiscrimination": [0, 0, 0, 0],
        "Numerical\nDiscrimination": [0, 0, 0, 0.01],
        "Shape\nRecognition": [0, 0.01, 0, 0.04]
    }

    # Nature color palette for networks
    colors = ["#4878D0", "#6ACC64", "#EE854A", "#D65F5F"]

    # Set width of bars
    bar_width = 0.18  # Slightly smaller to accommodate 4 networks
    capsize = 2
    index = np.arange(len(tasks))

    # Create the figure with Nature-compatible dimensions
    fig3, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Create grouped bars - now grouped by task, colored by network
    for i, network in enumerate(networks):
        # Get data for this network across all tasks
        network_data = [task_data[task][i] for task in tasks]
        network_errors = [task_errors[task][i] for task in tasks]

        bars = ax.bar(
            index + i * bar_width,
            network_data,
            bar_width,
            yerr=network_errors,
            label=network,
            color=colors[i],
            alpha=0.9,
            capsize=capsize,
            ecolor="black",
            error_kw={"elinewidth": 1},
        )

    # Add horizontal line for chance level (50% for binary classification)
    ax.axhline(y=50, linestyle="--", color="#666666", alpha=0.5, linewidth=1)

    # Add text label for chance level
    ax.text(len(tasks) - 1.35, 45, "Chance level", fontsize=10, color="#666666",
            bbox=dict(facecolor='white', edgecolor='#666666', boxstyle='round,pad=0.5', alpha=0.8))

    # Add labels and custom x-axis tick labels
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)  # Slightly higher to accommodate error bars
    ax.set_xticks(index + bar_width * 1.5)  # Center the labels between groups
    ax.set_xticklabels(tasks, fontsize=12, rotation=0)  # No rotation needed now

    # Add a legend
    ax.legend(fontsize=12, loc="lower right", framealpha=0.9)

    # Adjust layout
    plt.tight_layout()

    return fig3


def efficiency_comparison(neuron_position_data, connections_dict):

    # Compute wiring lenghts for each network
    original_length = compute_total_synapse_length(connections_dict["Biological"], neuron_position_data)
    unconstrained_length = compute_total_synapse_length(connections_dict["Random unconstrained"], neuron_position_data)
    pruned_length = compute_total_synapse_length(connections_dict["Random pruned"], neuron_position_data)
    binned_length = compute_total_synapse_length(connections_dict["Random bin-wise"], neuron_position_data)

    # Total synaptic length (m)
    wiring_length = [a / 1e9 for a in [original_length, unconstrained_length, pruned_length, binned_length]]
    # Mean accuracy (%) (from the sheets)
    accuracy = [84, 92, 91, 82]

    # Calculate efficiency (accuracy per unit wiring)
    efficiency = [acc / length * 100 for acc, length in zip(accuracy, wiring_length)]

    # Colors that work well for Nature (colorblind-friendly, print-friendly)
    colors = ["#0173B2", "#DE8F05", "#029E73", "#D55E00"]

    # Create the figure with Nature-compatible dimensions
    fig2, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)  # Nature's single column width

    # Create scatter plot with varying point sizes based on efficiency
    sizes = [e ** 2 for e in efficiency]  # Scale efficiency for better visualization
    scatter = ax.scatter(wiring_length, accuracy, s=sizes, c=colors, alpha=0.8, zorder=3)

    titles = list(connections_dict.keys())
    # split a title in two lines if it's too long
    titles = [title.replace(" ", "\n") if len(title) > 15 else title for title in titles]

    # Add labels with a white outline for better visibility
    for i, txt in enumerate(titles):
        if txt == split_title("Random bin-wise"):
            va = "top"
            x_offset = -20
            y_offset = -5
        elif txt == split_title("Random unconstrained"):
            va = "bottom"
            x_offset = -55
            y_offset = 5
        else:
            va = "bottom"
            x_offset = -25
            y_offset = 5

        text = ax.annotate(
            txt,
            (wiring_length[i], accuracy[i]),
            fontsize=8,
            ha="left",
            va=va,
            xytext=(x_offset, y_offset),
            textcoords="offset points",
        )
        text.set_path_effects([withStroke(linewidth=3, foreground="white")])

    # Add x and y-axis labels with units
    ax.set_xlabel("Total synaptic wiring length (m)", fontsize=9)
    ax.set_ylabel("Classification accuracy (%)", fontsize=9)

    # Set axis limits with some padding
    ax.set_xlim(1000, 3500)
    ax.set_ylim(70, 100)

    # Make tick labels smaller
    ax.tick_params(axis='both', which='major', labelsize=7)

    # Add grid for readability (light grid typical for Nature figures)
    ax.grid(linestyle="--", alpha=0.3, zorder=0)


    # Create legend for the point sizes representing efficiency
    # Create custom handles for legend
    class SizedPatchHandler(HandlerPatch):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def create_artists(
            self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
        ):
            size = orig_handle.get_width()
            p = mpatches.Circle(
                (0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent),
                size / 3,
                edgecolor=orig_handle.get_edgecolor(),
                facecolor=orig_handle.get_facecolor(),
                transform=trans,
            )
            return [p]


    # Create legend handles
    efficiency_levels = [min(efficiency), max(efficiency)]
    legend_sizes = [e * 2 for e in efficiency_levels]

    handles = [
        mpatches.Rectangle(
            (0, 0), legend_sizes[0], legend_sizes[0], facecolor="gray", alpha=0.5
        ),
        mpatches.Rectangle(
            (0, 0), legend_sizes[1], legend_sizes[1], facecolor="gray", alpha=0.5
        ),
    ]

    # Add legend with custom handler
    ax.legend(
        handles,
        [f"Lower efficiency", f"Higher efficiency"],
        title="Accuracy/Wiring Ratio",
        handler_map={mpatches.Rectangle: SizedPatchHandler()},
        loc="lower right",
        fontsize=7,
        title_fontsize=8,
    )

    # Adjust layout and save
    plt.tight_layout()
    fig2.subplots_adjust(right=0.98, top=0.95)

    return fig2