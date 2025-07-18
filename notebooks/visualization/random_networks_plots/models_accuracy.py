import matplotlib.pyplot as plt
import numpy as np



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
