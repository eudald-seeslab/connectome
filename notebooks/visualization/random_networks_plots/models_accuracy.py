from typing import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_config import apply_plot_style, get_randomization_colors, RANDOMIZATION_NAMES


def grouped_accuracy_comparison(df: pd.DataFrame):
    """
    Plots classification accuracy in two groups: length-constrained and unconstrained.

    This creates a side-by-side bar plot panel designed to show that:
    1. Within length-constrained networks, the biological connectome is optimal.
    2. Higher accuracy is achievable by relaxing length constraints, but at the
       cost of biological plausibility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with accuracy results, matching the format of
        'notebooks/visualization/data/randomizations_seeds.csv'.

    Returns
    -------
    matplotlib.figure.Figure
        A Nature-style figure object containing the panel of plots.
    """

    # --- Data Preparation ---------------------------------------------------
    # Define the groups based on expected wiring length
    # These names must match the standardized names from the CSV
    constrained_group = [
        RANDOMIZATION_NAMES["biological"],
        RANDOMIZATION_NAMES["neuron_binned"],
        RANDOMIZATION_NAMES["random_binned"],
    ]
    unconstrained_group = [
        RANDOMIZATION_NAMES["unconstrained"],
        RANDOMIZATION_NAMES["random_pruned"],
        RANDOMIZATION_NAMES["connection_pruned"],
    ]

    # Check if all strategies are categorized
    all_strategies = set(df["Randomization strategy"])
    categorized = set(constrained_group) | set(unconstrained_group)
    if all_strategies != categorized:
        uncategorized = all_strategies - categorized
        print(
            f"Warning: Uncategorized strategies found and will be ignored: {uncategorized}"
        )
        df = df[df["Randomization strategy"].isin(categorized)]

    # Drop rows with missing strategy
    df = df.dropna(subset=["Randomization strategy"])

    # Pre-process accuracy table (convert percentage strings to floats)
    replicate_cols = [
        c for c in df.columns if c not in {"Randomization strategy", "Sweep name"}
    ]
    for col in replicate_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace("", np.nan)
            .astype(float)
            * 100
        )

    # Calculate means and SEMs
    stats = df.set_index("Randomization strategy")[replicate_cols].agg(
        ["mean", "sem"], axis=1
    )

    # Filter and sort data for each group for plotting
    constrained_stats = stats.loc[stats.index.isin(constrained_group)].sort_values(
        "mean", ascending=False
    )
    unconstrained_stats = stats.loc[stats.index.isin(unconstrained_group)].sort_values(
        "mean", ascending=False
    )

    # --- Plotting -----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7, 4.5),  # Wide enough for two panels
        dpi=300,
        sharey=True,  # Share Y-axis for direct comparison
    )

    def plot_bars(ax, data, title):
        strategies = data.index
        means = data["mean"]
        sems = data["sem"]
        colors = [
            get_randomization_colors(label)
            for label in data.index
        ]

        ax.bar(
            x=np.arange(len(strategies)),
            height=means,
            yerr=sems,
            color=colors,
            capsize=5,
            width=0.7,
        )
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(np.arange(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Plot for each group
    plot_bars(ax1, constrained_stats, "Length-Constrained Networks")
    plot_bars(ax2, unconstrained_stats, "Length-Unconstrained Networks")

    # Shared Y-axis label and styling
    ax1.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.set_ylim(bottom=75)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    return fig


def grouped_accuracy_comparison_4groups(df: pd.DataFrame) -> plt.Figure:
    """
    Mostra la classificació en 4 blocs:
      1) Biological
      2) Mean length-constrained (neuron_binned, random_binned)
      3) Total length-constrained (random_pruned, connection_pruned)
      4) No constraints (unconstrained)
    Retorna un Figure “Nature style”.
    """

    # --- Definició dels grups ----------------------------------------------
    groups = OrderedDict([
        ("Biological",            [RANDOMIZATION_NAMES["biological"]]),
        ("Mean length-constr.",   [RANDOMIZATION_NAMES["neuron_binned"],
                                   RANDOMIZATION_NAMES["random_binned"]]),
        ("Total length-constr.",  [RANDOMIZATION_NAMES["random_pruned"],
                                   RANDOMIZATION_NAMES["connection_pruned"]]),
        ("No constraints",        [RANDOMIZATION_NAMES["unconstrained"]]),
    ])

    # --- Conversió de percentatges a float ---------------------------------
    replicate_cols = [c for c in df.columns
                      if c not in {"Randomization strategy", "Sweep name"}]
    df[replicate_cols] = (
        df[replicate_cols]
        .replace({",": "."}, regex=True)   # coma decimal → punt
        .replace(r"^\s*$", np.nan, regex=True)  # cadenes buides → NaN
        .astype(float) * 100
    )


    stats = df.set_index("Randomization strategy")[replicate_cols].agg(
                ["mean", "sem"], axis=1)

    # --- Preparar posicions de les barres ----------------------------------
    bar_positions, bar_means, bar_sems, bar_colors, bar_labels = [], [], [], [], []
    gap = 1.2      # separació extra entre blocs; ajusta’l al gust
    x = 0

    for group_name, strategies in groups.items():
        for strat in strategies:
            bar_positions.append(x)
            bar_means.append(stats.at[strat, "mean"])
            bar_sems.append(stats.at[strat, "sem"])
            bar_colors.append(get_randomization_colors(strat))
            bar_labels.append(strat)
            x += 1                      # pas normal dins del bloc

        x += gap                         # espai entre blocs

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)

    ax.bar(bar_positions,
           bar_means,
           yerr=bar_sems,
           color=bar_colors,
           capsize=5,
           width=0.7)

    # Etiquetes individuals (sota cada barra)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=9)

    # Etiquetes de cada bloc
    group_centers = []
    x = 0
    for g, strategies in groups.items():
        n = len(strategies)
        group_centers.append(x + (n - 1) / 2)
        x += n + gap

    for center, gname in zip(group_centers, groups.keys()):
        ax.text(center, 65,
                gname, ha="center", va="top",
                fontsize=12, fontstyle="italic")

    ax.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax.set_ylim(bottom=75)

    plt.tight_layout()
    return fig



def task_accuracy_comparison():
    # Task types (now on x-axis)
    tasks = [
        "Color\nDiscrimination",
        "Numerical\nDiscrimination", 
        "Shape\nRecognition",
    ]

    # Network configurations (now as colors/legend)
    networks = [
        RANDOMIZATION_NAMES["biological"],
        RANDOMIZATION_NAMES["neuron_binned"],
        RANDOMIZATION_NAMES["random_binned"],
        RANDOMIZATION_NAMES["random_pruned"],
        RANDOMIZATION_NAMES["connection_pruned"],
        RANDOMIZATION_NAMES["unconstrained"],
    ]

    # Made-up performance data for three tasks (percentage accuracy)
    # Reorganized: each row is a task, each column is a network
    task_data = {
        "Color\nDiscrimination": [100, 100, 100, 100, 100, 100],
        "Numerical\nDiscrimination": [84, 82, 82, 92, 91, 92],
        "Shape\nRecognition": [64, 60, 63, 69, 69, 70]
    }

    # Made-up error bars (standard error) - reorganized to match
    task_errors = {
        "Color\nDiscrimination": [0, 0, 0, 0, 0, 0],
        "Numerical\nDiscrimination": [0, 0, 0, 0, 0, 0],
        "Shape\nRecognition": [0, 0, 0, 0, 0, 0]
    }

    # Set width of bars
    bar_width = 0.14  # Slightly smaller to accommodate 4 networks
    capsize = 2
    index = np.arange(len(tasks))

    # Create the figure with Nature-compatible dimensions
    fig3, ax = plt.subplots(figsize=(9, 6), dpi=300)

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
            color=get_randomization_colors(network),
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