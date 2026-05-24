"""Model accuracy comparison plots."""

import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch

from .plot_config import (
    RANDOMIZATIONS,
    apply_plot_style,
    get_randomization_colors,
    darken_color,
    RANDOMIZATION_NAMES,
)
from .plots import _prepare_results_df
from paths import PROJECT_ROOT


def grouped_accuracy_comparison(df: pd.DataFrame) -> plt.Figure:
    """
    Plots classification accuracy in two groups: length-constrained and unconstrained.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with accuracy results, matching the format of
        'notebooks/visualization/data/randomizations_seeds.csv'.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object containing the panel of plots.
    """
    constrained_keys = ["biological", "random_binned", "neuron_binned"]
    unconstrained_keys = ["unconstrained", "random_pruned", "connection_pruned"]
    constrained_group = [RANDOMIZATIONS.label_for(key) for key in RANDOMIZATIONS.sort_keys(constrained_keys)]
    unconstrained_group = [RANDOMIZATIONS.label_for(key) for key in RANDOMIZATIONS.sort_keys(unconstrained_keys)]

    all_strategies = set(df["Randomization strategy"])
    categorized = set(constrained_group) | set(unconstrained_group)
    if all_strategies != categorized:
        uncategorized = all_strategies - categorized
        print(f"Warning: Uncategorized strategies found and will be ignored: {uncategorized}")
        df = df[df["Randomization strategy"].isin(categorized)]

    df = df.dropna(subset=["Randomization strategy"])

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

    stats = df.set_index("Randomization strategy")[replicate_cols].agg(
        ["mean", "sem"], axis=1
    )

    constrained_stats = stats.loc[stats.index.isin(constrained_group)].sort_values(
        "mean", ascending=False
    )
    unconstrained_stats = stats.loc[stats.index.isin(unconstrained_group)].sort_values(
        "mean", ascending=False
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.5), dpi=300, sharey=True)

    def plot_bars(ax, data, title):
        strategies = data.index
        means = data["mean"]
        sems = data["sem"]
        colors = [get_randomization_colors(label) for label in data.index]

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

    plot_bars(ax1, constrained_stats, "Length-Constrained Networks")
    plot_bars(ax2, unconstrained_stats, "Length-Unconstrained Networks")

    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.set_ylim(bottom=75)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def grouped_accuracy_comparison_4groups(df: pd.DataFrame) -> plt.Figure:
    """
    Show classification in 4 blocks: Biological, Mean length-constrained,
    Total length-constrained, and No constraints.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with accuracy results.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object.
    """
    groups = OrderedDict([
        ("Biological", [RANDOMIZATIONS.label_for("biological")]),
        (
            "Mean length-constr.",
            [RANDOMIZATIONS.label_for(key) for key in RANDOMIZATIONS.sort_keys(["random_binned", "neuron_binned"])],
        ),
        ("Total length-constr.", [RANDOMIZATIONS.label_for("connection_pruned")]),
        ("No constraints", [RANDOMIZATIONS.label_for("unconstrained")]),
    ])

    replicate_cols = [
        c for c in df.columns if c not in {"Randomization strategy", "Sweep name"}
    ]
    df[replicate_cols] = (
        df[replicate_cols]
        .replace({",": "."}, regex=True)
        .replace(r"^\s*$", np.nan, regex=True)
        .astype(float)
        * 100
    )

    stats = df.set_index("Randomization strategy")[replicate_cols].agg(
        ["mean", "sem"], axis=1
    )

    bar_positions, bar_means, bar_sems, bar_colors, bar_labels = [], [], [], [], []
    gap = 1.2
    x = 0

    for _, strategies in groups.items():
        for strat in strategies:
            bar_positions.append(x)
            bar_means.append(stats.at[strat, "mean"])
            bar_sems.append(stats.at[strat, "sem"])
            bar_colors.append(get_randomization_colors(strat))
            bar_labels.append(strat)
            x += 1
        x += gap

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)

    ax.bar(
        bar_positions,
        bar_means,
        yerr=bar_sems,
        color=bar_colors,
        capsize=5,
        width=0.7,
    )

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=9)

    group_centers = []
    x = 0
    for g, strategies in groups.items():
        n = len(strategies)
        group_centers.append(x + (n - 1) / 2)
        x += n + gap

    for center, gname in zip(group_centers, groups.keys()):
        ax.text(
            center, 65, gname, ha="center", va="top", fontsize=12, fontstyle="italic"
        )

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(bottom=75)

    plt.tight_layout()
    return fig


NUMERICAL_DISCRIMINATION_FILES = OrderedDict([
    ("biological_results.csv", "biological"),
    ("unconstrained_results.csv", "unconstrained"),
    ("connection_pruned_results.csv", "connection_pruned"),
    ("binned_results.csv", "random_binned"),
    ("neuron_binned_results.csv", "neuron_binned"),
])

TASK_IMAGE_FILES = {
    "Color\ndiscrimination": ("t5.png", "t6.png"),
    "Shape\nrecognition": ("t3.png", "t4.png"),
    "Numerical\ndiscrimination": ("t1.png", "t2.png"),
}


def _add_task_image_insets(
    ax: plt.Axes,
    tasks: list[str],
    task_positions: np.ndarray,
    *,
    x0: float = 4,
    width: float = 33,
    height: float = 0.74,
) -> None:
    """Draw task example images inside the low-accuracy region of the plot."""
    task_images_dir = os.path.join(PROJECT_ROOT, "paper_figures", "task_images")

    for y_center, task in zip(task_positions, tasks):
        image_names = TASK_IMAGE_FILES[task]
        container = ax.inset_axes(
            [x0, y_center - height / 2, width, height],
            transform=ax.transData,
            zorder=6,
        )
        container.set_axis_off()
        frame = FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            transform=container.transAxes,
            facecolor="white",
            edgecolor="white",
            linewidth=1.2,
            alpha=0.35,
            zorder=0,
        )
        container.add_patch(frame)

        for idx, image_name in enumerate(image_names):
            image_ax = container.inset_axes([0.06 + idx * 0.42, 0.05, 0.46, 0.9], zorder=1)
            image_ax.imshow(imread(os.path.join(task_images_dir, image_name)))
            image_ax.set_axis_off()
            image_ax.set_aspect("equal")


def _load_numerical_discrimination_summary(min_weber: float = 1.33) -> tuple[dict[str, float], dict[str, float]]:
    """Compute equalized numerical-discrimination accuracy from the supplementary CSVs."""
    folder = os.path.join(PROJECT_ROOT, "supplementary_data")
    means: dict[str, float] = {}
    errors: dict[str, float] = {}

    for filename, key in NUMERICAL_DISCRIMINATION_FILES.items():
        csv_path = os.path.join(folder, filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing numerical-discrimination results file: {csv_path}")

        df = pd.read_csv(csv_path)
        df = _prepare_results_df(df)
        df = df[(df["equalized"]) & (df["weber_ratio"] >= float(min_weber))].copy()
        if df.empty:
            raise ValueError(f"No equalized numerical-discrimination trials found in {csv_path}")

        label = RANDOMIZATIONS.label_for(key)
        accuracy = df["Is correct"].astype(float)
        means[label] = float(accuracy.mean() * 100)
        errors[label] = float(accuracy.sem() * 100)

    return means, errors


def _task_accuracy_specs() -> tuple[list[str], list[str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Return the task/network order and the accuracy values used in panel g."""
    tasks = [
        "Color\ndiscrimination",
        "Shape\nrecognition",
        "Numerical\ndiscrimination",
    ]

    networks = [RANDOMIZATIONS.label_for(key) for key in RANDOMIZATIONS.order]

    numerical_means, numerical_errors = _load_numerical_discrimination_summary()

    task_data = {
        "Color\ndiscrimination": {network: 100 for network in networks},
        "Shape\nrecognition": {
            RANDOMIZATIONS.label_for("biological"): 64,
            RANDOMIZATIONS.label_for("unconstrained"): 70,
            RANDOMIZATIONS.label_for("connection_pruned"): 69,
            RANDOMIZATIONS.label_for("random_binned"): 63,
            RANDOMIZATIONS.label_for("neuron_binned"): 60,
        },
        "Numerical\ndiscrimination": numerical_means,
    }

    task_errors = {
        "Color\ndiscrimination": {network: 0 for network in networks},
        "Shape\nrecognition": {
            RANDOMIZATIONS.label_for("biological"): 1,
            RANDOMIZATIONS.label_for("unconstrained"): 3,
            RANDOMIZATIONS.label_for("connection_pruned"): 2,
            RANDOMIZATIONS.label_for("random_binned"): 1,
            RANDOMIZATIONS.label_for("neuron_binned"): 2,
        },
        "Numerical\ndiscrimination": numerical_errors,
    }

    return tasks, networks, task_data, task_errors


def task_accuracy_table(include_errors: bool = True) -> pd.DataFrame:
    """Return the values shown in panel g as a notebook-friendly table."""
    tasks, networks, task_data, task_errors = _task_accuracy_specs()

    if include_errors:
        table = pd.DataFrame(
            {
                task.replace("\n", " "): [
                    f"{task_data[task][network]:.2f} +/- {task_errors[task][network]:.2f}"
                    for network in networks
                ]
                for task in tasks
            },
            index=networks,
        )
    else:
        table = pd.DataFrame(
            {
                task.replace("\n", " "): [task_data[task][network] for network in networks]
                for task in tasks
            },
            index=networks,
        )

    table.index.name = "Network"
    return table


def _add_single_task_image_inset(
    ax: plt.Axes,
    task: str,
    *,
    x0: float = 4,
    width: float = 43,
    y0: float = -0.203,
    height: float = 0.385,
) -> None:
    """Draw the paired task images inside one task row."""
    task_images_dir = os.path.join(PROJECT_ROOT, "paper_figures", "task_images")
    container = ax.inset_axes([x0, y0, width, height], transform=ax.transData, zorder=6)
    container.set_axis_off()
    frame = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        transform=container.transAxes,
        facecolor="white",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.5,
        zorder=0,
    )
    container.add_patch(frame)

    pad_x = 0.05
    pad_y = 0.02
    img_w = (1 - 3 * pad_x) / 2
    img_h = 1 - 2 * pad_y
    for idx, image_name in enumerate(TASK_IMAGE_FILES[task]):
        image_ax = container.inset_axes(
            [pad_x + idx * (img_w + pad_x), pad_y, img_w, img_h], zorder=1
        )
        image_ax.imshow(imread(os.path.join(task_images_dir, image_name)))
        image_ax.set_axis_off()
        image_ax.set_aspect("equal")


def plot_task_accuracy_row(
    task: str,
    *,
    ax: plt.Axes | None = None,
    show_xlabel: bool = False,
    show_chance_label: bool = False,
    chance_fontsize: int = 10,
    chance_y: float = -0.38,
) -> plt.Figure:
    """Plot one task row with horizontal grouped bars and task images."""
    apply_plot_style()
    tasks, networks, task_data, task_errors = _task_accuracy_specs()
    if task not in tasks:
        raise ValueError(f"Unknown task '{task}'. Expected one of {tasks}.")

    bar_height = 0.14
    capsize = 4
    y_positions = ((len(networks) - 1) / 2 - np.arange(len(networks))) * bar_height

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 2.6), dpi=300)
    else:
        fig = ax.get_figure()

    values = [task_data[task][network] for network in networks]
    errors = [task_errors[task][network] for network in networks]

    for y, network, value, error in zip(y_positions, networks, values, errors):
        ax.barh(
            y,
            value,
            bar_height,
            xerr=error,
            label=network,
            color=get_randomization_colors(network),
            alpha=0.9,
            capsize=capsize,
            ecolor=darken_color(get_randomization_colors(network)),
            error_kw={"elinewidth": 2, "capthick": 2},
        )

    ax.axvline(x=50, linestyle="--", color="#666666", alpha=0.5, linewidth=1)
    _add_single_task_image_inset(ax, task)

    if show_chance_label:
        ax.text(
            52,
            chance_y,
            "Chance level",
            fontsize=chance_fontsize,
            color="#666666",
            va="bottom",
            ha="left",
            bbox=dict(
                facecolor="white",
                edgecolor="#666666",
                boxstyle="round,pad=0.5",
                alpha=0.8,
            ),
        )

    ax.set_xlim(0, 105)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([0])
    ax.set_yticklabels([task], fontsize=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=14)

    if show_xlabel:
        ax.set_xlabel("Accuracy (%)", fontsize=16)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    return fig


def task_accuracy_comparison(ax=None, show_legend=True, chance_fontsize=10) -> plt.Figure:
    """
    Create a comparison plot of accuracy across different tasks and networks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    show_legend : bool
        Whether to show the legend.
    chance_fontsize : int
        Font size for the "Chance level" annotation.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object with the comparison.
    """

    apply_plot_style()
    
    tasks, networks, task_data, task_errors = _task_accuracy_specs()

    bar_height = 0.14
    capsize = 4
    task_positions = np.arange(len(tasks))[::-1]
    offsets = (np.arange(len(networks)) - (len(networks) - 1) / 2) * bar_height

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    else:
        fig = ax.get_figure()

    for i, network in enumerate(networks):
        network_data = [task_data[task][network] for task in tasks]
        network_errors = [task_errors[task][network] for task in tasks]
        ax.barh(
            task_positions + offsets[i],
            network_data,
            bar_height,
            xerr=network_errors,
            label=network,
            color=get_randomization_colors(network),
            alpha=0.9,
            capsize=capsize,
            ecolor=darken_color(get_randomization_colors(network)),
            error_kw={"elinewidth": 2, "capthick": 2},
        )

    ax.axvline(x=50, linestyle="--", color="#666666", alpha=0.5, linewidth=1)
    _add_task_image_insets(ax, tasks, task_positions)

    ax.text(
        52,
        task_positions[-1] - 0.42,
        "Chance level",
        fontsize=chance_fontsize,
        color="#666666",
        va="bottom",
        ha="left",
        bbox=dict(
            facecolor="white",
            edgecolor="#666666",
            boxstyle="round,pad=0.5",
            alpha=0.8,
        ),
    )

    ax.set_xlabel("Accuracy (%)", fontsize=16)
    ax.set_xlim(0, 105)
    ax.set_ylim(task_positions[-1] - 0.6, task_positions[0] + 0.6)
    ax.set_yticks(task_positions)
    ax.set_yticklabels(tasks, fontsize=16, rotation=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=14)
    if show_legend:
        ax.legend(fontsize=16, loc="lower right", frameon=False)

    return fig

