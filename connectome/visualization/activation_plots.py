"""Activation visualization plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter

from .plot_config import (
    apply_plot_style,
    get_randomization_colors,
    RANDOMIZATIONS,
    split_title,
)


def plot_activation_statistics(
    propagations_dict, neuron_position_data, num_steps=4, fig_width=120,
    axes=None, show_legend=True, show_title=True
):
    """
    Plot statistics about neuronal activations across different configurations.

    Parameters
    ----------
    propagations_dict : dict
        Dictionary of DataFrames with activation data for different configurations
    neuron_position_data : DataFrame
        DataFrame containing position data for neurons
    num_steps : int
        Number of message passing steps to plot
    fig_width : int
        Width in mm (183mm for double-column in Nature)
    axes : tuple of 3 Axes, optional
        If provided, plot on these axes instead of creating new figures
    show_legend : bool
        Whether to show legends on individual plots
    show_title : bool
        Whether to show titles on plots

    Returns
    -------
    tuple
        Three axes objects (or figures if axes not provided)
    """
    apply_plot_style({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 0.5,
        "lines.linewidth": 1,
        "lines.markersize": 3,
    })

    styles = {
        "Biological": dict(marker="o", ls="-", lw=1.1, zorder=4, alpha=0.8),
        "Unconstrained": dict(marker="s", ls="--", lw=2.2, alpha=0.8),
        "Random pruned": dict(marker="^", ls=":", lw=2.2, alpha=0.8),
        "Connection-pruned": dict(marker="v", ls="--", lw=2.2, alpha=0.8),
        "Binned": dict(marker="d", ls=":", lw=2.2, alpha=0.8),
        "Neuron binned": dict(marker="P", ls="--", lw=2.2, alpha=0.8),
    }

    configs = list(propagations_dict.keys())
    activation_percentages = {config: [] for config in configs}
    activation_distances = {config: [] for config in configs}
    rational_percentages = {config: [] for config in configs}

    rational_cell_types = ["KCapbp-m", "KCapbp-ap2", "KCapbp-ap1"]

    total_rational_neurons = neuron_position_data[
        neuron_position_data["cell_type"].isin(rational_cell_types)
    ]["root_id"].nunique()

    for config, prop_df in propagations_dict.items():
        total_neurons = len(neuron_position_data)

        for step in range(1, num_steps + 1):
            act_col = f"activation_{step}"
            if act_col in prop_df.columns:
                active_neurons = prop_df[prop_df[act_col] > 0]["root_id"].nunique()
                activation_percentages[config].append(100 * active_neurons / total_neurons)

                merged_rational = pd.merge(
                    prop_df[prop_df[act_col] > 0],
                    neuron_position_data[
                        neuron_position_data["cell_type"].isin(rational_cell_types)
                    ],
                    on="root_id",
                )
                active_rational_neurons = merged_rational["root_id"].nunique()
                rational_percentages[config].append(
                    100 * active_rational_neurons / total_rational_neurons
                    if total_rational_neurons > 0
                    else 0
                )
            else:
                activation_percentages[config].append(0)
                rational_percentages[config].append(0)

        merged = pd.merge(prop_df, neuron_position_data, on="root_id")

        input_active = merged[merged["input"] > 0]
        if not input_active.empty:
            eye_position = input_active[["pos_x", "pos_y", "pos_z"]].mean().values
        else:
            raise ValueError(
                f"No neurons are activated in the 'input' column for config {config}."
            )

        for step in range(1, num_steps + 1):
            act_col = f"activation_{step}"
            if act_col in merged.columns:
                active = merged[merged[act_col] > 0]
                if len(active) > 0:
                    positions = active[["pos_x", "pos_y", "pos_z"]].values
                    distances = np.sqrt(np.sum((positions - eye_position) ** 2, axis=1))
                    activation_distances[config].append(np.mean(distances) / 1000)
                else:
                    activation_distances[config].append(0)
            else:
                activation_distances[config].append(0)

    fig_width_in = fig_width * 0.0393701
    fig_height_in = fig_width_in / 1.4

    if axes is not None:
        ax1, ax2, ax3 = axes
    else:
        _, ax1 = plt.subplots(figsize=(fig_width_in, fig_height_in))
        _, ax2 = plt.subplots(figsize=(fig_width_in, fig_height_in))
        _, ax3 = plt.subplots(figsize=(fig_width_in, fig_height_in))

    for config in configs:
        ax1.plot(
            range(1, num_steps + 1),
            activation_percentages[config],
            label=RANDOMIZATIONS.label_for(config),
            color=get_randomization_colors(config),
            **styles[config],
        )

    ax1.set_xlabel("Message passing step")
    ax1.set_ylabel("Active neurons (%)")
    if show_title:
        ax1.set_title("Neural activation", pad=7)
    ax1.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax1.set_xticks(range(1, num_steps + 1))
    ymax1 = max([max(vals) for vals in activation_percentages.values()]) * 1.1
    ax1.set_ylim(0, ymax1)
    if show_legend:
        ax1.legend(loc="upper left", fontsize=9)

    for config in configs:
        ax2.plot(
            range(1, num_steps + 1),
            activation_distances[config],
            label=RANDOMIZATIONS.label_for(config),
            color=get_randomization_colors(config),
            **styles[config],
        )

    ax2.set_xlabel("Message passing step")
    ax2.set_ylabel("Mean distance from input (nm)")
    if show_title:
        ax2.set_title("Activation propagation distance", pad=7)
    ax2.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax2.set_xticks(range(1, num_steps + 1))
    ymax2 = max([max(vals) for vals in activation_distances.values()]) * 1.1
    ax2.set_ylim(0, ymax2)
    if show_legend:
        ax2.legend(loc="lower right", fontsize=9)

    for config in configs:
        ax3.plot(
            range(1, num_steps + 1),
            rational_percentages[config],
            label=RANDOMIZATIONS.label_for(config),
            color=get_randomization_colors(config),
            **styles[config],
        )

    ax3.set_xlabel("Message passing step")
    ax3.set_ylabel("Active Kenyon neurons (%)")
    if show_title:
        ax3.set_title("Kenyon cell types activation", pad=7)
    ax3.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax3.set_xticks(range(1, num_steps + 1))
    ymax3 = (
        max([max(vals) for vals in rational_percentages.values() if vals]) * 1.1
        if any([vals for vals in rational_percentages.values()])
        else 100
    )
    ax3.set_ylim(0, ymax3)
    if show_legend:
        ax3.legend(loc="upper left", fontsize=9)

    return ax1, ax2, ax3


def get_active_neuron_bounds(
    propagations_dict, neuron_position_data, padding_percent=10, num_steps=4
):
    """
    Calculate the bounds of active neurons across all configurations.

    Parameters
    ----------
    propagations_dict : dict
        Dictionary with configuration names and propagation dataframes
    neuron_position_data : pandas.DataFrame
        DataFrame with neuron positions
    padding_percent : float
        Percentage of padding to add around the active neurons
    num_steps : int
        Number of activation steps to consider

    Returns
    -------
    bounds : dict
        Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
    """
    all_active_x = []
    all_active_y = []
    all_active_z = []

    for config_name, prop_df in propagations_dict.items():
        merged_data = pd.merge(prop_df, neuron_position_data, on="root_id")

        active_mask = merged_data["activation_1"] > 0
        for step in range(2, num_steps + 1):
            active_mask |= merged_data[f"activation_{step}"] > 0

        active_neurons = merged_data[active_mask]

        if len(active_neurons) == 0:
            continue

        all_active_x.extend(active_neurons["pos_x"].values)
        all_active_y.extend(active_neurons["pos_y"].values)
        all_active_z.extend(active_neurons["pos_z"].values)

    if not all_active_x:
        return {
            "x_min": neuron_position_data["pos_x"].min(),
            "x_max": neuron_position_data["pos_x"].max(),
            "y_min": neuron_position_data["pos_y"].min(),
            "y_max": neuron_position_data["pos_y"].max(),
            "z_min": neuron_position_data["pos_z"].min(),
            "z_max": neuron_position_data["pos_z"].max(),
        }

    x_min, x_max = min(all_active_x), max(all_active_x)
    y_min, y_max = min(all_active_y), max(all_active_y)
    z_min, z_max = min(all_active_z), max(all_active_z)

    pad_x = (x_max - x_min) * padding_percent / 100
    pad_y = (y_max - y_min) * padding_percent / 100
    pad_z = (z_max - z_min) * padding_percent / 100

    return {
        "x_min": x_min - pad_x,
        "x_max": x_max + pad_x,
        "y_min": y_min - pad_y,
        "y_max": y_max + pad_y,
        "z_min": z_min - pad_z,
        "z_max": z_max + pad_z,
    }


def visualize_steps_separated_compact(
    propagations_dict,
    neuron_position_data,
    num_steps=4,
    max_neurons_percentage=5,
    voxel_size=None,
    smoothing=None,
    figsize=(20, 16),
    padding_percent=10,
    short_version=False,
    container=None,
    wspace=0.1,
    hspace=0.1,
    step_title_fontsize=20,
    row_label_fontsize=20,
    coordinate_label_fontsize=16,
    show_row_labels=True,
):
    """
    Create a compact grid of 3D visualizations with configurations as rows and steps as columns.

    Parameters
    ----------
    propagations_dict : dict
        Dictionary with configuration names (keys) and propagation dataframes (values)
    neuron_position_data : pandas.DataFrame
        DataFrame with columns 'root_id', 'pos_x', 'pos_y', 'pos_z'
    num_steps : int
        Number of activation steps to visualize
    max_neurons_percentage : int
        Maximum percentage of active neurons to plot per step
    voxel_size : int, optional
        Size of voxels for density calculation
    smoothing : float, optional
        Amount of Gaussian smoothing to apply
    figsize : tuple
        Figure size (width, height)
    padding_percent : float
        Percentage of padding to add around the active neurons
    short_version : bool
        Whether to only plot 2 randomizations instead of 4
    container : matplotlib.figure.Figure or matplotlib.figure.SubFigure, optional
        Figure-like container to draw into. If None, create a new figure.
    wspace : float
        Horizontal spacing between subplots
    hspace : float
        Vertical spacing between subplots
    step_title_fontsize : int
        Font size for step titles
    row_label_fontsize : int
        Font size for row labels
    coordinate_label_fontsize : int
        Font size for the X/Y/Z labels on the last panel
    show_row_labels : bool
        Whether to display randomization labels on the right side
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with the grid of visualizations
    """
    ordered_propagations = {}
    for key in RANDOMIZATIONS.order:
        label = RANDOMIZATIONS.label_for(key)
        if label in propagations_dict:
            ordered_propagations[label] = propagations_dict[label]

    for key, value in propagations_dict.items():
        if key not in ordered_propagations:
            ordered_propagations[key] = value

    propagations_dict = ordered_propagations

    if short_version:
        short_keys = ["biological", "neuron_binned", "connection_pruned", "unconstrained"]
        short_labels = {RANDOMIZATIONS.label_for(key) for key in short_keys}
        propagations_dict = {k: v for k, v in propagations_dict.items() if k in short_labels}

    bounds = get_active_neuron_bounds(
        propagations_dict, neuron_position_data, padding_percent, num_steps
    )
    x_min, x_max = bounds["x_min"], bounds["x_max"]
    y_min, y_max = bounds["y_min"], bounds["y_max"]
    z_min, z_max = bounds["z_min"], bounds["z_max"]

    apply_plot_style()

    subplot_kwargs = {"projection": "3d"}
    gridspec_kwargs = {"wspace": wspace, "hspace": hspace}

    if container is None:
        fig, axes = plt.subplots(
            len(propagations_dict),
            num_steps + 1,
            figsize=figsize,
            subplot_kw=subplot_kwargs,
            gridspec_kw=gridspec_kwargs,
        )
    else:
        fig = container
        axes = fig.subplots(
            len(propagations_dict),
            num_steps + 1,
            subplot_kw=subplot_kwargs,
            gridspec_kw=gridspec_kwargs,
        )

    for i, (config_name, prop_df) in enumerate(propagations_dict.items()):
        display_label = RANDOMIZATIONS.label_for(config_name)
        config_color = get_randomization_colors(display_label)
        if display_label == "Unconstrained":
            config_name_display = "Uncon\nstrained"
        else:
            config_name_display = split_title(display_label, 10)

        merged_data = pd.merge(prop_df, neuron_position_data, on="root_id")

        # First column: Input visualization (only in middle row)
        middle_row = len(propagations_dict) // 2
        if len(propagations_dict) > 1:
            ax = axes[i, 0]
        else:
            ax = axes[0]

        if i == middle_row:
            _style_3d_axis(ax)
            ax.set_title("Input", pad=5, fontsize=step_title_fontsize)
            input_data = merged_data[merged_data["input"] > 0].copy()

            if len(input_data) > 0:
                _plot_activation_step(
                    ax,
                    input_data,
                    "input",
                    "black",
                    voxel_size,
                    smoothing,
                    max_neurons_percentage,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    z_min,
                    z_max,
                )

            ax.view_init(elev=30, azim=45)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
        else:
            ax.set_axis_off()

        # Process each activation step
        for step in range(1, num_steps + 1):
            if len(propagations_dict) > 1:
                ax = axes[i, step]
            else:
                ax = axes[step]

            _style_3d_axis(ax)

            if i == len(propagations_dict) - 1 and step == num_steps:
                ax.set_xlabel("X", labelpad=-10, fontsize=coordinate_label_fontsize)
                ax.set_ylabel("Y", labelpad=-10, fontsize=coordinate_label_fontsize)
                ax.set_zlabel("Z", labelpad=-10, fontsize=coordinate_label_fontsize)

            # Row labels on the right side (last step column)
            if show_row_labels and step == num_steps:
                ax.text2D(
                    1.15,
                    0.5,
                    config_name_display,
                    transform=ax.transAxes,
                    va="center",
                    ha="center",
                    rotation=-90,
                    fontsize=row_label_fontsize,
                )

            act_col = f"activation_{step}"
            step_data = merged_data[merged_data[act_col] > 0].copy()

            if i == 0:
                ax.set_title(f"Step {step}", pad=5, fontsize=step_title_fontsize)

            if len(step_data) > 0:
                _plot_activation_step(
                    ax,
                    step_data,
                    act_col,
                    config_color,
                    voxel_size,
                    smoothing,
                    max_neurons_percentage,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    z_min,
                    z_max,
                )

            ax.view_init(elev=30, azim=45)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

    return fig


def _style_3d_axis(ax):
    """Apply consistent styling to a 3D axis."""
    ax.set_facecolor("white")
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def _plot_activation_step(
    ax,
    data,
    col,
    color,
    voxel_size,
    smoothing,
    max_neurons_percentage,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
):
    """Plot a single activation step on a 3D axis."""
    if voxel_size is not None:
        x_bins = np.linspace(x_min, x_max, voxel_size)
        y_bins = np.linspace(y_min, y_max, voxel_size)
        z_bins = np.linspace(z_min, z_max, voxel_size)

        H, _ = np.histogramdd(
            data[["pos_x", "pos_y", "pos_z"]].values,
            bins=[x_bins, y_bins, z_bins],
            weights=data[col].values,
        )

        H_smooth = gaussian_filter(H, sigma=smoothing)

        if H_smooth.max() > 0:
            x_centers = (x_bins[:-1] + x_bins[1:]) / 2
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2
            z_centers = (z_bins[:-1] + z_bins[1:]) / 2

            X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

            threshold = 0.25 * H_smooth.max()
            mask = H_smooth > threshold

            if np.any(mask):
                x_coords = X[mask]
                y_coords = Y[mask]
                z_coords = Z[mask]
                values = H_smooth[mask]

                norm_values = values / values.max()
                sizes = 40 * norm_values + 5
                alphas = 0.3 + 0.4 * norm_values

                rgba_colors = np.array([to_rgba(color, alpha=a) for a in alphas])

                ax.scatter(
                    x_coords,
                    y_coords,
                    z_coords,
                    c=rgba_colors,
                    s=sizes,
                    edgecolors="none",
                    depthshade=True,
                )

    # Sample individual neurons
    total_neurons = len(data)
    sample_size = int(total_neurons * (max_neurons_percentage / 100))
    if sample_size < total_neurons:
        neuron_sample = data.sample(sample_size, random_state=1234)
    else:
        neuron_sample = data

    max_activation = neuron_sample[col].max()
    if max_activation > 0:
        normalized_activation = neuron_sample[col] / max_activation
    else:
        normalized_activation = neuron_sample[col]

    neuron_sizes = 15 * normalized_activation + 3

    ax.scatter(
        neuron_sample["pos_x"],
        neuron_sample["pos_y"],
        neuron_sample["pos_z"],
        c=[color],
        s=neuron_sizes,
        alpha=0.7,
        edgecolors="none",
        depthshade=True,
    )


def plot_3d_activation_compact(
    ax,
    positions,
    alphas,
    color,
    title,
    label,
    marker_size=20,
    voxel_size=None,
    smoothing=None,
    bounds=None,
):
    """Plot a compact 3D activation visualization."""
    alphas = np.asarray(alphas, dtype=float)
    if alphas.size:
        if (alphas > 1).any() or (alphas < 0).any():
            max_val = alphas.max()
            if max_val > 0:
                alphas = alphas / max_val
        alphas = np.clip(alphas, 0.0, 1.0)

    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(False)

    if bounds:
        x_min, x_max = bounds["x_min"], bounds["x_max"]
        y_min, y_max = bounds["y_min"], bounds["y_max"]
        z_min, z_max = bounds["z_min"], bounds["z_max"]
    else:
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    if positions.shape[0] == 0:
        return

    if voxel_size:
        bins = [
            np.arange(x_min, x_max + voxel_size, voxel_size),
            np.arange(y_min, y_max + voxel_size, voxel_size),
            np.arange(z_min, z_max + voxel_size, voxel_size),
        ]
        hist, _ = np.histogramdd(positions, bins=bins, weights=alphas)

        if hist.max() > 0:
            hist /= hist.max()
        if smoothing:
            hist = gaussian_filter(hist, sigma=smoothing)

        x_centers = (bins[0][:-1] + bins[0][1:]) / 2
        y_centers = (bins[1][:-1] + bins[1][1:]) / 2
        z_centers = (bins[2][:-1] + bins[2][1:]) / 2
        x, y, z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

        rgba_colors = np.zeros(hist.shape + (4,))
        base_color = to_rgba(color)
        rgba_colors[..., :3] = base_color[:3]
        rgba_colors[..., 3] = hist

        ax.scatter(
            x.flatten(),
            y.flatten(),
            z.flatten(),
            c=rgba_colors.reshape(-1, 4),
            marker="s",
            s=voxel_size**2 * 0.8,
            edgecolors="none",
        )
    else:
        rgba_colors = np.array([to_rgba(color, alpha=a) for a in alphas])
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=rgba_colors,
            s=marker_size,
            edgecolors="none",
            depthshade=True,
        )

    ax.scatter([], [], [], c=[color], s=100, edgecolors="none", label=label)
    ax.set_title(title, fontsize=16, pad=-20)
    ax.set_aspect("equal", "box")

