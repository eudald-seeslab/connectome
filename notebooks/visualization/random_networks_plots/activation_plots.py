import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import to_rgba
import matplotlib as mpl

from notebooks.visualization.activations_funcs import split_title

from .plot_config import apply_plot_style, get_randomization_colors, RANDOMIZATION_NAMES


def plot_activation_statistics(
    propagations_dict, neuron_position_data, num_steps=4, fig_width=120
):
    """
    Plot statistics about neuronal activations across different configurations in Nature journal style.

    Parameters:
    -----------
    propagations_dict : dict
        Dictionary of DataFrames with activation data for different configurations
    neuron_position_data : DataFrame
        DataFrame containing position data for neurons
    fig_width : int
        Width in mm (183mm for double-column in Nature)

    Returns:
    --------
    tuple
        Three matplotlib.figure.Figure objects: one for activation percentages, one for activation distances,
        and one for rational cell types activation percentages.
    """
    # Apply centralized plotting style (small-text override)
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
        "Biological":        dict(marker="o",  ls="-", lw=1.1, zorder=4, alpha=.8),
        "Unconstrained":     dict(marker="s", ls="--", lw=2.2, alpha=.8),
        "Random pruned":     dict(marker="^", ls=":",  lw=2.2, alpha=.8),
        "Connection-pruned": dict(marker="v", ls="--", lw=2.2, alpha=.8),
        "Random bin-wise":   dict(marker="d", ls=":",  lw=2.2, alpha=.8),
        "Neuron binned":     dict(marker="P", ls="--", lw=2.2, alpha=.8),
    }   

    # Calculate metrics for each configuration
    configs = list(propagations_dict.keys())
    activation_percentages = {config: [] for config in configs}
    activation_distances = {config: [] for config in configs}
    rational_percentages = {config: [] for config in configs}  # New metric

    # Define rational cell types
    rational_cell_types = ["KCapbp-m", "KCapbp-ap2", "KCapbp-ap1"]
    
    # Count total rational neurons
    total_rational_neurons = neuron_position_data[
        neuron_position_data['cell_type'].isin(rational_cell_types)
    ]['root_id'].nunique()

    for config, prop_df in propagations_dict.items():
        # Calculate percentage of neurons active at each step
        total_neurons = len(neuron_position_data)

        for step in range(1, num_steps + 1):
            act_col = f"activation_{step}"
            if act_col in prop_df.columns:
                active_neurons = prop_df[prop_df[act_col] > 0]["root_id"].nunique()
                activation_percentages[config].append(
                    100 * active_neurons / total_neurons
                )
                
                # Calculate percentage of rational cell types active
                merged_rational = pd.merge(
                    prop_df[prop_df[act_col] > 0],
                    neuron_position_data[neuron_position_data['cell_type'].isin(rational_cell_types)],
                    on="root_id"
                )
                active_rational_neurons = merged_rational["root_id"].nunique()
                rational_percentages[config].append(
                    100 * active_rational_neurons / total_rational_neurons if total_rational_neurons > 0 else 0
                )
            else:
                activation_percentages[config].append(0)
                rational_percentages[config].append(0)

        # Merge prop_df with neuron_position_data to get positions
        merged = pd.merge(prop_df, neuron_position_data, on="root_id")

        # Calculate average distance of active neurons from eye
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
                    activation_distances[config].append(np.mean(distances))
                else:
                    activation_distances[config].append(0)
            else:
                activation_distances[config].append(0)

    # Convert mm to inches (1 mm = 0.0393701 inches)
    fig_width_in = fig_width * 0.0393701
    fig_height_in = fig_width_in / 1.4

    # Create figure for activation percentages
    fig1, ax1 = plt.subplots(figsize=(fig_width_in, fig_height_in))
    for i, config in enumerate(configs):
        ax1.plot(
            range(1, num_steps + 1),
            activation_percentages[config],
            label=RANDOMIZATION_NAMES.get(config, config),
            color=get_randomization_colors(config),
            **styles[config]
        )

    ax1.set_xlabel("Message Passing Step")
    ax1.set_ylabel("Neurons Active (%)")
    ax1.set_title("Neural Activation", pad=7)
    ax1.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax1.set_xticks(range(1, num_steps + 1))
    ymax1 = max([max(vals) for vals in activation_percentages.values()]) * 1.1
    ax1.set_ylim(0, ymax1)
    ax1.legend(loc="upper left", fontsize=9)

    # Create figure for activation distances
    fig2, ax2 = plt.subplots(figsize=(fig_width_in, fig_height_in))
    for i, config in enumerate(configs):
        ax2.plot(
            range(1, num_steps + 1),
            activation_distances[config],
            label=RANDOMIZATION_NAMES.get(config, config),
            color=get_randomization_colors(config),
            **styles[config]
        )

    ax2.set_xlabel("Message Passing Step")
    ax2.set_ylabel("Avg. Distance from Input (μm)")
    ax2.set_title("Activation Propagation Distance", pad=7)
    ax2.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax2.set_xticks(range(1, num_steps + 1))
    ymax2 = max([max(vals) for vals in activation_distances.values()]) * 1.1
    ax2.set_ylim(0, ymax2)
    ax2.legend(loc="lower right", fontsize=9)

    # Create figure for rational cell types activation
    fig3, ax3 = plt.subplots(figsize=(fig_width_in, fig_height_in))
    for i, config in enumerate(configs):
        ax3.plot(
            range(1, num_steps + 1),
            rational_percentages[config],
            label=RANDOMIZATION_NAMES.get(config, config),
            color=get_randomization_colors(config),
            **styles[config]
        )

    ax3.set_xlabel("Message Passing Step")
    ax3.set_ylabel("Rational Neurons Active (%)")
    ax3.set_title("Rational Cell Types Activation", pad=7)
    ax3.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax3.set_xticks(range(1, num_steps + 1))
    ymax3 = max([max(vals) for vals in rational_percentages.values() if vals]) * 1.1 if any([vals for vals in rational_percentages.values()]) else 100
    ax3.set_ylim(0, ymax3)
    ax3.legend(loc="upper left", fontsize=9)

    return fig1, fig2, fig3


def get_active_neuron_bounds(
    propagations_dict, neuron_position_data, padding_percent=10, num_steps=4
):
    """
    Calculate the bounds of active neurons across all configurations.

    Parameters:
    -----------
    propagations_dict : dict
        Dictionary with configuration names and propagation dataframes
    neuron_position_data : pandas.DataFrame
        DataFrame with neuron positions
    padding_percent : float
        Percentage of padding to add around the active neurons

    Returns:
    --------
    bounds : dict
        Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
    """
    # Create lists to store all active neuron positions
    all_active_x = []
    all_active_y = []
    all_active_z = []

    # Process each configuration
    for config_name, prop_df in propagations_dict.items():
        # Merge with position data
        merged_data = pd.merge(prop_df, neuron_position_data, on="root_id")

        # Collect positions of all active neurons (in any step)
        active_mask = merged_data["activation_1"] > 0
        for step in range(2, num_steps + 1):
            active_mask |= merged_data[f"activation_{step}"] > 0

        active_neurons = merged_data[active_mask]

        # Skip if no active neurons
        if len(active_neurons) == 0:
            continue

        # Add to lists
        all_active_x.extend(active_neurons["pos_x"].values)
        all_active_y.extend(active_neurons["pos_y"].values)
        all_active_z.extend(active_neurons["pos_z"].values)

    # If no active neurons found in any configuration, use full bounds
    if not all_active_x:
        return {
            "x_min": neuron_position_data["pos_x"].min(),
            "x_max": neuron_position_data["pos_x"].max(),
            "y_min": neuron_position_data["pos_y"].min(),
            "y_max": neuron_position_data["pos_y"].max(),
            "z_min": neuron_position_data["pos_z"].min(),
            "z_max": neuron_position_data["pos_z"].max(),
        }

    # Calculate bounds of active neurons
    x_min, x_max = min(all_active_x), max(all_active_x)
    y_min, y_max = min(all_active_y), max(all_active_y)
    z_min, z_max = min(all_active_z), max(all_active_z)

    # Add padding
    pad_x = (x_max - x_min) * padding_percent / 100
    pad_y = (y_max - y_min) * padding_percent / 100
    pad_z = (z_max - z_min) * padding_percent / 100

    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y
    z_min -= pad_z
    z_max += pad_z

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
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
):
    """
    Create a compact 4x(num_steps+1) grid of 3D visualizations with configurations as rows 
    and steps as columns, including a first column for input visualization.

    Parameters:
    -----------
    propagations_dict : dict
        Dictionary with configuration names (keys) and propagation dataframes (values)
    neuron_position_data : pandas.DataFrame
        DataFrame with columns 'root_id', 'pos_x', 'pos_y', 'pos_z'
    max_neurons_percentage : int
        Maximum percentage of active neurons to plot per step (for performance)
    voxel_size : int
        Size of voxels for density calculation
    smoothing : float
        Amount of Gaussian smoothing to apply
    figsize : tuple
        Figure size (width, height)
    padding_percent : float
        Percentage of padding to add around the active neurons

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with the 4x(num_steps+1) grid of visualizations
    """

    # Stop if voxel_size is None and smoothing is not None or vice versa
    if voxel_size is None and smoothing is not None:
        raise ValueError("voxel_size must be provided if smoothing is provided")
    if voxel_size is not None and smoothing is None:
        raise ValueError("smoothing must be provided if voxel_size is provided")

    # Get bounds of active neurons
    bounds = get_active_neuron_bounds(
        propagations_dict, neuron_position_data, padding_percent, num_steps
    )
    x_min, x_max = bounds["x_min"], bounds["x_max"]
    y_min, y_max = bounds["y_min"], bounds["y_max"]
    z_min, z_max = bounds["z_min"], bounds["z_max"]

    # Nature-friendly colors for each step
    step_colors = ["#4878D0", "#6ACC64", "#EE854A", "#D65F5F"]
    input_color = "#8A2BE2"  # A distinct purple color for input visualization

    # Create a figure with 4x(num_steps+1) grid: rows=configurations, columns=input + steps
    fig, axes = plt.subplots(
        len(propagations_dict), num_steps + 1, figsize=figsize, subplot_kw={"projection": "3d"}
    )

    # Process each configuration (rows)
    for i, (config_name, prop_df) in enumerate(propagations_dict.items()):

        config_name = split_title(config_name, 10)

        # Merge with position data
        merged_data = pd.merge(prop_df, neuron_position_data, on="root_id")
        
        # First column: Input visualization
        if len(propagations_dict) > 1:
            ax = axes[i, 0]
        else:
            ax = axes[0]
            
        # Style the subplot for input
        ax.set_facecolor("white")
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("lightgray")
        ax.yaxis.pane.set_edgecolor("lightgray")
        ax.zaxis.pane.set_edgecolor("lightgray")

        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # Add configuration label to the leftmost column only
        ax.text2D(
            -0.2,
            0.5,
            config_name,
            transform=ax.transAxes,
            va="center",
            ha="center",
            rotation=90,
            fontsize=16,
        )
        
        # Add title to the top row only
        if i == 0:
            ax.set_title("Input", pad=5)
            
        # Filter data for input
        input_data = merged_data[merged_data["input"] > 0].copy()
        
        # Skip if no data
        if len(input_data) == 0:
            continue
            
        if voxel_size is not None:
            # Create 3D histogram for density visualization
            x_bins = np.linspace(x_min, x_max, voxel_size)
            y_bins = np.linspace(y_min, y_max, voxel_size)
            z_bins = np.linspace(z_min, z_max, voxel_size)

            H, edges = np.histogramdd(
                input_data[["pos_x", "pos_y", "pos_z"]].values,
                bins=[x_bins, y_bins, z_bins],
                weights=input_data["input"].values,
            )

            # Apply Gaussian smoothing
            H_smooth = gaussian_filter(H, sigma=smoothing)

            # Skip if all zeros
            if H_smooth.max() == 0:
                continue

            # Get voxel coordinates
            x_centers = (x_bins[:-1] + x_bins[1:]) / 2
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2
            z_centers = (z_bins[:-1] + z_bins[1:]) / 2

            # Create meshgrid of voxel centers
            X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

            # Get non-zero voxels (above threshold of max value)
            threshold = 0.25 * H_smooth.max()
            mask = H_smooth > threshold

            # Skip if nothing to plot
            if not np.any(mask):
                continue

            # Get coordinates and values of voxels above threshold
            x_coords = X[mask]
            y_coords = Y[mask]
            z_coords = Z[mask]
            values = H_smooth[mask]

            # Normalize values for sizing
            norm_values = values / values.max()

            # Calculate sizes based on activation strength
            sizes = 40 * norm_values + 5

            # Calculate alpha values based on activation strength
            alphas = 0.3 + 0.4 * norm_values

            # Plot density as scatter with varying alpha
            rgba_colors = np.array([to_rgba(input_color, alpha=a) for a in alphas])

            ax.scatter(
                x_coords,
                y_coords,
                z_coords,
                c=rgba_colors,
                s=sizes,
                edgecolors="none",
                depthshade=True,
            )
        
        # Sample individual neurons for overlay
        total_neurons = len(input_data)
        sample_size = int(total_neurons * (max_neurons_percentage / 100))
        if sample_size < total_neurons:
            neuron_sample = input_data.sample(
                sample_size, random_state=1234
            )
        else:
            neuron_sample = input_data

        # Scale activation values for better visualization
        max_activation = neuron_sample["input"].max()
        if max_activation > 0:
            normalized_activation = neuron_sample["input"] / max_activation
        else:
            normalized_activation = neuron_sample["input"]

        # Calculate point sizes based on activation strength
        neuron_sizes = 15 * normalized_activation + 3

        # Plot individual neurons with higher opacity for emphasis
        ax.scatter(
            neuron_sample["pos_x"],
            neuron_sample["pos_y"],
            neuron_sample["pos_z"],
            c=[input_color],
            s=neuron_sizes,
            alpha=0.7,
            edgecolors="none",
            depthshade=True,
        )

        # Set consistent view angle
        ax.view_init(elev=30, azim=45)

        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Process each activation step (columns)
        for step in range(1, num_steps + 1):
            # Get subplot position (shifted by 1 to account for input column)
            if len(propagations_dict) > 1:
                ax = axes[i, step]
            else:
                ax = axes[step]

            # Style the subplot
            ax.set_facecolor("white")
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("lightgray")
            ax.yaxis.pane.set_edgecolor("lightgray")
            ax.zaxis.pane.set_edgecolor("lightgray")

            # Remove tick labels
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # Add axis labels only for the bottom-right plot
            if i == len(propagations_dict) - 1 and step == num_steps:  # Bottom-right plot
                ax.set_xlabel("X", labelpad=-10)
                ax.set_ylabel("Y", labelpad=-10)
                ax.set_zlabel("Z", labelpad=-10)

            # Get color for this step
            color = step_colors[step - 1]

            # Filter data for this step
            act_col = f"activation_{step}"
            step_data = merged_data[merged_data[act_col] > 0].copy()

            # Add title to the top row only
            if i == 0:
                ax.set_title(f"Step {step}", pad=5)

            # Skip if no data
            if len(step_data) == 0:
                continue
            
            if voxel_size is not None:
                # Create 3D histogram for density visualization
                x_bins = np.linspace(x_min, x_max, voxel_size)
                y_bins = np.linspace(y_min, y_max, voxel_size)
                z_bins = np.linspace(z_min, z_max, voxel_size)

                H, edges = np.histogramdd(
                    step_data[["pos_x", "pos_y", "pos_z"]].values,
                    bins=[x_bins, y_bins, z_bins],
                    weights=step_data[act_col].values,
                )

                # Apply Gaussian smoothing
                H_smooth = gaussian_filter(H, sigma=smoothing)

                # Skip if all zeros
                if H_smooth.max() == 0:
                    continue

                # Get voxel coordinates
                x_centers = (x_bins[:-1] + x_bins[1:]) / 2
                y_centers = (y_bins[:-1] + y_bins[1:]) / 2
                z_centers = (z_bins[:-1] + z_bins[1:]) / 2

                # Create meshgrid of voxel centers
                X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

                # Get non-zero voxels (above threshold of max value)
                threshold = 0.25 * H_smooth.max()
                mask = H_smooth > threshold

                # Skip if nothing to plot
                if not np.any(mask):
                    continue

                # Get coordinates and values of voxels above threshold
                x_coords = X[mask]
                y_coords = Y[mask]
                z_coords = Z[mask]
                values = H_smooth[mask]

                # Normalize values for sizing
                norm_values = values / values.max()

                # Calculate sizes based on activation strength
                sizes = 40 * norm_values + 5

                # Calculate alpha values based on activation strength
                alphas = 0.3 + 0.4 * norm_values

                # Plot density as scatter with varying alpha
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
            
            # Sample individual neurons for overlay
            total_neurons = len(step_data)
            sample_size = int(total_neurons * (max_neurons_percentage / 100))
            if sample_size < total_neurons:
                neuron_sample = step_data.sample(
                    sample_size, random_state=step + 1234
                )
            else:
                neuron_sample = step_data

            # Scale activation values for better visualization
            max_activation = neuron_sample[act_col].max()
            if max_activation > 0:
                normalized_activation = neuron_sample[act_col] / max_activation
            else:
                normalized_activation = neuron_sample[act_col]

            # Calculate point sizes based on activation strength
            neuron_sizes = 15 * normalized_activation + 3

            # Plot individual neurons with higher opacity for emphasis
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

            # Set consistent view angle
            ax.view_init(elev=30, azim=45)

            # Set axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

    plt.subplots_adjust(wspace=-0.5, hspace=-0.06)

    return fig


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
    # --- Ensure alphas are in the valid [0, 1] range ------------------------
    alphas = np.asarray(alphas, dtype=float)
    if alphas.size:
        # If values are outside [0,1], rescale by the maximum positive value.
        if (alphas > 1).any() or (alphas < 0).any():
            max_val = alphas.max()
            if max_val > 0:
                alphas = alphas / max_val
        # Finally, clip to be absolutely safe
        alphas = np.clip(alphas, 0.0, 1.0)

    # Set axis properties for a clean look
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(False)

    # Add padding to bounds
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

    # Handle case with no active neurons
    if positions.shape[0] == 0:
        return

    # Option 1: Voxel-based heatmap with Gaussian smoothing
    if voxel_size:
        # Create a 3D histogram (voxel grid)
        bins = [
            np.arange(x_min, x_max + voxel_size, voxel_size),
            np.arange(y_min, y_max + voxel_size, voxel_size),
            np.arange(z_min, z_max + voxel_size, voxel_size),
        ]
        hist, _ = np.histogramdd(positions, bins=bins, weights=alphas)

        # Normalize and smooth
        if hist.max() > 0:
            hist /= hist.max()
        if smoothing:
            hist = gaussian_filter(hist, sigma=smoothing)

        # Get voxel centers and values for plotting
        x_centers = (bins[0][:-1] + bins[0][1:]) / 2
        y_centers = (bins[1][:-1] + bins[1][1:]) / 2
        z_centers = (bins[2][:-1] + bins[2][1:]) / 2
        x, y, z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

        # Create RGBA colors with varying alpha
        rgba_colors = np.zeros(hist.shape + (4,))
        # Get base color
        base_color = to_rgba(color)
        rgba_colors[..., :3] = base_color[:3]  # Set RGB
        rgba_colors[..., 3] = hist  # Set alpha based on density

        # Plot voxels
        ax.scatter(
            x.flatten(),
            y.flatten(),
            z.flatten(),
            c=rgba_colors.reshape(-1, 4),
            marker="s",
            s=voxel_size**2 * 0.8,  # Adjust marker size
            edgecolors="none",
        )

    # Option 2: Scatter plot (if not using voxels)
    else:
        # Create RGBA colors with varying alpha based on activation strength
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

    # Add a single point for the legend
    ax.scatter(
        [],
        [],
        [],
        c=[color],
        s=100,
        edgecolors="none",
        label=label,
    )
    # add a title to the subplot
    ax.set_title(title, fontsize=16, pad=-20)
    
    # set aspect ratio
    ax.set_aspect('equal', 'box')

