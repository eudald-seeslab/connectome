import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import to_rgba
import matplotlib as mpl
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke

from notebooks.visualization.activations_funcs import split_title
from utils.helpers import compute_individual_synapse_lengths, compute_total_synapse_length


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
    # Set Nature style parameters
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
        }
    )

    # Nature-friendly color scheme
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]

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

    # print the activation percentages
    print(activation_percentages)

    # Create figure for activation percentages
    fig1, ax1 = plt.subplots(figsize=(fig_width_in, fig_height_in))
    for i, config in enumerate(configs):
        ax1.plot(
            range(1, num_steps + 1),
            activation_percentages[config],
            marker="o",
            label=config,
            color=colors[i % len(colors)],
            linewidth=1.2,
            markersize=3.5,
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
            marker="o",
            label=config,
            color=colors[i % len(colors)],
            linewidth=1.2,
            markersize=3.5,
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
            marker="o",
            label=config,
            color=colors[i % len(colors)],
            linewidth=1.2,
            markersize=3.5,
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


def plot_synapse_length_distributions(neuron_coords, conns_dict, use_density=True, num_confidence_interval_se=1):
    """
    Plot synapse length distributions for multiple network types.
    
    Parameters:
    -----------
    neuron_coords : DataFrame
        Contains neuron coordinates
    conns_dict : dict
        Dictionary of network types with their connection DataFrames
    use_density : bool, default=True
        Whether to normalize histograms to density
    num_confidence_interval_se : int, default=1
        Number of standard errors for confidence interval bands
        
    Returns:
    --------
    tuple: (fig1, fig2) - Two figure objects for histogram and synapse strength vs distance
    """
    titles = list(conns_dict.keys())
    n_plots = len(titles)
    
    # Ensure we have no more than 6 plots
    if n_plots > 6:
        raise ValueError(f"Too many networks to plot ({n_plots}). Maximum supported is 6.")
    
    # Extended color palette for up to 6 plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:n_plots]

    # Pre-calculate distances for each dataframe
    dists = {name: compute_individual_synapse_lengths(df, neuron_coords)
            for name, df in conns_dict.items()}
    weights = {name: df["syn_count"].to_numpy()
              for name, df in conns_dict.items()}

    # Get 99 percentile of all distances to avoid outliers
    all_d = np.concatenate(list(dists.values()))
    max_len = np.percentile(all_d, 99)
    bins = np.linspace(0, max_len, 100)

    # Get common y-max for all plots
    max_val = 0
    for name in titles:
        hist, _ = np.histogram(dists[name], bins=bins,
                              weights=weights[name], density=use_density)
        max_val = max(max_val, hist.max())
    max_val *= 1.1  # Add a small margin

    # ——— Figure 1: Histogram of distance distribution ———
    fig1, axs1 = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots), 
                             sharex=True, constrained_layout=True)
    
    # Ensure axs1 is always iterable (when n_plots=1, axs1 is a single Axes object)
    if n_plots == 1:
        axs1 = [axs1]

    total_mm = {}  # Total wiring lengths for annotation

    for ax, title, col in zip(axs1, titles, colors):
        w = weights[title]
        L = dists[title]
        ax.hist(L, bins=bins, weights=w, density=use_density,
               color=col, alpha=0.7)

        # Weighted mean
        mean_nm = np.average(L, weights=w)
        ax.axvline(mean_nm, ls='--', c='k', lw=1)
        # Display mean in µm
        ax.text(mean_nm*1.05, 0.7*max_val,
               f"Mean: {mean_nm / 1e3:,.2f} µm", fontsize=12)

        # Total wiring length (m)
        tot_nm = float(np.sum(L * w))
        tot_m = tot_nm / 1e12
        total_mm[title] = tot_m
        ax.text(0.95, 0.85, f"Total: {tot_m:,.2f} km",
               transform=ax.transAxes, ha='right',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_ylim(0, max_val)
        ax.set_ylabel("Density" if use_density else "Count")
        ax.set_title(title)

    axs1[-1].set_xlabel("Synapse Length (nm)")
    
    # ——— Figure 2: Synapse strength vs distance ———
    # Create bins for distance ranges
    bin_edges = np.linspace(0, max_len, 20)  # Fewer bins for better statistics
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig2, axs2 = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots), 
                             sharex=True, constrained_layout=True)
    
    # Ensure axs2 is always iterable
    if n_plots == 1:
        axs2 = [axs2]
    
    for ax, title, col in zip(axs2, titles, colors):
        L = dists[title]
        w = weights[title]
        
        # Compute statistics for each bin
        means = []
        errors = []
        
        for i in range(len(bin_edges) - 1):
            mask = (L >= bin_edges[i]) & (L < bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_weights = w[mask]
                mean_weight = np.mean(bin_weights)
                # Standard error = std / sqrt(n)
                std_err = np.std(bin_weights) / np.sqrt(len(bin_weights))
                means.append(mean_weight)
                errors.append(std_err)
            else:
                means.append(0)
                errors.append(0)
        
        # Plot the mean line
        ax.plot(bin_centers, means, 'o-', color=col, markersize=5, alpha=0.9, label=title)
        
        # Add confidence interval bands
        upper_bound = [m + e * num_confidence_interval_se for m, e in zip(means, errors)]
        lower_bound = [m - e * num_confidence_interval_se for m, e in zip(means, errors)]
        ax.fill_between(bin_centers, lower_bound, upper_bound, color=col, alpha=0.2)
        
        ax.set_ylabel("Avg. Synapse Count")
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    axs2[-1].set_xlabel("Synapse Length (nm)")
    plt.tight_layout()

    return fig1, fig2


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

def accuracy_comparison():
    # Network configurations
    networks = [
        "Biological",
        "Random\nunconstrained",
        "Random\npruned",
        "Random\nbin-wise",
    ]

    # Made-up performance data for three tasks (percentage accuracy)
    numerical_task = [84, 92, 91, 82] 
    color_task = [100, 100, 100, 100] 
    shape_task = [64, 69, 69, 60] 

    # Made-up error bars (standard error)
    numerical_err = [0, 0, 0, 0.01]
    color_err = [0, 0, 0, 0]
    shape_err = [0, 0.01, 0, 0.04]

    # Nature color palette
    colors = ["#4878D0", "#6ACC64", "#EE854A"]

    # Set width of bars
    bar_width = 0.24
    capsize = 2
    index = np.arange(len(networks))

    # Create the figure with Nature-compatible dimensions
    fig3, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Create grouped bars in the requested order with error bars

    bars = ax.bar(
        index - bar_width,
        color_task,
        bar_width,
        yerr=color_err,
        label="Color\nDiscrimination",
        color=colors[1],
        alpha=0.9,
        capsize=capsize,
        ecolor="black",
        error_kw={"elinewidth": 1},
    )

    bars = ax.bar(
        index,
        numerical_task,
        bar_width,
        yerr=numerical_err,
        label="Numerical\nDiscrimination",
        color=colors[0],
        alpha=0.9,
        capsize=capsize,
        ecolor="black",
        error_kw={"elinewidth": 1},
    )
    bars = ax.bar(
        index + bar_width,
        shape_task,
        bar_width,
        yerr=shape_err,
        label="Shape\nRecognition",
        color=colors[2],
        alpha=0.9,
        capsize=capsize,
        ecolor="black",
        error_kw={"elinewidth": 1},
    )

    # Add horizontal line for chance level (50% for binary classification)
    ax.axhline(y=50, linestyle="--", color="#666666", alpha=0.5, linewidth=1)

    # Add text label for chance level
    ax.text(len(networks) - 1.35, 45, "Chance level", fontsize=10, color="#666666", 
            bbox=dict(facecolor='white', edgecolor='#666666', boxstyle='round,pad=0.5', alpha=0.8))

    # Add labels and custom x-axis tick labels
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)  # Slightly higher to accommodate error bars
    ax.set_xticks(index)
    ax.set_xticklabels(networks, fontsize=12, rotation=90)  # Vertical labels

    # Add a legend
    ax.legend(fontsize=12, loc="lower right", framealpha=0.9)

    # Adjust layout with extra bottom margin for vertical labels
    plt.tight_layout()
    fig3.subplots_adjust(bottom=0.2)  # Make room for vertical labels

    return fig3

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

        config_name = split_title(config_name)

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

def plot_synapse_counts_histogram(conns_dict, bins=30, figsize=(12, 8), log_scale=False):
    """
    Plot simple histograms of synapse counts for each network type.
    
    Parameters:
    -----------
    conns_dict : dict
        Dictionary of network types with their connection DataFrames
    bins : int or list, default=30
        Number of bins or bin edges for histogram
    figsize : tuple, default=(12, 8)
        Figure size (width, height)
    log_scale : bool, default=False
        Whether to use log scale for y-axis
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with histograms
    """
    titles = list(conns_dict.keys())
    n_plots = len(titles)
    
    # Ensure we have no more than 6 plots
    if n_plots > 6:
        raise ValueError(f"Too many networks to plot ({n_plots}). Maximum supported is 6.")
    
    # Extended color palette for up to 6 plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:n_plots]
    
    # Create figure with subplots (one per network)
    fig, axs = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, constrained_layout=True)
    
    # Ensure axs is always iterable (when n_plots=1, axs is a single Axes object)
    if n_plots == 1:
        axs = [axs]
        
    for ax, title, color in zip(axs, titles, colors):
        # Get synapse counts for this network
        syn_counts = conns_dict[title]["syn_count"].values
        
        # Plot histogram
        ax.hist(syn_counts, bins=bins, color=color, alpha=0.7)
        
        # Calculate statistics
        mean_count = np.mean(syn_counts)
        median_count = np.median(syn_counts)
        max_count = np.max(syn_counts)
        total_synapses = np.sum(syn_counts)
        
        # Add statistics as text
        stats_text = (f"Mean: {mean_count:.2f}\n"
                     f"Median: {median_count:.2f}\n"
                     f"Max: {max_count:.2f}\n"
                     f"Total: {total_synapses:,}")
        
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add title and labels
        ax.set_title(title)
        ax.set_ylabel("Count")
        
        # Set log scale if requested
        if log_scale:
            ax.set_yscale('log')
            
    # Add x-label to bottom subplot only
    axs[-1].set_xlabel("Synapse Count")
    
    return fig
