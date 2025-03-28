import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy.ndimage import gaussian_filter
from matplotlib.colors import to_rgba


def visualize_connectome_activation(
    propagation, neuron_position_data, input_value=None, bin_size=50
):
    """
    Create an interactive 3D visualization of activation density in the drosophila connectome.

    Parameters:
    -----------
    propagation : DataFrame
        The propagation dataframe with root_id, input, and activation_1 through activation_4
    neuron_position_data : DataFrame
        The neuron position dataframe with root_id, pos_x, pos_y, pos_z
    input_value : optional
        If specified, filter for a specific input value
    bin_size : int
        Number of bins for each dimension (controls visualization granularity)

    Returns:
    --------
    plotly.graph_objects.Figure
        An interactive 3D visualization figure
    """
    # Prepare the merged data
    all_data = []

    for step in range(1, 5):
        # Get the activation column for this step
        activation_col = f"activation_{step}"

        # Skip if this column doesn't exist
        if activation_col not in propagation.columns:
            continue

        # Filter for a specific input if provided
        if input_value is not None:
            step_data = propagation[propagation["input"] == input_value].copy()
        else:
            step_data = propagation.copy()

        # Merge with position data
        merged = pd.merge(
            step_data[["root_id", activation_col]],
            neuron_position_data,
            on="root_id",
            how="inner",
        )

        # Skip if no data for this step
        if len(merged) == 0:
            continue

        # Add to our collection
        merged["step"] = step
        merged["activation"] = merged[activation_col]
        all_data.append(
            merged[["root_id", "step", "activation", "pos_x", "pos_y", "pos_z"]]
        )

    # Combine all data
    if not all_data:
        raise ValueError("No valid data found for visualization")

    combined_data = pd.concat(all_data)

    # Get the overall min and max coordinates for consistent axes
    x_min, x_max = combined_data["pos_x"].min(), combined_data["pos_x"].max()
    y_min, y_max = combined_data["pos_y"].min(), combined_data["pos_y"].max()
    z_min, z_max = combined_data["pos_z"].min(), combined_data["pos_z"].max()

    # Define the bin edges for all dimensions
    x_bins = np.linspace(x_min, x_max, bin_size)
    y_bins = np.linspace(y_min, y_max, bin_size)
    z_bins = np.linspace(z_min, z_max, bin_size)

    # Define colors for each step - using distinct colorscales
    step_colors = ["Blues", "Greens", "Oranges", "Reds"]

    # Create figure
    fig = go.Figure()

    # Process each step
    for step in range(1, 5):
        step_data = combined_data[combined_data["step"] == step]

        # Skip if no data for this step
        if len(step_data) == 0:
            continue

        # Create a 3D histogram using histogram binning
        H, edges = np.histogramdd(
            step_data[["pos_x", "pos_y", "pos_z"]].values,
            bins=[x_bins, y_bins, z_bins],
            weights=step_data["activation"].values,
        )

        # Get non-zero bins for plotting (to reduce visual clutter)
        non_zero_indices = np.where(H > 0)

        if len(non_zero_indices[0]) == 0:
            continue

        # Get bin centers
        x_centers = (edges[0][:-1] + edges[0][1:]) / 2
        y_centers = (edges[1][:-1] + edges[1][1:]) / 2
        z_centers = (edges[2][:-1] + edges[2][1:]) / 2

        x = x_centers[non_zero_indices[0]]
        y = y_centers[non_zero_indices[1]]
        z = z_centers[non_zero_indices[2]]
        intensity = H[non_zero_indices]

        # Normalize intensity for better visualization
        if np.max(intensity) > 0:
            norm_intensity = intensity / np.max(intensity)
            # Size based on activation intensity (smaller range to avoid large markers)
            size = 2 + 8 * norm_intensity

            # Add trace for this step
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=intensity,
                        colorscale=step_colors[step - 1],
                        opacity=0.7,
                        colorbar=dict(
                            title=f"Activation (Step {step})",
                            len=0.5,
                            y=0.8 - (step - 1) * 0.2,
                            yanchor="top",
                            thickness=20,
                        ),
                    ),
                    name=f"Step {step}",
                    visible=(step == 1),  # Only first step visible initially
                )
            )

    # Create step selection buttons
    step_buttons = []
    for step in range(1, 5):
        visible_array = [trace.name == f"Step {step}" for trace in fig.data]
        step_buttons.append(
            dict(
                method="update",
                label=f"Step {step}",
                args=[
                    {"visible": visible_array},
                    {"title": f"Activation Density - Step {step}"},
                ],
            )
        )

    # Add a button to show all steps together
    all_visible_button = dict(
        method="update",
        label="All Steps",
        args=[
            {"visible": [True] * len(fig.data)},
            {"title": "Activation Density - All Steps"},
        ],
    )
    step_buttons.append(all_visible_button)

    # Add buttons menu
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=step_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ]
    )

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X Position"),
            yaxis=dict(title="Y Position"),
            zaxis=dict(title="Z Position"),
            aspectmode="cube",
        ),
        title=(
            f"Connectome Activation Density (Input: {input_value})"
            if input_value is not None
            else "Connectome Activation Density"
        ),
        height=800,
        width=1000,
        legend=dict(orientation="h"),
    )

    return fig


def compare_activation_patterns(
    propagation_dfs, neuron_position_data, labels, input_value=None
):
    """
    Compare activation patterns across different connectome configurations.

    Parameters:
    -----------
    propagation_dfs : list of DataFrames
        List of propagation dataframes for different connectome configurations
    neuron_position_data : DataFrame
        The neuron position dataframe
    labels : list of str
        Labels for each connectome configuration
    input_value : optional
        If specified, filter for a specific input value

    Returns:
    --------
    plotly.graph_objects.Figure
        A figure with comparison metrics
    """
    # Calculate activation metrics for each configuration
    activation_metrics = []

    for config_idx, (label, prop_df) in enumerate(zip(labels, propagation_dfs)):
        # Filter for a specific input if provided
        if input_value is not None:
            filtered_df = prop_df[prop_df["input"] == input_value].copy()
        else:
            filtered_df = prop_df.copy()

        # Calculate metrics for each activation step
        for step in range(1, 5):
            activation_col = f"activation_{step}"

            # Skip if column doesn't exist
            if activation_col not in filtered_df.columns:
                continue

            # Count active neurons (activation > 0)
            active_neurons = filtered_df[filtered_df[activation_col] > 0]
            active_count = len(active_neurons)
            total_count = len(filtered_df)

            # Calculate percentage of active neurons
            percent_active = (
                (active_count / total_count) * 100 if total_count > 0 else 0
            )

            # Calculate average activation among active neurons
            avg_activation = (
                active_neurons[activation_col].mean() if active_count > 0 else 0
            )

            # Calculate total activation
            total_activation = filtered_df[activation_col].sum()

            # Store metrics
            activation_metrics.append(
                {
                    "Configuration": label,
                    "Step": step,
                    "Active Neurons (%)": percent_active,
                    "Average Activation": avg_activation,
                    "Total Activation": total_activation,
                }
            )

    # Convert to dataframe
    metrics_df = pd.DataFrame(activation_metrics)

    # Create a figure with subplots for different metrics
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Percentage of Active Neurons",
            "Average Activation (Active Neurons)",
            "Total Network Activation",
        ],
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
    )

    # Add traces for each configuration
    colors = ["blue", "red", "green", "purple", "orange", "cyan", "brown", "pink"]

    for i, config in enumerate(labels):
        config_data = metrics_df[metrics_df["Configuration"] == config]

        # Sort by step
        config_data = config_data.sort_values("Step")

        # Add percentage active trace
        fig.add_trace(
            go.Scatter(
                x=config_data["Step"],
                y=config_data["Active Neurons (%)"],
                mode="lines+markers",
                name=config,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=8),
            ),
            row=1,
            col=1,
        )

        # Add average activation trace
        fig.add_trace(
            go.Scatter(
                x=config_data["Step"],
                y=np.log(config_data["Average Activation"]),
                mode="lines+markers",
                name=config,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=8),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Add total activation trace
        fig.add_trace(
            go.Scatter(
                x=config_data["Step"],
                y=np.log(config_data["Total Activation"]),
                mode="lines+markers",
                name=config,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=8),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    # Update axis labels and layout
    for i in range(1, 4):
        fig.update_xaxes(title_text="Message Passing Step", row=1, col=i)

    fig.update_yaxes(title_text="% of Neurons", row=1, col=1)
    fig.update_yaxes(title_text="Log Avg. Activation Value", row=1, col=2)
    fig.update_yaxes(title_text="Log Sum of Activation", row=1, col=3)

    # Update layout
    fig.update_layout(
        title=f"Activation Comparison Across Connectome Configurations"
        + (f" (Input: {input_value})" if input_value is not None else ""),
        height=500,
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def generate_slice_visualization(
    propagation,
    neuron_position_data,
    input_value=None,
    slice_axis="z",
    num_slices=3,
    bin_size=100,
):
    """
    Create 2D slice visualizations of the connectome activation.

    Parameters:
    -----------
    propagation : DataFrame
        The propagation dataframe
    neuron_position_data : DataFrame
        The neuron position dataframe
    input_value : optional
        If specified, filter for a specific input value
    slice_axis : str
        Axis along which to create slices ('x', 'y', or 'z')
    num_slices : int
        Number of slices to create
    bin_size : int
        Number of bins for the 2D histograms

    Returns:
    --------
    matplotlib.figure.Figure
        A figure with 2D slice visualizations
    """
    # Prepare the data
    all_data = []

    for step in range(1, 5):
        activation_col = f"activation_{step}"

        # Skip if this column doesn't exist
        if activation_col not in propagation.columns:
            continue

        # Filter for a specific input if provided
        if input_value is not None:
            step_data = propagation[propagation["input"] == input_value].copy()
        else:
            step_data = propagation.copy()

        # Merge with position data
        merged = pd.merge(
            step_data[["root_id", activation_col]],
            neuron_position_data,
            on="root_id",
            how="inner",
        )

        # Skip if no data for this step
        if len(merged) == 0:
            continue

        # Add to our collection
        merged["step"] = step
        merged["activation"] = merged[activation_col]
        all_data.append(
            merged[["root_id", "step", "activation", "pos_x", "pos_y", "pos_z"]]
        )

    combined_data = pd.concat(all_data)

    # Define the axes
    axes = {"x": 0, "y": 1, "z": 2}
    slice_idx = axes[slice_axis]
    other_axes = [ax for ax in ["x", "y", "z"] if ax != slice_axis]

    # Get the min/max for the slice dimension
    slice_col = f"pos_{slice_axis}"
    slice_min, slice_max = (
        combined_data[slice_col].min(),
        combined_data[slice_col].max(),
    )

    # Calculate positions for the slices
    slice_positions = np.linspace(slice_min, slice_max, num_slices + 2)[1:-1]
    slice_width = (slice_max - slice_min) / (num_slices + 1) / 2

    # Define colorscales for each step
    cmap_list = [plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds]

    # Create the figure
    fig, axes_array = plt.subplots(4, num_slices, figsize=(4 * num_slices, 16))

    # Process each step and slice
    for step in range(1, 5):
        step_data = combined_data[combined_data["step"] == step]

        if len(step_data) == 0:
            continue

        for i, slice_pos in enumerate(slice_positions):
            # Filter data near this slice
            slice_data = step_data[
                (step_data[slice_col] >= slice_pos - slice_width)
                & (step_data[slice_col] <= slice_pos + slice_width)
            ]

            if len(slice_data) == 0:
                continue

            # Get the subplot
            ax = axes_array[step - 1, i]

            # Create a 2D histogram
            h, xedges, yedges = np.histogram2d(
                slice_data[f"pos_{other_axes[0]}"],
                slice_data[f"pos_{other_axes[1]}"],
                bins=bin_size,
                weights=slice_data["activation"],
            )

            # Plot the heatmap
            im = ax.imshow(
                h.T,  # Transpose for correct orientation
                origin="lower",
                extent=[
                    slice_data[f"pos_{other_axes[0]}"].min(),
                    slice_data[f"pos_{other_axes[0]}"].max(),
                    slice_data[f"pos_{other_axes[1]}"].min(),
                    slice_data[f"pos_{other_axes[1]}"].max(),
                ],
                cmap=cmap_list[step - 1],
                aspect="auto",
            )

            # Add labels
            if step == 4:
                ax.set_xlabel(f"{other_axes[0].upper()}")
            if i == 0:
                ax.set_ylabel(f"{other_axes[1].upper()}")

            # Add slice position as title
            ax.set_title(f"{slice_axis.upper()}={slice_pos:.1f}")

            # Add colorbar
            if i == num_slices - 1:
                plt.colorbar(im, ax=ax, label=f"Activation (Step {step})")

    # Set overall title
    plt.suptitle(
        f"Connectome Activation - 2D Slices Along {slice_axis.upper()} Axis"
        + (f" (Input: {input_value})" if input_value is not None else ""),
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    return fig


# Example usage:
# For a single configuration
# fig = visualize_connectome_activation(propagation_df, neuron_position_data)
# fig.show()

# For comparing configurations:
# fig_comparison = compare_activation_patterns(
#     [original_df, random_df],
#     neuron_position_data,
#     ["Original Connectome", "Randomized Connectome"]
# )
# fig_comparison.show()

# For a 2D slice visualization:
# fig_slices = generate_slice_visualization(propagation_df, neuron_position_data, slice_axis='z', num_slices=3)
# plt.show()


def create_nature_style_projection(
    propagation, neuron_position_data, input_value=None, step=1, fig_width=89, dpi=300
):
    """
    Create a Nature-style 2D projection visualization of connectome activation.

    Parameters:
    -----------
    propagation : DataFrame
        The propagation dataframe with root_id, input, and activation columns
    neuron_position_data : DataFrame
        The neuron position dataframe with root_id, pos_x, pos_y, pos_z
    input_value : optional
        If specified, filter for a specific input value
    step : int
        Which activation step to visualize (1-4)
    fig_width : int
        Width in mm (Nature single column is 89mm)
    dpi : int
        Resolution (Nature requires at least 300 dpi)

    Returns:
    --------
    matplotlib.figure.Figure
        A Nature-style figure with 2D projections
    """
    # Set Nature style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "savefig.dpi": dpi,
            "savefig.format": "tiff",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    # Convert mm to inches (1 mm = 0.0393701 inches)
    fig_width_in = fig_width * 0.0393701

    # Prepare data
    activation_col = f"activation_{step}"

    # Filter for a specific input if provided
    if input_value is not None:
        step_data = propagation[propagation["input"] == input_value].copy()
    else:
        step_data = propagation.copy()

    # Ensure the activation column exists
    if activation_col not in step_data.columns:
        raise ValueError(f"Activation column {activation_col} not found in data")

    # Merge with position data
    merged = pd.merge(
        step_data[["root_id", activation_col]],
        neuron_position_data,
        on="root_id",
        how="inner",
    )

    # Keep only active neurons (activation > 0)
    active_neurons = merged[merged[activation_col] > 0].copy()

    if len(active_neurons) == 0:
        raise ValueError("No active neurons found for this step/input")

    # Create the figure with 3 panels (xy, xz, yz projections)
    # Adjust height based on golden ratio
    fig_height_in = fig_width_in / 1.618
    fig, axes = plt.subplots(1, 3, figsize=(fig_width_in, fig_height_in))

    # Custom colormap similar to Nature style
    colors = [(0.95, 0.95, 0.95, 0), (0.0, 0.0, 0.0, 1)]
    cmap = LinearSegmentedColormap.from_list("nature_cmap", colors)

    # Create the projections
    projections = [
        {"axes": (0, 1), "labels": ("x", "y"), "title": "Top View", "ax": axes[0]},
        {"axes": (0, 2), "labels": ("x", "z"), "title": "Side View", "ax": axes[1]},
        {"axes": (1, 2), "labels": ("y", "z"), "title": "Front View", "ax": axes[2]},
    ]

    # Scale for marker sizes based on activation
    max_activation = active_neurons[activation_col].max()
    min_activation = active_neurons[activation_col].min()

    # Normalize activations for color intensity
    norm_activations = (active_neurons[activation_col] - min_activation) / (
        max_activation - min_activation
    )
    norm_activations = np.clip(norm_activations, 0.05, 1.0)  # Ensure minimum visibility

    # Plot each projection
    for proj in projections:
        ax = proj["ax"]
        x_idx, y_idx = proj["axes"]
        pos_cols = ["pos_x", "pos_y", "pos_z"]

        x_data = active_neurons[pos_cols[x_idx]]
        y_data = active_neurons[pos_cols[y_idx]]

        # Create a 2D kernel density estimate
        try:
            # For very sparse data, KDE might fail - handle this case
            if len(x_data) >= 10:  # Need reasonable number of points for KDE
                xy = np.vstack([x_data, y_data])
                z = gaussian_kde(xy)(xy)

                # Sort the points by density for better visualization
                idx = z.argsort()
                x_sorted, y_sorted = x_data.iloc[idx], y_data.iloc[idx]
                activation_sorted = active_neurons[activation_col].iloc[idx]
                norm_act_sorted = norm_activations.iloc[idx]

                # Plot as scatter with transparency based on activation
                scatter = ax.scatter(
                    x_sorted,
                    y_sorted,
                    c=activation_sorted,
                    cmap="viridis",
                    s=1 + 4 * norm_act_sorted,  # Size based on activation
                    alpha=0.7,
                    edgecolors="none",
                )
            else:
                # For very sparse data, just plot the points
                scatter = ax.scatter(
                    x_data,
                    y_data,
                    c=active_neurons[activation_col],
                    cmap="viridis",
                    s=1 + 4 * norm_activations,
                    alpha=0.7,
                    edgecolors="none",
                )
        except np.linalg.LinAlgError:
            # Fallback for KDE failure
            scatter = ax.scatter(
                x_data,
                y_data,
                c=active_neurons[activation_col],
                cmap="viridis",
                s=1 + 4 * norm_activations,
                alpha=0.7,
                edgecolors="none",
            )

        # Add contour lines to help visualize density
        try:
            if len(x_data) >= 20:  # Need enough points for valid contours
                sns.kdeplot(
                    x=x_data,
                    y=y_data,
                    ax=ax,
                    levels=4,
                    colors=["#333333"],
                    linewidths=0.5,
                    alpha=0.5,
                )
        except Exception:
            # Skip contours if they can't be calculated
            pass

        # Set labels
        ax.set_xlabel(f'{proj["labels"][0]} position')
        ax.set_ylabel(f'{proj["labels"][1]} position')
        ax.set_title(proj["title"])

        # Set tight, equal aspect ratio
        ax.set_aspect("auto")  # 'equal' would distort if scales are very different

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(f"Activation (Step {step})")

    # Add percentage of active neurons as text
    active_percent = len(active_neurons) / len(step_data) * 100
    fig.text(
        0.5,
        0.02,
        f"Active neurons: {active_percent:.1f}% ({len(active_neurons)} of {len(step_data)})",
        ha="center",
        fontsize=7,
    )

    # Set overall title
    title = f"Connectome Activation - Message Passing Step {step}"
    if input_value is not None:
        title += f" (Input: {input_value})"
    fig.suptitle(title)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])

    return fig


def create_activation_comparison_plot(
    propagation_dfs, labels, colors=None, fig_width=183, dpi=300
):
    """
    Create a Nature-style comparison plot showing percentage of active neurons,
    average activation, and total activation across different connectome configurations.

    Parameters:
    -----------
    propagation_dfs : list of DataFrames
        List of propagation dataframes for different connectome configurations
    labels : list of str
        Labels for each connectome configuration
    colors : list of str, optional
        Colors for each configuration
    fig_width : int
        Width in mm (Nature double column is 183mm)
    dpi : int
        Resolution (Nature requires at least 300 dpi)

    Returns:
    --------
    matplotlib.figure.Figure
        A Nature-style figure with comparison plots
    """
    # Set Nature style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "savefig.dpi": dpi,
            "savefig.format": "tiff",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    # Default Nature-friendly colors if not provided
    if colors is None:
        colors = [
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#56B4E9",
            "#E69F00",
            "#F0E442",
        ]

    # Convert mm to inches (1 mm = 0.0393701 inches)
    fig_width_in = fig_width * 0.0393701

    # Figure height (golden ratio)
    fig_height_in = fig_width_in / 2.5  # Less height for horizontal layout

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(fig_width_in, fig_height_in))

    # Calculate activation metrics for each configuration
    all_metrics = []

    for config_idx, (label, prop_df) in enumerate(zip(labels, propagation_dfs)):
        metrics = []

        # Calculate metrics for each activation step
        for step in range(1, 5):
            activation_col = f"activation_{step}"

            # Skip if column doesn't exist
            if activation_col not in prop_df.columns:
                continue

            # Count active neurons (activation > 0)
            active_neurons = prop_df[prop_df[activation_col] > 0]
            active_count = len(active_neurons)
            total_count = len(prop_df)

            # Calculate percentage of active neurons
            percent_active = (
                (active_count / total_count) * 100 if total_count > 0 else 0
            )

            # Calculate average activation among active neurons
            avg_activation = (
                active_neurons[activation_col].mean() if active_count > 0 else 0
            )

            # Calculate total activation
            total_activation = prop_df[activation_col].sum()

            # Store metrics
            metrics.append(
                {
                    "Step": step,
                    "Active Neurons (%)": percent_active,
                    "Average Activation": avg_activation,
                    "Total Activation": total_activation,
                }
            )

        # Add to overall metrics with configuration label
        metrics_df = pd.DataFrame(metrics)
        metrics_df["Configuration"] = label
        all_metrics.append(metrics_df)

    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)

    # Plot titles and y-labels
    titles = ["Active Neurons (%)", "Average Activation", "Total Activation"]
    ylabels = ["% of Neurons", "Avg. Activation Value", "Sum of Activation"]

    # Plot data
    for i, (title, ylabel) in enumerate(zip(titles, ylabels)):
        ax = axes[i]
        y_col = title

        # Plot each configuration
        for j, config in enumerate(labels):
            config_data = combined_metrics[combined_metrics["Configuration"] == config]

            # Plot line
            ax.plot(
                config_data["Step"],
                config_data[y_col],
                marker="o",
                label=config,
                color=colors[j % len(colors)],
                linewidth=1.5,
                markersize=4,
            )

        # Set labels and title
        ax.set_xlabel("Message Passing Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Set y-axis to log scale for better visibility (for avg and total activation)
        if i > 0:  # For average and total activation
            ax.set_yscale("log")

        # Set x-axis ticks to integers
        ax.set_xticks(range(1, 5))

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.5)

    # Place legend outside and below the plot
    axes[0].legend(
        frameon=False, bbox_to_anchor=(2.3, -0.5), loc="upper center", ncol=1
    )

    # Set tight layout
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # [left, bottom, right, top]

    return fig


# Example usage:
# fig1 = create_nature_style_projection(propagation, neuron_position_data, step=1)
# plt.savefig('connectome_activation_projection.tiff', dpi=300)

# fig2 = create_activation_comparison_plot(
#     [original_df, random_df],
#     ["Original Connectome", "Randomized Connectome"]
# )
# plt.savefig('connectome_activation_comparison.tiff', dpi=300)


def create_activation_density_plot(
    propagation,
    neuron_position_data,
    activation_step=None,
    resolution=100,
    sigma=1.5,
    alpha_scale=5.0,
    cmap="viridis",
    background_color="black",
    figsize=(12, 10),
):
    """
    Create a 2D density plot of 3D neuronal activations.

    Parameters:
    -----------
    propagation : pandas.DataFrame
        DataFrame with columns root_id and activation_1 through activation_4
    neuron_position_data : pandas.DataFrame
        DataFrame with columns root_id, pos_x, pos_y, pos_z
    activation_step : int or None
        Which activation step to visualize (1-4), or None to visualize all steps
    resolution : int
        Resolution of the 3D grid used for density calculation
    sigma : float
        Sigma for Gaussian smoothing
    alpha_scale : float
        Scaling factor for activation values when computing alpha
    cmap : str
        Colormap name
    background_color : str
        Background color for the plot
    figsize : tuple
        Figure size (width, height) in inches

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Merge dataframes to get activated neurons with positions
    merged_data = pd.merge(propagation, neuron_position_data, on="root_id")

    # Get bounds of neuron positions
    x_min, x_max = merged_data["pos_x"].min(), merged_data["pos_x"].max()
    y_min, y_max = merged_data["pos_y"].min(), merged_data["pos_y"].max()
    z_min, z_max = merged_data["pos_z"].min(), merged_data["pos_z"].max()

    # Add some padding
    x_padding = 0.05 * (x_max - x_min)
    y_padding = 0.05 * (y_max - y_min)
    z_padding = 0.05 * (z_max - z_min)

    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    z_min -= z_padding
    z_max += z_padding

    # Create a 3D grid
    grid_x = np.linspace(x_min, x_max, resolution)
    grid_y = np.linspace(y_min, y_max, resolution)
    grid_z = np.linspace(z_min, z_max, resolution)

    # Initialize 3D volume for density
    volume = np.zeros((resolution, resolution, resolution, 3))  # RGB channels

    # Set up colormaps for different activation steps
    if activation_step is None:
        # All steps with different colors
        cmaps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds]
    else:
        # Single step
        cmaps = [plt.get_cmap(cmap)]

    # Process each activation step
    steps_to_process = [activation_step] if activation_step is not None else range(1, 5)

    for step_idx, step in enumerate(steps_to_process):
        if activation_step is not None:
            act_col = f"activation_{activation_step}"
            step_data = merged_data[merged_data[act_col] > 0]
        else:
            act_col = f"activation_{step}"
            step_data = merged_data[merged_data[act_col] > 0]

        # Skip if no activations for this step
        if len(step_data) == 0:
            continue

        # Create 3D histogram for this activation step
        H, _ = np.histogramdd(
            step_data[["pos_x", "pos_y", "pos_z"]].values,
            bins=(resolution, resolution, resolution),
            range=((x_min, x_max), (y_min, y_max), (z_min, z_max)),
            weights=step_data[act_col].values,
        )

        # Smooth the volume with a Gaussian filter
        H_smooth = gaussian_filter(H, sigma=sigma)

        # Normalize
        if H_smooth.max() > 0:
            H_smooth = H_smooth / H_smooth.max()

        # Convert to RGB using colormap
        cmap = cmaps[step_idx % len(cmaps)]
        for i in range(3):  # RGB channels
            rgb_slice = cmap(H_smooth)[..., i]
            volume[..., i] = np.maximum(volume[..., i], rgb_slice)

    # Create maximum intensity projections for different views
    mip_xy = np.max(volume, axis=2)  # Top view (xy plane)
    mip_xz = np.max(volume, axis=1)  # Front view (xz plane)
    mip_yz = np.max(volume, axis=0)  # Side view (yz plane)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor=background_color)

    # Main projection (top view)
    ax_main = fig.add_subplot(111)
    ax_main.imshow(
        mip_xy,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="bilinear",
    )
    ax_main.set_xlabel("X Position", color="white")
    ax_main.set_ylabel("Y Position", color="white")

    # Add small inset axes for the other views
    ax_xz = fig.add_axes([0.65, 0.15, 0.2, 0.2], facecolor=background_color)
    ax_xz.imshow(
        mip_xz,
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        interpolation="bilinear",
    )
    ax_xz.set_xlabel("X", color="white", fontsize=8)
    ax_xz.set_ylabel("Z", color="white", fontsize=8)
    ax_xz.tick_params(colors="white", labelsize=6)

    ax_yz = fig.add_axes([0.15, 0.65, 0.2, 0.2], facecolor=background_color)
    ax_yz.imshow(
        mip_yz,
        origin="lower",
        extent=[y_min, y_max, z_min, z_max],
        interpolation="bilinear",
    )
    ax_yz.set_xlabel("Y", color="white", fontsize=8)
    ax_yz.set_ylabel("Z", color="white", fontsize=8)
    ax_yz.tick_params(colors="white", labelsize=6)

    # Style the main plot
    for ax in [ax_main, ax_xz, ax_yz]:
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.tick_params(colors="white")

    # Title
    if activation_step is not None:
        title = f"Neuronal Activation Density - Step {activation_step}"
    else:
        title = "Neuronal Activation Density - All Steps"
    ax_main.set_title(title, color="white")

    # Add legend for activation steps if showing all
    if activation_step is None:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=cmaps[0](0.7), label="Step 1"),
            Patch(facecolor=cmaps[1](0.7), label="Step 2"),
            Patch(facecolor=cmaps[2](0.7), label="Step 3"),
            Patch(facecolor=cmaps[3](0.7), label="Step 4"),
        ]
        ax_main.legend(
            handles=legend_elements,
            loc="upper right",
            frameon=True,
            framealpha=0.8,
            facecolor="black",
            edgecolor="white",
            labelcolor="white",
        )

    plt.tight_layout()
    return fig

# Example usage:
# fig = create_activation_density_plot(propagation, neuron_position_data, activation_step=None)
# plt.savefig('neuronal_activation_density.tiff', dpi=300)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import pandas as pd
from matplotlib.colors import to_rgba
import matplotlib.gridspec as gridspec


def get_active_neuron_bounds(
    propagations_dict, neuron_position_data, padding_percent=10
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
        for step in range(2, 5):
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

########## THIS IS THE ONE ##########
def visualize_steps_separated_compact(
    propagations_dict,
    neuron_position_data,
    max_neurons_per_step=1000,
    voxel_size=20,
    smoothing=1.0,
    figsize=(16, 12),
    padding_percent=10,
):
    """
    Create a compact 4x4 grid of 3D visualizations with configurations as rows and steps as columns.

    Parameters:
    -----------
    propagations_dict : dict
        Dictionary with configuration names (keys) and propagation dataframes (values)
    neuron_position_data : pandas.DataFrame
        DataFrame with columns 'root_id', 'pos_x', 'pos_y', 'pos_z'
    max_neurons_per_step : int
        Maximum number of neurons to plot per step (for performance)
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
        Figure with the 4x4 grid of visualizations
    """
    # Get bounds of active neurons
    bounds = get_active_neuron_bounds(
        propagations_dict, neuron_position_data, padding_percent
    )
    x_min, x_max = bounds["x_min"], bounds["x_max"]
    y_min, y_max = bounds["y_min"], bounds["y_max"]
    z_min, z_max = bounds["z_min"], bounds["z_max"]

    # Nature-friendly colors for each step
    step_colors = ["#4878D0", "#6ACC64", "#EE854A", "#D65F5F"]

    # Create a figure with 4x4 grid: rows=configurations, columns=steps
    # Replace figure and GridSpec creation with:
    fig, axes = plt.subplots(
        len(propagations_dict), 4, figsize=figsize, subplot_kw={"projection": "3d"}
    )

    # Process each configuration (rows)
    for i, (config_name, prop_df) in enumerate(propagations_dict.items()):
        # Merge with position data
        merged_data = pd.merge(prop_df, neuron_position_data, on="root_id")

        # Process each activation step (columns)
        for step in range(1, 5):
            # Get subplot position
            if len(propagations_dict) > 1:
                ax = axes[i, step - 1]
            else:
                ax = axes[step - 1]

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

            # Add minimal axis labels only for the leftmost column and bottom row
            if step == 1:  # First column
                ax.set_ylabel("Y", labelpad=-10)
            if i == len(propagations_dict) - 1:  # Last row
                ax.set_xlabel("X", labelpad=-10)

            # Add Z label only for a subset
            if step == 1 and i == 0:
                ax.set_zlabel("Z", labelpad=-10)

            # Get color for this step
            color = step_colors[step - 1]

            # Filter data for this step
            act_col = f"activation_{step}"
            step_data = merged_data[merged_data[act_col] > 0].copy()

            # Add title to the top row only
            if i == 0:
                ax.set_title(f"Step {step}", fontsize=12, fontweight="bold", pad=5)

            # Add configuration label to the leftmost column only
            if step == 1:
                # Add y-axis label as configuration name
                ax.text2D(
                    -0.15,
                    0.5,
                    config_name,
                    transform=ax.transAxes,
                    va="center",
                    ha="center",
                    rotation=90,
                    fontsize=12,
                    fontweight="bold",
                )

            # Skip if no data
            if len(step_data) == 0:
                continue

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
            if len(step_data) > max_neurons_per_step:
                neuron_sample = step_data.sample(
                    max_neurons_per_step, random_state=step + 42
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

    plt.subplots_adjust(wspace=-0.6, hspace=-0.05)

    return fig


def plot_activation_statistics(propagations_dict, neuron_position_data):
    """
    Plot statistics about activations across different configurations.
    """
    # Calculate metrics for each configuration
    configs = list(propagations_dict.keys())
    activation_percentages = {config: [] for config in configs}
    activation_distances = {config: [] for config in configs}

    # Get bounds of neuron positions
    x_min, x_max = (
        neuron_position_data["pos_x"].min(),
        neuron_position_data["pos_x"].max(),
    )
    y_min, y_max = (
        neuron_position_data["pos_y"].min(),
        neuron_position_data["pos_y"].max(),
    )
    z_min, z_max = (
        neuron_position_data["pos_z"].min(),
        neuron_position_data["pos_z"].max(),
    )

    for config, prop_df in propagations_dict.items():
        # Calculate percentage of neurons active at each step
        total_neurons = len(neuron_position_data)

        for step in range(1, 5):
            act_col = f"activation_{step}"
            active_neurons = prop_df[prop_df[act_col] > 0]["root_id"].nunique()
            activation_percentages[config].append(100 * active_neurons / total_neurons)

        # Calculate average distance of active neurons from eye
        eye_position = np.array(
            [x_min, (y_max + y_min) / 2, (z_max + z_min) / 2]
        )  # Approximate

        merged = pd.merge(prop_df, neuron_position_data, on="root_id")
        for step in range(1, 5):
            act_col = f"activation_{step}"
            active = merged[merged[act_col] > 0]

            if len(active) > 0:
                positions = active[["pos_x", "pos_y", "pos_z"]].values
                distances = np.sqrt(np.sum((positions - eye_position) ** 2, axis=1))
                activation_distances[config].append(np.mean(distances))
            else:
                activation_distances[config].append(0)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="black")

    # Plot 1: Percentage of active neurons by step
    for i, config in enumerate(configs):
        ax1.plot(
            range(1, 5),
            activation_percentages[config],
            "o-",
            label=config,
            linewidth=2,
            markersize=8,
        )

    ax1.set_xlabel("Activation Step", color="white")
    ax1.set_ylabel("% of Neurons Active", color="white")
    ax1.set_title("Neuronal Activation by Configuration", color="white")
    ax1.grid(True, alpha=0.3)
    ax1.legend(facecolor="black", edgecolor="white", labelcolor="white")

    # Plot 2: Average distance of activated neurons
    for i, config in enumerate(configs):
        ax2.plot(
            range(1, 5),
            activation_distances[config],
            "o-",
            label=config,
            linewidth=2,
            markersize=8,
        )

    ax2.set_xlabel("Activation Step", color="white")
    ax2.set_ylabel("Avg. Distance from Input", color="white")
    ax2.set_title("Activation Propagation Distance", color="white")
    ax2.grid(True, alpha=0.3)

    # Style adjustments
    for ax in [ax1, ax2]:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

    plt.tight_layout()
    return fig

# Example usage:
# propagations_dict = {
#     "Original": original_df,
#     "Randomized": random_df,
#     "Shuffled": shuffled_df,
# }
# fig = plot_activation_statistics(propagations_dict, neuron_position_data)
# plt.savefig('activation_statistics.tiff', dpi=300)
# plt.show()
