from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd
from scipy.spatial import voronoi_plot_2d

cmap = plt.cm.binary


def propagate_data_once(input_data, connections):
    df = input_data.merge(
        connections, left_on="root_id", right_on="pre_root_id", how="left"
    )
    df["activation"] = df["activation"] * df["syn_count"] * df["weight"]
    df = df.drop(columns=["root_id"]).rename(columns={"post_root_id": "root_id"})

    return df[["root_id", "voronoi_indices", "activation"]]


def propagate_data_with_steps(input_data, connections, step):
    input_cols = input_data.columns
    df = input_data.merge(
        connections, left_on="root_id", right_on="pre_root_id", how="left"
    )
    new_name = f"activation_{step + 1}"
    df[new_name] = df[input_cols[-1]] * (df["syn_count"] * df["weight"] - 1)
    df = df.drop(columns=["root_id"]).rename(columns={"post_root_id": "root_id"})
    return df[["root_id", new_name]].groupby("root_id").sum().reset_index()


def compute_voronoi_cells(neuron_data, voronoi):
    post_synapse_cells = (
        neuron_data[["voronoi_indices", "activation"]]
        .groupby("voronoi_indices")
        .sum("activation")
    )
    post_synapse_cells.index = [
        voronoi.point_region[int(i)] for i in post_synapse_cells.index
    ]
    return post_synapse_cells


def process_image(img, tree):
    pixel_num = img.shape[0]
    # create image template and tesselate with the voronoi cells created with the neuron data
    img_coords = (
        np.array(np.meshgrid(np.arange(pixel_num), np.arange(pixel_num), indexing="xy"))
        .reshape(2, -1)
        .T
    )
    img_coords[:, 1] = pixel_num - img_coords[:, 1] - 1
    image_indices = tree.query(img_coords)[1]
    # convert imge to 0-1
    img = img / 255
    # add each pixel to the corresponding voronoi cell
    img = img.reshape(-1, 3)
    # compute the mean of the three channels
    img = np.concatenate([img, np.mean(img, axis=1).reshape(-1, 1)], axis=1)
    # add the voronoi cell of each pixel
    img = np.concatenate([img, image_indices.reshape(-1, 1)], axis=1)
    df = pd.DataFrame(img, columns=["r", "g", "b", "mean", "cell"])
    df = df.groupby("cell").mean()

    return df


def plot_input_image(img, voronoi, ax=None):
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Display the image
    ax.imshow(img, extent=[0, 512, 0, 512], origin="upper")
    voronoi_plot_2d(
        voronoi,
        ax=ax,
        show_vertices=False,
        line_colors="orange",
        line_width=2,
        line_alpha=0.6,
        point_size=2,
    )

    return fig


def get_colour_average(values, colour):
    return (values[colour] + values["mean"]) / 2


def plot_first_activations(neuron_data, voronoi, ax=None):
    voronoi_plot_2d(
        voronoi,
        ax=ax,
        show_vertices=False,
        line_colors="orange",
        line_width=2,
        line_alpha=0.6,
        point_size=2,
    )

    neuron_data["voronoi_indices"] = [voronoi.point_region[i] for i in neuron_data["voronoi_indices"]]

    rgb_values = (
        neuron_data.loc[:, ["voronoi_indices", "cell_type", "activation"]]
        .groupby(["voronoi_indices", "cell_type"])
        .mean()
        .pivot_table(index="voronoi_indices", columns="cell_type", values="activation")
        .fillna(0)
        .rename(columns={"R1-6": "mean", "R7": "b", "R8p": "g", "R8y": "r"})
    )

    rgb_values["b"] = get_colour_average(rgb_values, "b")
    rgb_values["g"] = get_colour_average(rgb_values, "g")
    rgb_values["r"] = get_colour_average(rgb_values, "r")

    # Fill Voronoi regions with colors based on aggregated RGB values
    for region_index in voronoi.point_region:
        region = voronoi.regions[region_index]
        if not -1 in region:
            polygon = [voronoi.vertices[i] for i in region]
            color = rgb_values.loc[region_index, ["r", "g", "b"]]
            ax.fill(*zip(*polygon), color=color)

    return ax


def plot_voronoi_activations(df, voronoi, ax=None, cmap=plt.cm.binary):

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Assuming 'post_synapse_cells' is a DataFrame and contains the activation values
    min_activation = df["activation"].min()
    max_activation = df["activation"].max()

    norm = mcolors.Normalize(vmin=min_activation, vmax=max_activation)

    voronoi_plot_2d(
        voronoi,
        ax=ax,
        show_vertices=False,
        line_colors="orange",
        line_width=2,
        line_alpha=0.6,
        point_size=2,
    )
    for region_index in voronoi.point_region:
        region = voronoi.regions[region_index]
        if not -1 in region:
            polygon = [voronoi.vertices[i] for i in region]
            activation = df.loc[region_index, "activation"]
            color = cmap(norm(activation))
            ax.fill(*zip(*polygon), color=color)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This line may be necessary depending on Matplotlib version
    cbar = plt.colorbar(sm, ax=ax)

    return fig


def process_and_plot_data(img, neuron_data, connections, voronoi, num_passes):
    current_data = neuron_data.copy()

    num_columns = 3
    num_rows = (num_passes + 2 + num_columns - 1) // num_columns

    # Create a figure with subplots
    fig, axes = plt.subplots(
        num_rows, num_columns, figsize=(num_columns * 5, num_rows * 5)
    )
    axes = axes.flatten()

    # Initial plot for the input image
    plot_input_image(img, voronoi, axes[0])
    axes[0].set_title("Input Image")

    plot_first_activations(neuron_data.copy(), voronoi, axes[1])
    axes[1].set_title("First activations")

    # Process data and plot activations
    for i in range(num_passes):
        current_data = propagate_data_once(current_data, connections)
        post_synapse_cells = compute_voronoi_cells(current_data, voronoi)
        plot_voronoi_activations(post_synapse_cells, voronoi, ax=axes[i + 2])
        axes[i + 2].set_title(f"Pass {i + 1}")

    # Hide any unused axes if there are any
    for j in range(num_passes + 2, num_rows * num_columns):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def normalize_non_zero(x, min_val=None, max_val=None):
    non_zero_mask = x != 0
    non_zero_entries = x[non_zero_mask]
    if min_val is not None:
        x[non_zero_mask] = (non_zero_entries - min_val) / (max_val - min_val)
    else:
        x[non_zero_mask] = (non_zero_entries - non_zero_entries.min()) / (
            non_zero_entries.max() - non_zero_entries.min()
        )
    return x


def get_neuron_type(x, activation_cols):
    # The logic here is:
    # - If decision_making is active, return this, since it's the most interesting
    # - If not, return the first active activation
    for col in [activation_cols[-1]] + activation_cols[:-1]:
        if x[col] != 0:
            return col
    return "no_activation"
