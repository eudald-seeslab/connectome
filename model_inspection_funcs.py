import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
cmap = plt.get_cmap("viridis")
from imageio.v3 import imread
from scipy.spatial import cKDTree

from train_funcs import get_activation_from_cell_type, assign_cell_type
from model_inspection_utils import (
    process_image,
    propagate_data_with_steps,
)


data_cols = ["x_axis", "y_axis"]


def activation_cols_and_colours(num_passes):
    activation_cols = (
        ["input"]
        + [f"activation_{i}" for i in range(1, num_passes)]
        + ["decision_making"]
    )

    no_activation_colour = "#ffffff"
    colours = {
        a: to_hex(cmap(i))
        for a, i in zip(activation_cols, np.linspace(0, 1, num_passes + 1))
    }
    colours.update({"no_activation": no_activation_colour})
    return activation_cols, colours


def sample_images(base_dir, sub_dirs, sample_size=10):
    sampled_images = []

    # Loop through the subdirectories
    for sub_dir in sub_dirs:
        path = os.path.join(base_dir, sub_dir)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        sampled_files = random.sample(files, min(sample_size, len(files)))
        sampled_images.extend([os.path.join(path, file) for file in sampled_files])

    return sampled_images


def neuron_data_from_image(img_path, neuron_data):
    img = imread(img_path)
    # tesselation
    centers = neuron_data[neuron_data["cell_type"] == "R7"][data_cols].values
    tree = cKDTree(centers)
    neuron_indices = tree.query(neuron_data[data_cols].values)[1]
    neuron_data["voronoi_indices"] = neuron_indices

    processed_image = process_image(img, tree)
    neuron_data = neuron_data.merge(
        processed_image, left_on="voronoi_indices", right_index=True
    )
    neuron_data["cell_type"] = neuron_data.apply(assign_cell_type, axis=1)
    neuron_data["activation"] = neuron_data.apply(get_activation_from_cell_type, axis=1)
    return neuron_data


def propagate_neuron_data(neuron_data, connections, coords, neurons, num_passes):
    propagation = (
        coords.merge(
            neuron_data[["root_id", "activation"]], on="root_id", how="left"
        )
        .fillna(0)
        .rename(columns={"activation": "input"})
    )
    activation = neuron_data[["root_id", "activation"]]

    for i in range(num_passes):
        activation = propagate_data_with_steps(activation.copy(), connections, i)
        propagation = propagation.merge(activation, on="root_id", how="left").fillna(0)

    cols = propagation.columns.tolist()
    propagation = propagation.merge(
        neurons[["root_id", "decision_making"]], on="root_id"
    )
    propagation["decision_making"] = (
        propagation["decision_making"] * propagation[cols[-1]]
    )
    return propagation.drop(columns=[cols[-1]])


def propagate_data_without_deciding(neuron_data, connections, coords, num_passes):
    propagation = (
        coords.merge(neuron_data[["root_id", "activation"]], on="root_id", how="left")
        .fillna(0)
        .rename(columns={"activation": "input"})
    )
    activation = neuron_data[["root_id", "activation"]]

    for i in range(num_passes):
        activation = propagate_data_with_steps(activation.copy(), connections, i)
        propagation = propagation.merge(activation, on="root_id", how="left").fillna(0)

    # return last column of propagation
    return propagation.iloc[:, -1]
