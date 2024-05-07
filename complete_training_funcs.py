from concurrent.futures import ThreadPoolExecutor
import os
from imageio.v3 import imread
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

import torch


num_workers = os.cpu_count() - 2


def get_voronoi_cells(neuron_data, pixel_num=512, ommatidia_size=8):

    # sample n_centers random points with ommatidia_size neurons, in average, each
    n_centers = neuron_data.shape[0] // ommatidia_size
    rand_points = neuron_data[["y", "z"]].sample(n_centers).values
    # create the voronoi tree
    tree = cKDTree(rand_points)
    _, neuron_indices = tree.query(neuron_data[["y", "z"]].values)

    # create a grid of 512x512 where each point is just its own coordinates
    img_coords = (
        np.array(np.meshgrid(np.arange(pixel_num), np.arange(pixel_num), indexing="ij"))
        .reshape(2, -1)
        .T
    )
    # apply the tree to the images grid
    _, img_indices = tree.query(img_coords)
    return neuron_indices, img_indices


def import_images(img_paths):
    # Read images using a thread pool to speed up disk I/O operations
    with ThreadPoolExecutor() as executor:
        imgs = list(executor.map(imread, img_paths))

    # Stack images into a single NumPy array
    imgs = np.stack(imgs, axis=0)
    return imgs


def process_images(imgs, voronoi_indices):
    # Reshape images: [n_images, n_pixels*n_pixels, n_channels]
    imgs = imgs.reshape(imgs.shape[0], -1, imgs.shape[-1])

    # Convert to 0-1 scale
    imgs = imgs / 255.0

    # Calculate mean of channels and stack it
    mean_channel = np.mean(imgs, axis=2, keepdims=True)
    imgs = np.concatenate([imgs, mean_channel], axis=2)

    # Prepare Voronoi indices
    # Shape it to (1, n_pixels*n_pixels, 1)
    voronoi_indices = voronoi_indices.reshape(1, -1, 1)
    # Repeat for all images
    voronoi_indices = np.repeat(voronoi_indices, imgs.shape[0], axis=0)

    # Append Voronoi indices
    imgs = np.concatenate([imgs, voronoi_indices], axis=2)

    return imgs


def get_voronoi_averages(processed_imgs):
    dfs = []
    for img in processed_imgs:
        img = pd.DataFrame(img)
        img.columns = ["r", "g", "b", "mean", "cell"]
        dfs.append(img.groupby("cell").mean())
    return dfs


def assign_cell_type(row):
    # In right_visual, when cell_type is R8, we can have two types of cells
    # R8p (30%) and R8y (70%). Let's create a new column and randomly assign
    # the cell type to each cell.
    if row["cell_type"] == "R8":
        return "R8p" if np.random.rand(1) < 0.3 else "R8y"
    return row["cell_type"]


def get_activation_from_cell_type(row):
    match row["cell_type"]:
        case "R1-6":
            return row["mean"]
        case "R7":
            return row["b"]
        case "R8p":
            return row["g"]
        case "R8y":
            return row["r"]
        case _:
            raise ValueError("Invalid cell type")


def get_neuron_activations(right_visual, voronoi_average):
    neuron_activations = right_visual.merge(
        voronoi_average, left_on="voronoi_indices", right_index=True
    )
    neuron_activations["activation"] = neuron_activations.apply(
        get_activation_from_cell_type, axis=1
    )
    return neuron_activations.set_index("root_id")[["activation"]]


def get_side_decision_making_vector(right_root_ids, side):
    cell_type_rational = pd.read_csv("data/cell_type_rational_short.csv")
    # get the cell types with rational = 1
    rational_cell_types = cell_type_rational[cell_type_rational["rational"] == 1][
        "cell_type"
    ]
    right_neurons = pd.read_csv("adult_data/classification.csv")
    right_neurons = right_neurons[right_neurons["side"] == side]
    right_neurons = right_neurons[
        right_neurons["root_id"].isin(right_root_ids["root_id"])
    ]
    rational_neurons = right_neurons[
        right_neurons["cell_type"].isin(rational_cell_types)
    ]
    temp = right_root_ids.merge(rational_neurons, on="root_id", how="left")
    return torch.tensor(temp.assign(rational=np.where(temp["side"].isna(), 0, 1))["rational"].values)
