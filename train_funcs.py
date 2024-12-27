from concurrent.futures import ThreadPoolExecutor
from imageio.v3 import imread
from PIL import Image
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import torch


def import_images(img_paths):
    # Read images using a thread pool to speed up disk I/O operations
    with ThreadPoolExecutor() as executor:
        imgs = list(executor.map(imread, img_paths))

    # Stack images into a single NumPy array
    imgs = np.stack(imgs, axis=0)
    return imgs


def preprocess_images(imgs):
    # The system is designed to work with 512x512 colour images. The obvious solution
    # would be to resize the tesselation of the neurons, who doesn't care about scale, 
    # but beware, traveler, because very small images will make the Voronoi tessellation 
    # fail (because there might not be enough pixels to create the cells). So, we are 
    # doomed to resize the image...
    # The colour part is easy, just repeat the image 3 times to create the 3 channels.
    
    if imgs[0].shape[0] != 512:
        imgs = np.array([np.array(Image.fromarray(a).resize((512, 512))) for a in imgs])
    
    if imgs.ndim == 3:
        imgs = np.stack([imgs]*3, axis=-1)
        
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
        img = pd.DataFrame(img, columns=["r", "g", "b", "mean", "cell"])
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


def apply_inhibitory_r7_r8(neuron_activations):
    mask = neuron_activations["b"] > neuron_activations[["r", "g"]].max(axis=1)
    neuron_activations.loc[mask, ["r", "g"]] = 0
    neuron_activations.loc[~mask, "b"] = 0
    return neuron_activations


def get_neuron_activations(right_visual, voronoi_average, inhibitory_r7_r8=False):
    neuron_activations = right_visual.merge(
        voronoi_average, left_on="voronoi_indices", right_index=True
    )
    if inhibitory_r7_r8:
        neuron_activations = apply_inhibitory_r7_r8(neuron_activations)

    neuron_activations["activation"] = neuron_activations.apply(
        get_activation_from_cell_type, axis=1
    )
    return neuron_activations.set_index("root_id")[["activation"]]


def get_side_decision_making_vector(root_ids, rational_cell_types, neurons, side=None):

    if side is not None:
        neurons = neurons[neurons["side"] == side]

    rational_neurons = neurons[neurons["cell_type"].isin(rational_cell_types)]
    temp = root_ids.merge(rational_neurons, on="root_id", how="left")
    return torch.tensor(
        temp.assign(rational=np.where(temp["side"].isna(), 0, 1))["rational"].values
    )


def construct_synaptic_matrix(neuron_classification, connections, root_ids):

    all_neurons = neuron_classification.merge(root_ids, on="root_id").fillna("Unknown")

    ix_conns = connections.merge(
        all_neurons[["root_id", "index_id"]], left_on="pre_root_id", right_on="root_id"
    ).merge(
        all_neurons[["root_id", "index_id"]],
        left_on="post_root_id",
        right_on="root_id",
        suffixes=("_pre", "_post"),
    )

    # Sum duplicate connections
    grouped = (
        ix_conns.groupby(["index_id_pre", "index_id_post"])["syn_count"]
        .sum()
        .reset_index()
    )

    matrix = coo_matrix(
        (grouped["syn_count"], (grouped["index_id_pre"], grouped["index_id_post"])),
        shape=(len(root_ids), len(root_ids)),
        dtype=np.int32,
    )

    # Force the matrix to sum duplicates and sort indices
    matrix.sum_duplicates()

    return matrix
