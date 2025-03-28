import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import connectome

project_root = connectome.setup_notebook(use_project_root_as_cwd=True)

sys.path.insert(0, str(project_root))

from utils.model_inspection_funcs import (
    neuron_data_from_image,
    sample_images,
)
from utils.model_inspection_utils import propagate_data_with_steps

cmap = plt.cm.binary

device = torch.device("cpu")
dtype = torch.float32


def load_visual_neuron_data():
    return pd.read_csv(
        os.path.join(
            project_root, "new_data", "right_visual_positions_selected_neurons.csv"
        ),
        dtype={"root_id": "string"},
    ).drop(columns=["x", "y", "z", "PC1", "PC2"])


def load_neuron_position_data():
    return pd.read_table(
        os.path.join(project_root, "new_data", "neuron_annotations.tsv"),
        dtype={"root_id": "string"},
        usecols=["root_id", "pos_x", "pos_y", "pos_z"],
    )


def load_connections(file_name):
    temp = pd.read_csv(
        os.path.join(project_root, "new_data", file_name),
        dtype={
            "pre_root_id": "string",
            "post_root_id": "string",
            "syn_count": np.int32,
        },
        index_col=0,
    )
    grouped = (
        temp.groupby(["pre_root_id", "post_root_id"]).sum("syn_count").reset_index()
    )
    return grouped.sort_values(["pre_root_id", "post_root_id"])


def propagate_activations(activated_data, connections, num_passes):
    propagation = (
        activated_data[["root_id", "activation"]]
        .fillna(0)
        .rename(columns={"activation": "input"})
    )
    activation = activated_data[["root_id", "activation"]]
    connections["weight"] = 1

    for i in range(num_passes):
        activation = propagate_data_with_steps(activation.copy(), connections, i)
        propagation = propagation.merge(activation, on="root_id", how="left").fillna(0)

    return propagation


def reshuffle_connections(connections):
    shuffled_connections = connections.copy()
    shuffled_connections["post_root_id"] = np.random.permutation(
        connections["post_root_id"]
    )
    return shuffled_connections


def get_activation_dictionnary():
    num_passes = 4
    base_dir = os.path.join(project_root, "images", "one_to_ten", "train")
    sub_dirs = ["yellow", "blue"]

    sampled_images = sample_images(base_dir, sub_dirs, 1)
    img = sampled_images[0]

    visual_neuron_data = load_visual_neuron_data()
    activated_data = neuron_data_from_image(img, visual_neuron_data)

    # Real connectome
    connections = load_connections("connections.csv")
    propagation = propagate_activations(activated_data, connections, num_passes)

    # Shuffled connectome
    shuffled_connections = reshuffle_connections(connections)
    propagation2 = propagate_activations(
        activated_data, shuffled_connections, num_passes
    )

    # Random pruned connectome
    random_equalized_connections = load_connections("connections_random.csv")
    propagation3 = propagate_activations(
        activated_data, random_equalized_connections, num_passes
    )

    # Random bin-wise connectome
    shuffled_connections_3 = load_connections("connections_random3.csv")
    propagation4 = propagate_activations(
        activated_data, shuffled_connections_3, num_passes
    )

    # Return all propagations for further use
    return {
        "Original": propagation,
        "Random unconstrained": propagation2,
        "Random pruned": propagation3,
        "Random bin-wise": propagation4,
    }
