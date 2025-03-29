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


def propagate_through_connectome(activated_data, connections, num_passes, weights=None):
    """
    Propagate activations through a connectome for a specified number of passes.

    Parameters:
    -----------
    activated_data : pd.DataFrame
        DataFrame containing initial activations with columns 'root_id' and 'activation'

    connections : pd.DataFrame
        DataFrame containing synaptic connections with columns 'pre_root_id', 'post_root_id',
        and optionally 'syn_count'

    num_passes : int
        Number of propagation passes to simulate

    weights : float, pd.DataFrame, or None
        Weights to use for connections. If None, all weights are set to 1.
        If float, all connections will have that weight.
        If DataFrame, should have the same structure as connections with a 'weight' column.

    Returns:
    --------
    propagation : pd.DataFrame
        DataFrame containing neuron IDs and their activations at each step
        Columns: 'root_id', 'input', 'activation_1', 'activation_2', ..., 'activation_n'
    """
    import pandas as pd
    import numpy as np

    # Make a copy of connections to avoid modifying the original
    connections_copy = connections.copy()

    # Get all unique neuron IDs from both sides of the connections
    all_neurons = pd.concat(
        [
            connections_copy["pre_root_id"].rename("root_id"),
            connections_copy["post_root_id"].rename("root_id"),
        ]
    ).drop_duplicates()

    # Create propagation dataframe with all neurons
    propagation = pd.DataFrame(all_neurons)
    propagation["input"] = 0

    # Set initial activations for visual neurons
    initial_activations = activated_data[["root_id", "activation"]].fillna(0)
    propagation = propagation.merge(
        initial_activations, on="root_id", how="left"
    ).fillna(0)

    # Replace the 'input' column with 'activation' values where available
    propagation["input"] = propagation["activation"].fillna(propagation["input"])
    propagation = propagation.drop(columns=["activation"])

    # Initialize activation with the same data
    activation = propagation.rename(columns={"input": "activation"})

    # Set weights for connections
    if weights is None:
        connections_copy["weight"] = 1
    elif isinstance(weights, (int, float)):
        connections_copy["weight"] = float(weights)
    elif isinstance(weights, pd.DataFrame) and "weight" in weights.columns:
        connections_copy = connections_copy.merge(
            weights[["pre_root_id", "post_root_id", "weight"]],
            on=["pre_root_id", "post_root_id"],
            how="left",
        )

    # Propagation loop
    for i in range(num_passes):
        activation = propagate_data_with_steps(activation.copy(), connections_copy, i)
        # If the propagate_data_with_steps function outputs a column named 'activation'
        if "activation" in activation.columns:
            activation = activation.rename(columns={"activation": f"activation_{i+1}"})
        propagation = propagation.merge(
            activation[
                ["root_id"]
                + [col for col in activation.columns if col.startswith("activation_")]
            ],
            on="root_id",
            how="left",
        ).fillna(0)

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
    propagation = propagate_through_connectome(activated_data, connections, num_passes)

    # Shuffled connectome
    shuffled_connections = reshuffle_connections(connections)
    propagation2 = propagate_through_connectome(
        activated_data, shuffled_connections, num_passes
    )

    # Random pruned connectome
    random_equalized_connections = load_connections("connections_random_pruned.csv")
    propagation3 = propagate_through_connectome(
        activated_data, random_equalized_connections, num_passes
    )

    # Random bin-wise connectome
    shuffled_connections_3 = load_connections("connections_random_binned.csv")
    propagation4 = propagate_through_connectome(
        activated_data, shuffled_connections_3, num_passes
    )

    # Return all propagations for further use
    return {
        "Original": propagation,
        "Random unconstrained": propagation2,
        "Random pruned": propagation3,
        "Random bin-wise": propagation4,
    }
