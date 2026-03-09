"""Functions for loading connections and computing activations across randomization strategies."""

import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from paths import PROJECT_ROOT
from connectome.visualization.plot_config import RANDOMIZATIONS
from utils.model_inspection_utils import propagate_data_with_steps


DATA_DIR = os.path.join(PROJECT_ROOT, "new_data")

CONNECTION_FILES = {
    "biological": "connections.csv",
    "random_binned": "connections_random_binned.csv",
    "unconstrained": "connections_random_unconstrained.csv",
    "random_pruned": "connections_random_pruned.csv",
    "connection_pruned": "connections_random_conn_pruned.csv",
    "neuron_binned": "connections_random_mantain_neuron_wiring_length.csv",
}


def _load_single_connections(file_name: str) -> pd.DataFrame:
    """Load a single connections CSV and prepare it for propagation."""
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(
        path,
        dtype={
            "pre_root_id": "string",
            "post_root_id": "string",
            "syn_count": "int32",
        },
    )
    
    df = (
        df.groupby(["pre_root_id", "post_root_id"], as_index=False)
        .sum("syn_count")
        .sort_values(["pre_root_id", "post_root_id"])
    )
    
    df["weight"] = 1.0
    return df


def get_all_connections() -> dict[str, pd.DataFrame]:
    """Load all connection variants and return as a dictionary.
    
    Returns
    -------
    dict
        Dictionary with display names as keys and connection DataFrames as values.
    """
    connections = {}
    
    for key in RANDOMIZATIONS.order:
        file_name = CONNECTION_FILES[key]
        df = _load_single_connections(file_name)
        if df is not None:
            display_name = RANDOMIZATIONS.label_for(key)
            connections[display_name] = df
    
    return connections


def _get_sample_visual_neurons() -> pd.DataFrame:
    """Load visual neuron data for creating input activations."""
    path = os.path.join(DATA_DIR, "right_visual_positions_all_neurons.csv")
    df = pd.read_csv(path, dtype={"root_id": "string"})
    return df


def _create_sample_activation(visual_neurons: pd.DataFrame) -> pd.DataFrame:
    """Create a sample activation pattern from R7 neurons.
    
    Simulates a uniform white input across all R7 cells.
    """
    r7_neurons = visual_neurons[visual_neurons["cell_type"] == "R7"].copy()
    
    centers = r7_neurons[["x_axis", "y_axis"]].values
    tree = cKDTree(centers)
    
    visual_neurons = visual_neurons.copy()
    visual_neurons["voronoi_indices"] = tree.query(
        visual_neurons[["x_axis", "y_axis"]].values
    )[1]
    
    activation = np.zeros(len(visual_neurons))
    retinal_mask = visual_neurons["cell_type"].isin(["R1-6", "R7", "R8"])
    activation[retinal_mask] = 1.0
    visual_neurons["activation"] = activation
    
    return visual_neurons[["root_id", "activation"]]


def _propagate_single_network(
    connections: pd.DataFrame,
    initial_activation: pd.DataFrame,
    all_root_ids: pd.DataFrame,
    num_passes: int,
) -> pd.DataFrame:
    """Propagate activation through a single network configuration.
    
    Returns a DataFrame with root_id, input, and activation_1..N columns only.
    Position columns are NOT included - merge with neuron_position_data separately.
    """
    propagation = (
        all_root_ids[["root_id"]].merge(initial_activation, on="root_id", how="left")
        .fillna(0)
        .rename(columns={"activation": "input"})
    )
    
    activation = initial_activation.copy()
    
    for i in range(num_passes):
        activation = propagate_data_with_steps(activation, connections, i)
        propagation = propagation.merge(activation, on="root_id", how="left").fillna(0)
    
    return propagation


def get_activation_dictionnary(
    connections_dict: dict[str, pd.DataFrame],
    num_passes: int = 4,
) -> dict[str, pd.DataFrame]:
    """Compute activation propagation for each network configuration.
    
    Parameters
    ----------
    connections_dict : dict
        Dictionary of connection DataFrames from get_all_connections()
    num_passes : int
        Number of message passing steps
        
    Returns
    -------
    dict
        Dictionary with the same keys as connections_dict, containing
        activation DataFrames with columns: root_id, input, activation_1, ..., activation_N
    """
    from connectome.data_helpers import load_neuron_coordinates
    
    coords = load_neuron_coordinates()
    visual_neurons = _get_sample_visual_neurons()
    initial_activation = _create_sample_activation(visual_neurons)
    
    activations = {}
    
    for name, connections in connections_dict.items():
        propagation = _propagate_single_network(
            connections, initial_activation, coords, num_passes
        )
        activations[name] = propagation
    
    return activations

