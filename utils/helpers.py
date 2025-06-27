import numpy as np
import os
import pandas as pd
import logging
from paths import PROJECT_ROOT


def add_coords(connections_df, coords_df):
    """Add pre and post neuron coordinates to connections dataframe"""

    # If coords' columns are based on soma, rename them
    if "soma_x" in coords_df.columns:
        coords_df = coords_df.rename(
            columns={
                "soma_x": "pos_x",
                "soma_y": "pos_y",
                "soma_z": "pos_z",
            }
        )

    # Make sure all root ids are strings
    coords_df["root_id"] = coords_df["root_id"].astype(str)
    connections_df["pre_root_id"] = connections_df["pre_root_id"].astype(str)
    connections_df["post_root_id"] = connections_df["post_root_id"].astype(str)
    
    # Add pre-neuron coordinates
    df = connections_df.merge(
        coords_df[["root_id", "pos_x", "pos_y", "pos_z"]],
        left_on="pre_root_id",
        right_on="root_id",
        how="left",
        suffixes=("", "_pre"),
    )
    
    # Remove unnecessary column
    if "root_id" in df.columns:
        df = df.drop("root_id", axis=1)
    
    # Add post-neuron coordinates
    df = df.merge(
        coords_df[["root_id", "pos_x", "pos_y", "pos_z"]],
        left_on="post_root_id",
        right_on="root_id",
        how="left",
        suffixes=("_pre", "_post"),
    )
    
    # Remove unnecessary column
    if "root_id" in df.columns:
        df = df.drop("root_id", axis=1)

    # Rename columns for clarity
    df = df.rename(
        columns={
            "pos_x_pre": "pre_x",
            "pos_y_pre": "pre_y",
            "pos_z_pre": "pre_z",
            "pos_x_post": "post_x",
            "pos_y_post": "post_y",
            "pos_z_post": "post_z",
        }
    )
    
    return df

def compute_individual_synapse_lengths(connections, neuron_coords):
    """
    Compute the length of each synapse.
    """
    conns_with_coords = add_coords(connections, neuron_coords)
    return np.linalg.norm(
        conns_with_coords[["pre_x", "pre_y", "pre_z"]].values -
        conns_with_coords[["post_x", "post_y", "post_z"]].values,
        axis=1
    )


def compute_total_synapse_length(connections, neuron_coords):
    """
    Compute the total wiring length of all synapses.
    
    Parameters:
    -----------
    connections : DataFrame
        Contains pre_root_id, post_root_id, and syn_count columns
    neuron_coords : DataFrame
        Contains root_id, x, y, z coordinates for each neuron
        
    Returns:
    --------
    float: The total wiring length
    """
    
    # Calculate total length by multiplying each connection distance by its synapse count
    return np.sum(compute_individual_synapse_lengths(connections, neuron_coords) * connections["syn_count"])


def add_distance_column(connections_df, coords_df, distance_col: str = "distance"):
    """Return a copy of `connections_df` with coordinates and a new column
    containing the Euclidean distance between the pre- and post-synaptic
    neurons.

    This wraps `add_coords` and centralises the distance computation that was
    duplicated across several modules.
    """

    df = add_coords(connections_df, coords_df)
    df[distance_col] = np.linalg.norm(
        df[["pre_x", "pre_y", "pre_z"]].values -
        df[["post_x", "post_y", "post_z"]].values,
        axis=1,
    )
    return df


def setup_logging(level: int = logging.INFO) -> None:
    """Configure global logging in a consistent way.

    If the root logger already has handlers attached, the function does
    nothing so it can be called safely from multiple modules.
    """
    if logging.getLogger().handlers:
        return  # logging already configured elsewhere

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_connections(file_name: str = "connections.csv", root_dir: str = PROJECT_ROOT) -> pd.DataFrame:
    """Load a connections CSV and aggregate duplicate rows.

    The function enforces the column dtypes used across the codebase and
    groups any repeated (pre, post) pairs by summing *syn_count*.
    """
    path = os.path.join(root_dir, "new_data", file_name)

    df = pd.read_csv(
        path,
        dtype={
            "pre_root_id": "string",
            "post_root_id": "string",
            "syn_count": "int32",
        },
    )

    # Aggregate duplicates just in case
    df = (
        df.groupby(["pre_root_id", "post_root_id"], as_index=False)
        .sum("syn_count")
        .sort_values(["pre_root_id", "post_root_id"])
    )
    return df


def load_neuron_annotations(file_name: str = "neuron_annotations.tsv", root_dir: str = PROJECT_ROOT) -> pd.DataFrame:
    """Load the master neuron annotation table with cleaned coordinates."""
    path = os.path.join(root_dir, "new_data", file_name)

    nc = pd.read_table(
        path,
        dtype={
            "root_id": "string",
            "soma_x": "float32",
            "soma_y": "float32",
            "soma_z": "float32",
            "cell_type": "string",
        },
        usecols=[
            "root_id",
            "pos_x",
            "pos_y",
            "pos_z",
            "soma_x",
            "soma_y",
            "soma_z",
            "cell_type",
        ],
    )

    # Prefer soma coordinates; fall back to centroid (pos_*) when missing
    nc["soma_x"] = nc["soma_x"].fillna(nc["pos_x"])
    nc["soma_y"] = nc["soma_y"].fillna(nc["pos_y"])
    nc["soma_z"] = nc["soma_z"].fillna(nc["pos_z"])

    nc = (
        nc.drop(columns=["pos_x", "pos_y", "pos_z"])
        .rename(columns={"soma_x": "pos_x", "soma_y": "pos_y", "soma_z": "pos_z"})
    )
    return nc


def shuffle_post_root_id(connections: pd.DataFrame, random_state: int | None = None) -> pd.DataFrame:
    """Return a copy with *post_root_id* randomly permuted (degree-preserving)."""
    rng = np.random.default_rng(random_state)
    shuffled = connections.copy()
    shuffled["post_root_id"] = rng.permutation(shuffled["post_root_id"].values)
    return shuffled


def shuffle_within_bin(bin_group):
    if len(bin_group) <= 1:
        return bin_group
    shuffled_post_ids = np.random.permutation(bin_group['post_root_id'].values)
    result = bin_group.copy()
    result['post_root_id'] = shuffled_post_ids
    return result
