import numpy as np


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
