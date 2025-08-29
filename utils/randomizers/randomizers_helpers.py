import numpy as np
import pandas as pd

from utils.helpers import add_coords


__all__ = [
    "add_coords",
    "add_distance_column",
    "compute_individual_synapse_lengths",
    "compute_total_synapse_length",
]


def add_coords(connections_df: pd.DataFrame, coords_df: pd.DataFrame) -> pd.DataFrame:
    """Attach pre- and post-synaptic 3-D coordinates to a connections table.

    The implementation mirrors the original version in ``utils.helpers`` but
    lives in a dedicated module so that randomisation code can depend on it
    without importing the large *helpers* module (which drags additional
    pandas/OS utilities).
    """

    # If coords' columns are based on soma, rename them to pos_* for uniformity
    if "soma_x" in coords_df.columns:
        coords_df = coords_df.rename(
            columns={"soma_x": "pos_x", "soma_y": "pos_y", "soma_z": "pos_z"}
        )

    # Ensure consistent dtypes to avoid pandas warnings / copies
    coords_df = coords_df.copy()
    coords_df["root_id"] = coords_df["root_id"].astype(str)

    conns = connections_df.copy()
    conns["pre_root_id"] = conns["pre_root_id"].astype(str)
    conns["post_root_id"] = conns["post_root_id"].astype(str)

    # Merge pre neuron coordinates
    df = conns.merge(
        coords_df[["root_id", "pos_x", "pos_y", "pos_z"]],
        left_on="pre_root_id",
        right_on="root_id",
        how="left",
        suffixes=("", "_pre"),
    )
    df = df.drop(columns=["root_id"], errors="ignore")

    # Merge post neuron coordinates
    df = df.merge(
        coords_df[["root_id", "pos_x", "pos_y", "pos_z"]],
        left_on="post_root_id",
        right_on="root_id",
        how="left",
        suffixes=("_pre", "_post"),
    )
    df = df.drop(columns=["root_id"], errors="ignore")

    # Rename for clarity
    return df.rename(
        columns={
            "pos_x_pre": "pre_x",
            "pos_y_pre": "pre_y",
            "pos_z_pre": "pre_z",
            "pos_x_post": "post_x",
            "pos_y_post": "post_y",
            "pos_z_post": "post_z",
        }
    )


def add_distance_column(
    connections_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    distance_col: str = "distance",
) -> pd.DataFrame:
    """Return ``connections_df`` plus a Euclidean *distance* column (vectorised)."""
    df = add_coords(connections_df, coords_df)
    df[distance_col] = np.linalg.norm(
        df[["pre_x", "pre_y", "pre_z"]].values
        - df[["post_x", "post_y", "post_z"]].values,
        axis=1,
    )
    return df


def compute_individual_synapse_lengths(
    connections: pd.DataFrame, neuron_coords: pd.DataFrame
) -> np.ndarray:
    """Vector of per-synapse lengths (one entry per row in *connections*)."""
    coords = add_coords(connections, neuron_coords)
    return np.linalg.norm(
        coords[["pre_x", "pre_y", "pre_z"]].values
        - coords[["post_x", "post_y", "post_z"]].values,
        axis=1,
    )


def compute_total_synapse_length(
    connections: pd.DataFrame, neuron_coords: pd.DataFrame
) -> float:
    """Total wiring length (∑ distance × syn_count)."""
    return float(
        np.sum(
            compute_individual_synapse_lengths(connections, neuron_coords)
            * connections["syn_count"].values
        )
    )


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
