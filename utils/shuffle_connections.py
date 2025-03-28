from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import os
from paths import PROJECT_ROOT


def load_data():
    connections = pd.read_csv(
        os.path.join(PROJECT_ROOT, "new_data", "connections.csv"),
        dtype={
            "pre_root_id": "string",
            "post_root_id": "string",
            "syn_count": np.int32,
        },
    )

    nc = pd.read_table(
        os.path.join(PROJECT_ROOT, "new_data", "neuron_annotations.tsv"),
        dtype={
            "root_id": "string",
            "soma_x": np.float32,
            "soma_y": np.float32,
            "soma_z": np.float32,
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
    nc["soma_x"] = nc["soma_x"].fillna(nc["pos_x"])
    nc["soma_y"] = nc["soma_y"].fillna(nc["pos_y"])
    nc["soma_z"] = nc["soma_z"].fillna(nc["pos_z"])
    nc = nc.drop(columns=["pos_x", "pos_y", "pos_z"])

    return connections, nc


def compute_total_synapse_length(connections, nc):
    # Pre-merge nc for efficiency
    nc_pre = nc.rename(
        columns={
            "root_id": "pre_root_id",
            "soma_x": "soma_x_pre",
            "soma_y": "soma_y_pre",
            "soma_z": "soma_z_pre",
        }
    )
    nc_post = nc.rename(
        columns={
            "root_id": "post_root_id",
            "soma_x": "soma_x_post",
            "soma_y": "soma_y_post",
            "soma_z": "soma_z_post",
        }
    )

    df = connections.merge(nc_pre, on="pre_root_id").merge(nc_post, on="post_root_id")

    # Use NumPy for distance calculations
    soma_pre = df[["soma_x_pre", "soma_y_pre", "soma_z_pre"]].to_numpy()
    soma_post = df[["soma_x_post", "soma_y_post", "soma_z_post"]].to_numpy()
    distances = np.linalg.norm(soma_pre - soma_post, axis=1)

    return np.sum(distances * df["syn_count"].values)


def shuffle_post_root_id(connections):
    new_conns = connections.copy()
    shuffled_posts = np.random.permutation(new_conns["post_root_id"].values)
    new_conns["post_root_id"] = shuffled_posts
    return new_conns


def shuffle_within_bin_parallel(bin_group):
    if len(bin_group) <= 1:
        return bin_group
    shuffled_indices = np.random.permutation(bin_group.index)
    return bin_group.loc[shuffled_indices]


def match_wiring_length_with_syn_scale(
    connections,
    nc,
    real_length,
    scale_low=0.0,
    scale_high=2.0,
    max_iter=5,
    tolerance=0.01,
):
    conns_scaled = connections.copy()

    for _ in range(max_iter):
        mid = 0.5 * (scale_low + scale_high)
        conns_scaled["syn_count"] = connections["syn_count"] * mid
        length_est = compute_total_synapse_length(conns_scaled, nc)

        ratio = length_est / real_length
        if length_est < real_length:
            scale_low = mid
        else:
            scale_high = mid

        if abs(ratio - 1.0) < tolerance:
            break

    final_scale = 0.5 * (scale_low + scale_high)
    conns_scaled["syn_count"] = connections["syn_count"] * final_scale
    return conns_scaled


def create_length_preserving_random_network(
    connections, neurons, bins=10, tolerance=0.1
):
    connections = connections.copy()
    neurons = neurons.copy()

    # Perform type conversions once
    connections = connections.astype(
        {"pre_root_id": int, "post_root_id": int, "syn_count": int}
    )
    neurons = neurons.astype({"root_id": int})

    # Precompute retinal and decision IDs
    retinal_ids = set(
        neurons[neurons["cell_type"].isin(["R1-6", "R7", "R8"])]["root_id"]
    )
    decision_ids = set(
        neurons[neurons["cell_type"].isin(["KCapbp-m", "KCapbp-ap2", "KCapbp-ap1"])][
            "root_id"
        ]
    )

    preserve_mask = connections["pre_root_id"].isin(retinal_ids) | connections[
        "post_root_id"
    ].isin(decision_ids)
    preserved_connections = connections[preserve_mask].copy()
    randomizable_connections = connections[~preserve_mask].copy()

    pre_neurons = neurons[["root_id", "soma_x", "soma_y", "soma_z"]].copy()
    pre_neurons.columns = ["pre_root_id", "pre_x", "pre_y", "pre_z"]

    post_neurons = neurons[["root_id", "soma_x", "soma_y", "soma_z"]].copy()
    post_neurons.columns = ["post_root_id", "post_x", "post_y", "post_z"]

    randomizable_with_coords = randomizable_connections.merge(
        pre_neurons, on="pre_root_id"
    ).merge(post_neurons, on="post_root_id")

    # Use NumPy for distance calculations
    pre_coords = randomizable_with_coords[["pre_x", "pre_y", "pre_z"]].to_numpy()
    post_coords = randomizable_with_coords[["post_x", "post_y", "post_z"]].to_numpy()
    distances = np.linalg.norm(pre_coords - post_coords, axis=1)
    randomizable_with_coords["distance"] = distances

    randomizable_with_coords["bin"] = pd.qcut(
        randomizable_with_coords["distance"], bins, labels=False
    )

    # Parallelize bin shuffling
    shuffled_randomizable = Parallel(n_jobs=-1)(
        delayed(shuffle_within_bin_parallel)(
            randomizable_with_coords[randomizable_with_coords["bin"] == bin_id]
        )
        for bin_id in range(bins)
    )
    shuffled_randomizable = pd.concat(shuffled_randomizable)

    final_connections = pd.concat(
        [
            preserved_connections[["pre_root_id", "post_root_id", "syn_count"]],
            shuffled_randomizable,
        ],
        ignore_index=True,
    )

    # Check whether we are within tolerance
    real_length = compute_total_synapse_length(connections, neurons)
    final_length = compute_total_synapse_length(final_connections, neurons)
    assert (
        abs(final_length - real_length) / real_length < tolerance
    ), "Final length does not match real length"

    return final_connections


if __name__ == "__main__":
    np.random.seed(12345)

    connections, nc = load_data()
    total_length = compute_total_synapse_length(connections, nc)

    connections_shuffled = shuffle_post_root_id(connections)
    scaled_random = match_wiring_length_with_syn_scale(
        connections_shuffled,
        nc,
        total_length,
        scale_low=0.0,
        scale_high=2.0,
        max_iter=10,
        tolerance=0.05,
    )
    scaled_random["syn_count"] = np.round(scaled_random["syn_count"]).astype(np.int32)
    scaled_random.to_csv(
        os.path.join(PROJECT_ROOT, "new_data", "connections_random_pruned.csv"),
        index=False,
    )

    random_connections = create_length_preserving_random_network(
        connections, nc, bins=20, tolerance=0.05
    )
    random_connections.to_csv(
        os.path.join(PROJECT_ROOT, "new_data", "connections_random_binned.csv"),
        index=False,
    )
