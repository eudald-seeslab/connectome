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

    soma_pre = df[["soma_x_pre", "soma_y_pre", "soma_z_pre"]].to_numpy()
    soma_post = df[["soma_x_post", "soma_y_post", "soma_z_post"]].to_numpy()
    distances = np.linalg.norm(soma_pre - soma_post, axis=1)

    return np.sum(distances * df["syn_count"].values)


def shuffle_post_root_id(connections):
    new_conns = connections.copy()
    shuffled_posts = np.random.permutation(new_conns["post_root_id"].values)
    new_conns["post_root_id"] = shuffled_posts
    return new_conns


def shuffle_within_bin(bin_group):
    if len(bin_group) <= 1:
        return bin_group
    shuffled_post_ids = np.random.permutation(bin_group['post_root_id'].values)
    result = bin_group.copy()
    result['post_root_id'] = shuffled_post_ids
    return result

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

    # final check that we are within tolerance
    final_length = compute_total_synapse_length(conns_scaled, nc)
    dif_ratio = abs(final_length - real_length) / real_length
    print(f"Pruned: ratio of final length to real length: {dif_ratio}")
    assert dif_ratio < tolerance, "Final length does not match real length"

    return conns_scaled


def create_length_preserving_random_network_parallel(
    connections, neurons, bins=10, tolerance=0.1
):
    # Bins > 30 or so kill your memory
    connections = connections.copy()
    neurons = neurons.copy()

    # Perform type conversions once
    connections = connections.astype(
        {"pre_root_id": int, "post_root_id": int, "syn_count": int}
    )
    neurons = neurons.astype({"root_id": int})

    pre_neurons = neurons[["root_id", "soma_x", "soma_y", "soma_z"]].copy()
    pre_neurons.columns = ["pre_root_id", "pre_x", "pre_y", "pre_z"]

    post_neurons = neurons[["root_id", "soma_x", "soma_y", "soma_z"]].copy()
    post_neurons.columns = ["post_root_id", "post_x", "post_y", "post_z"]

    randomizable_with_coords = connections.merge(
        pre_neurons, on="pre_root_id"
    ).merge(post_neurons, on="post_root_id")

    pre_coords = randomizable_with_coords[["pre_x", "pre_y", "pre_z"]].to_numpy()
    post_coords = randomizable_with_coords[["post_x", "post_y", "post_z"]].to_numpy()
    distances = np.linalg.norm(pre_coords - post_coords, axis=1)
    randomizable_with_coords["distance"] = distances

    randomizable_with_coords["bin"] = pd.qcut(
        randomizable_with_coords["distance"], bins, labels=False
    )

    # Parallelize bin shuffling
    shuffled_randomizable = Parallel(n_jobs=-1)(
        delayed(shuffle_within_bin)(
            randomizable_with_coords[randomizable_with_coords["bin"] == bin_id]
        )
        for bin_id in range(bins)
    )
    final_connections = pd.concat(shuffled_randomizable)

    # Check whether we are within tolerance
    real_length = compute_total_synapse_length(connections, neurons)
    final_length = compute_total_synapse_length(final_connections, neurons)
    dif_ratio = abs(final_length - real_length) / real_length

    print(f"Binned: ratio of final length to real length: {dif_ratio}")
    assert dif_ratio < tolerance, "Final length does not match real length"

    return final_connections


def create_length_preserving_random_network(
    connections, neurons, bins=100, tolerance=0.1
):
    """
    Create a randomized network that preserves the total wiring length.
    Process bin by bin to reduce memory usage.
    """
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    connections = connections.copy()
    neurons = neurons.copy()

    # Perform type conversions once
    connections = connections.astype(
        {"pre_root_id": int, "post_root_id": int, "syn_count": int}
    )
    neurons = neurons.astype({"root_id": int})

    # Create position dataframes for pre and post neurons
    pre_neurons = neurons[["root_id", "soma_x", "soma_y", "soma_z"]].copy()
    pre_neurons.columns = ["pre_root_id", "pre_x", "pre_y", "pre_z"]

    post_neurons = neurons[["root_id", "soma_x", "soma_y", "soma_z"]].copy()
    post_neurons.columns = ["post_root_id", "post_x", "post_y", "post_z"]

    # Add distance column to connections
    print("Calculating distances...")
    connections_with_coords = connections.merge(pre_neurons, on="pre_root_id").merge(
        post_neurons, on="post_root_id"
    )

    pre_coords = connections_with_coords[["pre_x", "pre_y", "pre_z"]].to_numpy()
    post_coords = connections_with_coords[["post_x", "post_y", "post_z"]].to_numpy()
    distances = np.linalg.norm(pre_coords - post_coords, axis=1)
    connections_with_coords["distance"] = distances

    # Get bin edges for consistent binning
    print(f"Creating {bins} distance bins...")
    bin_edges = pd.qcut(connections_with_coords["distance"], bins, retbins=True)[1]

    # Process each bin separately to reduce memory usage
    print("Shuffling connections within bins...")
    shuffled_connections = []

    # Create bins based on distance
    connections_with_coords["bin"] = pd.cut(
        connections_with_coords["distance"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    # Process each bin
    for bin_idx in tqdm(range(bins)):
        # Get only connections in current bin
        bin_mask = connections_with_coords["bin"] == bin_idx
        if not np.any(bin_mask):
            continue

        bin_data = connections_with_coords[bin_mask].copy()

        # Shuffle both pre and post IDs
        pre_ids = bin_data['pre_root_id'].values.copy()
        post_ids = bin_data['post_root_id'].values.copy()
        syn_counts = bin_data['syn_count'].values

        # This is crucial: shuffle the pre-post pairs, not individually
        indices = np.random.permutation(len(pre_ids))
        shuffled_pre = pre_ids[indices]
        shuffled_post = post_ids[indices]

        # Create new connections that preserve in/out degree distribution
        bin_result = pd.DataFrame({
            'pre_root_id': shuffled_pre,
            'post_root_id': shuffled_post,
            'syn_count': syn_counts
        })

        shuffled_connections.append(bin_result)

    # Combine all shuffled bins
    print("Combining results...")
    final_connections = pd.concat(shuffled_connections, ignore_index=True)

    # Only keep the necessary columns
    final_connections = final_connections[["pre_root_id", "post_root_id", "syn_count"]]

    # Recalculate distances and check if within tolerance
    print("Validating total wiring length...")
    real_length = compute_total_synapse_length(connections, neurons)
    final_length = compute_total_synapse_length(final_connections, neurons)
    dif_ratio = abs(final_length - real_length) / real_length

    print(f"Original wiring length: {real_length}")
    print(f"Randomized wiring length: {final_length}")
    print(f"Difference ratio: {dif_ratio:.4f} (should be < {tolerance})")

    assert (
        dif_ratio < 1 + tolerance
    ), f"Final length differs by {dif_ratio:.4f}, exceeding tolerance of {tolerance}"

    return final_connections


def shuffle_within_bin(bin_group):
    """Shuffle post_root_id values within a distance bin."""
    if len(bin_group) <= 1:
        return bin_group

    result = bin_group.copy()
    shuffled_post_ids = np.random.permutation(bin_group["post_root_id"].values)
    result["post_root_id"] = shuffled_post_ids

    return result


if __name__ == "__main__":

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
    # remove objects to save memory
    del connections_shuffled
    del scaled_random
    
    random_connections = create_length_preserving_random_network(
        connections, nc, bins=100, tolerance=0.05
    )
    random_connections.to_csv(
        os.path.join(PROJECT_ROOT, "new_data", "connections_random_binned.csv"),
        index=False,
    )
