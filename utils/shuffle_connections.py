import logging
import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from paths import PROJECT_ROOT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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


def add_coords(connections_df, coords_df):
    """Add pre and post neuron coordinates to connections dataframe"""
    # Add pre-neuron coordinates
    df = connections_df.merge(
        coords_df[["root_id", "soma_x", "soma_y", "soma_z"]],
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
        coords_df[["root_id", "soma_x", "soma_y", "soma_z"]],
        left_on="post_root_id",
        right_on="root_id",
        how="left",
        suffixes=("", "_post"),
    )
    
    # Remove unnecessary column
    if "root_id" in df.columns:
        df = df.drop("root_id", axis=1)

    # Rename columns for clarity
    df = df.rename(
        columns={
            "soma_x": "pre_x",
            "soma_y": "pre_y",
            "soma_z": "pre_z",
            "soma_x_post": "post_x",
            "soma_y_post": "post_y",
            "soma_z_post": "post_z",
        }
    )
    
    return df


def match_wiring_length_with_syn_scale(connections, nc, real_length, scale_low=0.0, scale_high=2.0, 
                                      max_iter=20, tolerance=0.01, allow_zeros=False):
    """Match wiring length by scaling synapse counts with fine-tuned convergence."""
    
    conns_scaled = connections.copy()
    logger.info("Starting synapse count scaling to match wiring length...")

    # PHASE 1: Binary search to get close to target
    for i in range(max_iter):
        mid = 0.5 * (scale_low + scale_high)
        if allow_zeros:
            conns_scaled["syn_count"] = np.round(connections["syn_count"] * mid)
        else:
            conns_scaled["syn_count"] = np.ceil(connections["syn_count"] * mid)
        length_est = compute_total_synapse_length(conns_scaled, nc)

        ratio = length_est / real_length
        logger.info(f"Iteration {i+1}/{max_iter}: scale={mid:.4f}, length={length_est:.2f}, ratio={ratio:.3f}")
        
        if length_est < real_length:
            scale_low = mid
        else:
            scale_high = mid

        if abs(ratio - 1.0) < 0.05:
            logger.info("Initial convergence achieved, moving to fine-tuning phase")
            break
    
    # Get the best scale from binary search
    best_scale = 0.5 * (scale_low + scale_high)
    
    # PHASE 2: Fine-tuning with deterministic adjustments
    logger.info("Starting fine-tuning phase...")
    
    # First create integer synapse counts with probabilistic rounding
    float_counts = connections["syn_count"] * best_scale
    int_counts = np.floor(float_counts).astype(int)
    fractions = float_counts - int_counts
    
    # Calculate distance for each connection
    conns_with_coords = add_coords(connections, nc)
    conns_with_coords["distance"] = np.sqrt(
        (conns_with_coords["pre_x"] - conns_with_coords["post_x"])**2 +
        (conns_with_coords["pre_y"] - conns_with_coords["post_y"])**2 +
        (conns_with_coords["pre_z"] - conns_with_coords["post_z"])**2
    )
    
    # Calculate impact = distance × fraction (how much adding 1 synapse affects length)
    conns_with_coords["impact"] = conns_with_coords["distance"] * fractions
    
    # Start with the base integer counts
    conns_with_coords["syn_count"] = int_counts
    
    # Ensure minimum synapse count is 1
    conns_with_coords.loc[conns_with_coords["syn_count"] < 1, "syn_count"] = 1
    
    # Calculate initial length
    current_length = compute_total_synapse_length(conns_with_coords[["pre_root_id", "post_root_id", "syn_count"]], nc)
    logger.info(f"Initial integer-based length: {current_length:.2f}")
    logger.info(f"Target length: {real_length:.2f}")
    logger.info(f"Initial ratio: {current_length/real_length:.4f}")
    
    # Iteratively adjust synapse counts to reach target
    max_fine_iter = 20
    for i in range(max_fine_iter):
        ratio = current_length / real_length
        
        if abs(ratio - 1.0) < tolerance:
            logger.info(f"Fine-tuning converged after {i+1} iterations")
            break
            
        # If current length is too small, increase counts on high-impact connections
        if current_length < real_length:
            # Sort by impact descending (add to highest impact connections first)
            adjustment_candidates = conns_with_coords.sort_values("impact", ascending=False)
            increment = True
        else:
            # Sort by impact ascending (remove from lowest impact connections first)
            adjustment_candidates = conns_with_coords.sort_values("impact", ascending=True)
            # Only consider connections with syn_count > 1 for decrementing
            adjustment_candidates = adjustment_candidates[adjustment_candidates["syn_count"] > 1]
            increment = False
        
        # Calculate how many connections to adjust
        # More aggressive adjustments when far from target, gentler when close
        adjustment_size = int(max(100, len(conns_with_coords) * abs(ratio - 1.0) * 0.1))
        adjustment_size = min(adjustment_size, len(adjustment_candidates))
        
        # Make adjustments
        if increment:
            conns_with_coords.loc[adjustment_candidates.index[:adjustment_size], "syn_count"] += 1
        else:
            conns_with_coords.loc[adjustment_candidates.index[:adjustment_size], "syn_count"] -= 1
        
        # Recalculate length
        new_length = compute_total_synapse_length(conns_with_coords[["pre_root_id", "post_root_id", "syn_count"]], nc)
        
        # Log progress
        logger.info(f"Fine-tuning iter {i+1}: adjusted {adjustment_size} connections, " +
                    f"new length={new_length:.2f}, ratio={new_length/real_length:.4f}")
        
        # Update for next iteration
        current_length = new_length
        
        # If we made adjustments but length is going in wrong direction, use smaller adjustments
        if (increment and new_length < current_length) or (not increment and new_length > current_length):
            logger.info("Reducing adjustment size due to unexpected direction change")
            adjustment_size = max(10, adjustment_size // 2)
    
    # Final results
    final_synapses = conns_with_coords["syn_count"].sum()
    logger.info(f"Final integer-based length: {current_length:.2f}")
    logger.info(f"Target length: {real_length:.2f}")
    logger.info(f"Final ratio: {current_length/real_length:.4f}")
    logger.info(f"Total synapses: {final_synapses}")
    
    return conns_with_coords[["pre_root_id", "post_root_id", "syn_count"]]


def match_wiring_length_with_random_pruning(connections, nc, real_length, scale_low=0.0, scale_high=2.0, 
                                       max_coarse_iter=20, max_fine_iter=50, tolerance=0.01, allow_zeros=False):
    """
    Match wiring length by scaling synapse counts with random pruning, using fully vectorized operations.
    
    Parameters:
    -----------
    connections : DataFrame
        DataFrame containing pre_root_id, post_root_id, and syn_count columns.
    nc : DataFrame 
        Neuron coordinates with root_id, x, y, z columns.
    real_length : float
        Target total wiring length to match.
    scale_low, scale_high : float
        Initial bounds for binary search.
    max_coarse_iter : int
        Maximum iterations for binary search.
    max_fine_iter : int
        Maximum iterations for fine-tuning.
    tolerance : float
        Acceptable relative error for the final result.
    allow_zeros : bool
        If True, allows connections to have zero synapses.
        
    Returns:
    --------
    DataFrame with pre_root_id, post_root_id, and adjusted syn_count columns.
    """
    import numpy as np
    import pandas as pd
    
    conns_scaled = connections.copy()
    logger.info("Starting synapse count scaling to match wiring length...")

    # PHASE 1: Binary search to get close to target
    for i in range(max_coarse_iter):
        mid = 0.5 * (scale_low + scale_high)
        if allow_zeros:
            conns_scaled["syn_count"] = np.round(connections["syn_count"] * mid)
        else:
            conns_scaled["syn_count"] = np.ceil(connections["syn_count"] * mid).astype(int)
        length_est = compute_total_synapse_length(conns_scaled, nc)

        ratio = length_est / real_length
        logger.info(f"Iteration {i+1}/{max_coarse_iter}: scale={mid:.4f}, length={length_est:.2f}, ratio={ratio:.3f}")
        
        if length_est < real_length:
            scale_low = mid
        else:
            scale_high = mid

        if abs(ratio - 1.0) < 0.05:
            logger.info("Binary search phase converged, moving to randomized fine-tuning")
            break
    
    # Get the best scale from binary search
    best_scale = 0.5 * (scale_low + scale_high)
    
    # PHASE 2: Vectorized random fine-tuning
    logger.info("Starting vectorized random fine-tuning...")
    
    # Add coordinate data and calculate distances vectorially
    conns_with_coords = add_coords(connections, nc)
    conns_with_coords["distance"] = np.sqrt(
        (conns_with_coords["pre_x"] - conns_with_coords["post_x"])**2 +
        (conns_with_coords["pre_y"] - conns_with_coords["post_y"])**2 +
        (conns_with_coords["pre_z"] - conns_with_coords["post_z"])**2
    )
    
    # Initialize synapse counts with rounded values from binary search
    # We use ceiling to ensure we don't get zeros (unless allowed)
    conns_with_coords["syn_count"] = np.ceil(connections["syn_count"] * best_scale).astype(int)
    
    # Set minimum synapse count if needed
    if not allow_zeros:
        conns_with_coords.loc[conns_with_coords["syn_count"] < 1, "syn_count"] = 1
    
    # Calculate current total length vectorially
    current_length = compute_total_synapse_length(
        conns_with_coords[["pre_root_id", "post_root_id", "syn_count"]], 
        nc
    )
    
    logger.info(f"Initial vectorized length: {current_length:.2f}")
    logger.info(f"Target length: {real_length:.2f}")
    logger.info(f"Initial ratio: {current_length/real_length:.4f}")
    
    # For each iteration, we'll randomly select connections to adjust
    min_synapses = 0 if allow_zeros else 1
    
    # Create lookup arrays for faster vectorized operations
    conn_indices = np.arange(len(conns_with_coords))
    distances = conns_with_coords["distance"].values
    syn_counts = conns_with_coords["syn_count"].values.copy()
    
    for i in range(max_fine_iter):
        length_diff = current_length - real_length
        ratio = current_length / real_length
        
        if abs(ratio - 1.0) < tolerance:
            logger.info(f"Vectorized random fine-tuning converged after {i+1} iterations")
            break
        
        # Determine if we need to add or remove synapses
        if length_diff < 0:  # Need to add synapses
            # Calculate how many synapses to add (estimate based on average distance)
            avg_distance = np.sum(distances * syn_counts) / np.sum(syn_counts)
            synapses_to_add = max(10, int(abs(length_diff) / avg_distance * 1.2))
            
            # Randomly select connection indices (with replacement is fine for this purpose)
            selected_indices = np.random.choice(conn_indices, 
                                               size=min(synapses_to_add, len(conn_indices)),
                                               replace=True)
            
            # Count occurrences of each index (how many synapses to add per connection)
            unique_indices, counts = np.unique(selected_indices, return_counts=True)
            
            # Vectorized update of synapse counts and total length
            syn_counts[unique_indices] += counts
            current_length += np.sum(distances[unique_indices] * counts)
        
        else:  # Need to remove synapses
            # Identify connections that have more than minimum synapses
            removable_mask = syn_counts > min_synapses
            removable_indices = conn_indices[removable_mask]
            
            if len(removable_indices) == 0:
                logger.warning("No more synapses can be removed while respecting constraints")
                break
            
            # Calculate how many synapses to remove
            avg_distance = np.sum(distances * syn_counts) / np.sum(syn_counts)
            synapses_to_remove = max(10, int(abs(length_diff) / avg_distance))
            
            # Randomly select indices to remove synapses from
            selected_indices = np.random.choice(
                removable_indices, 
                size=min(synapses_to_remove, len(removable_indices)),
                replace=True
            )
            
            # Count occurrences of each index
            unique_indices, counts = np.unique(selected_indices, return_counts=True)
            
            # Vectorized update of synapse counts and total length
            # First, calculate how many we can actually remove from each connection
            max_removable = syn_counts[unique_indices] - min_synapses
            actual_removals = np.minimum(counts, max_removable)
            
            # Update counts and length
            syn_counts[unique_indices] -= actual_removals
            current_length -= np.sum(distances[unique_indices] * actual_removals)
        
        # Log progress
        total_synapses = np.sum(syn_counts)
        logger.info(f"Random iter {i+1}: length={current_length:.2f}, " +
                   f"ratio={current_length/real_length:.4f}, " +
                   f"synapses={total_synapses}")
    
    # Update the DataFrame with the final counts
    conns_with_coords["syn_count"] = syn_counts
    
    # Final results
    final_synapses = np.sum(syn_counts)
    logger.info(f"Final vectorized length: {current_length:.2f}")
    logger.info(f"Target length: {real_length:.2f}")
    logger.info(f"Final ratio: {current_length/real_length:.4f}")
    logger.info(f"Total synapses: {final_synapses}")
    
    return conns_with_coords[["pre_root_id", "post_root_id", "syn_count"]]


def compute_individual_synapse_lengths(connections, neuron_coords):
    # use np linalg to compute the distance between the pre and post neurons
    conns_with_coords = add_coords(connections, neuron_coords)
    return np.linalg.norm(
        conns_with_coords[["pre_x", "pre_y", "pre_z"]] - conns_with_coords[["post_x", "post_y", "post_z"]], 
        axis=1
        )

def compute_total_synapse_length(connections, neuron_coords):
    """
    Compute the total wiring length of all synapses using compute_individual_synapse_lengths.    
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
    return np.sum(compute_individual_synapse_lengths(connections, neuron_coords) * connections["syn_count"])


def create_length_preserving_random_network(
    connections, neurons, bins=100, tolerance=0.1
):
    """
    Create a randomized network that preserves the total wiring length.
    Process bin by bin to reduce memory usage.
    """
    connections = connections.copy()
    neurons = neurons.copy()
    logger.info("Starting length-preserving network randomization...")

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
    logger.info("Calculating distances between neurons...")
    connections_with_coords = connections.merge(pre_neurons, on="pre_root_id").merge(
        post_neurons, on="post_root_id"
    )

    pre_coords = connections_with_coords[["pre_x", "pre_y", "pre_z"]].to_numpy()
    post_coords = connections_with_coords[["post_x", "post_y", "post_z"]].to_numpy()
    distances = np.linalg.norm(pre_coords - post_coords, axis=1)
    connections_with_coords["distance"] = distances

    # Get bin edges for consistent binning
    logger.info(f"Creating {bins} distance bins...")
    bin_edges = pd.qcut(connections_with_coords["distance"], bins, retbins=True)[1]

    # Process each bin separately to reduce memory usage
    logger.info("Shuffling connections within bins...")
    shuffled_connections = []

    # Create bins based on distance
    connections_with_coords["bin"] = pd.cut(
        connections_with_coords["distance"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    # Process each bin
    for bin_idx in tqdm(range(bins), desc="Processing bins"):
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
        logger.info(f"Processed bin {bin_idx + 1}/{bins}")

    # Combine all shuffled bins
    logger.info("Combining results from all bins...")
    final_connections = pd.concat(shuffled_connections, ignore_index=True)

    # Only keep the necessary columns
    final_connections = final_connections[["pre_root_id", "post_root_id", "syn_count"]]

    # Recalculate distances and check if within tolerance
    logger.info("Validating total wiring length...")
    real_length = compute_total_synapse_length(connections, neurons)
    final_length = compute_total_synapse_length(final_connections, neurons)
    dif_ratio = abs(final_length - real_length) / real_length

    logger.info(f"Original wiring length: {real_length:.2f}")
    logger.info(f"Randomized wiring length: {final_length:.2f}")
    logger.info(f"Difference ratio: {dif_ratio:.4f} (should be < {tolerance})")

    assert (
        dif_ratio < 1 + tolerance
    ), f"Final length differs by {dif_ratio:.4f}, exceeding tolerance of {tolerance}"

    return final_connections


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate randomized network connections')
    parser.add_argument('--unconstrained', action='store_true',
                      help='Generate unconstrained randomized network')
    parser.add_argument('--pruned', action='store_true',
                      help='Generate pruned randomized network')
    parser.add_argument('--binned', action='store_true',
                      help='Generate binned randomized network')
    args = parser.parse_args()
    
    # If no arguments provided, run all randomizations
    run_all = not (args.unconstrained or args.pruned or args.binned)
    
    # Load data
    logger.info("Loading data...")
    connections, nc = load_data()
    total_length = compute_total_synapse_length(connections, nc)
    logger.info(f"Total wiring length of original network: {total_length:.2f}")

    # Unconstrained randomization
    if run_all or args.unconstrained:
        logger.info("Starting unconstrained randomization...")
        connections_shuffled = shuffle_post_root_id(connections)
        connections_shuffled.to_csv(
            os.path.join(PROJECT_ROOT, "new_data", "connections_random_unconstrained.csv"),
            index=False,
        )
        logger.info("Unconstrained randomization completed")
    else:
        connections_shuffled = None

    # Pruned randomization
    if run_all or args.pruned:
        if connections_shuffled is None and (args.pruned or run_all):
            logger.info("Starting unconstrained randomization for pruned version...")
            connections_shuffled = shuffle_post_root_id(connections)
        
        logger.info("Starting synapse count scaling...")
        scaled_random = match_wiring_length_with_random_pruning(
            connections_shuffled,
            nc,
            total_length,
            scale_low=0.0,
            scale_high=2.0,
            max_coarse_iter=20,
            max_fine_iter=50,
            tolerance=0.01,
            allow_zeros=True,
        )
        scaled_random["syn_count"] = np.round(scaled_random["syn_count"]).astype(np.int32)
        scaled_random.to_csv(
            os.path.join(PROJECT_ROOT, "new_data", "connections_random_pruned.csv"),
            index=False,
        )
        logger.info("Synapse count scaling completed")
        
        # Clean up memory
        del scaled_random

    # Clean up memory
    if connections_shuffled is not None:
        del connections_shuffled

    # Binned randomization
    if run_all or args.binned:
        logger.info("Starting length-preserving randomization...")
        random_connections = create_length_preserving_random_network(
            connections, nc, bins=100, tolerance=0.01
        )
        random_connections.to_csv(
            os.path.join(PROJECT_ROOT, "new_data", "connections_random_binned.csv"),
            index=False,
        )
        logger.info("Length-preserving randomization completed")
