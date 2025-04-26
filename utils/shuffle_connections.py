import logging
import os
import argparse

from matplotlib import pyplot as plt
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


def load_connections():
    return pd.read_csv(
        os.path.join(PROJECT_ROOT, "new_data", "connections.csv"),
        dtype={
            "pre_root_id": "string",
            "post_root_id": "string",
            "syn_count": np.int32,
        },
    )

def load_neuron_coordinates():
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
    nc = nc.drop(columns=["pos_x", "pos_y", "pos_z"]).rename(columns={"soma_x": "pos_x", "soma_y": "pos_y", "soma_z": "pos_z"})

    return nc


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


def match_wiring_length_with_random_pruning_old(connections, nc, real_length,max_iter_coarse=20, max_iter_fine=50, 
                                            tolerance=0.01, allow_zeros=False):
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
    max_iter : int
        Maximum iterations for binary search.
    tolerance : float
        Acceptable relative error for the final result.
    allow_zeros : bool
        If True, allows connections to have zero synapses.
        
    Returns:
    --------
    DataFrame with pre_root_id, post_root_id, and adjusted syn_count columns.
    """
    
    conns_scaled = connections.copy()
    logger.info("Starting synapse count scaling to match wiring length...")

    # PHASE 1: Simple adaptive scaling starting at 0.5
    scale = 0.5  # Initial scale
    prev_ratio = None

    for i in range(max_iter_coarse):
        # Apply scaling
        if allow_zeros:
            conns_scaled["syn_count"] = np.round(connections["syn_count"] * scale)
        else:
            conns_scaled["syn_count"] = np.ceil(connections["syn_count"] * scale).astype(int)
        
        length_est = compute_total_synapse_length(conns_scaled, nc)
        ratio = length_est / real_length
        
        logger.info(f"Iteration {i+1}/{max_iter_coarse}: scale={scale:.4f}, length={length_est:.2f}, ratio={ratio:.3f}")
        
        # Check if we're close enough
        if abs(ratio - 1.0) < tolerance * 5:
            logger.info("Adaptive scaling converged, moving to randomized fine-tuning")
            break
        
        # Intelligent scaling based on ratio
        if ratio < 1.0:  # Need to increase scale
            # Simply adjust scale by dividing by the ratio
            # If ratio is 0.8, multiplying by 1/0.8 = 1.25 should get us closer to target
            new_scale = scale / ratio
        else:  # Need to decrease scale
            # Adjust scale by multiplying by the ratio
            # If ratio is 1.2, multiplying by 1/1.2 = 0.83 should get us closer to target
            new_scale = scale / ratio
        
        # Limit the change to avoid oscillation
        max_change = 0.3 * scale
        if abs(new_scale - scale) > max_change:
            if new_scale > scale:
                scale = scale + max_change
            else:
                scale = scale - max_change
        else:
            scale = new_scale
            
        # Add dampening as we get closer to prevent overshooting
        if prev_ratio is not None and abs(ratio - 1.0) < 0.2:
            # If we're changing direction, dampen more aggressively
            if (ratio < 1.0 and prev_ratio > 1.0) or (ratio > 1.0 and prev_ratio < 1.0):
                scale = 0.7 * scale + 0.3 * (scale / ratio)
            else:
                scale = 0.5 * scale + 0.5 * (scale / ratio)
                
        prev_ratio = ratio
    
    
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
    if allow_zeros:
        conns_with_coords["syn_count"] = np.round(connections["syn_count"] * scale).astype(int)
    else:
        conns_with_coords["syn_count"] = np.ceil(connections["syn_count"] * scale).astype(int)
    
    # Calculate current total length vectorially
    current_length = compute_total_synapse_length(
        conns_with_coords[["pre_root_id", "post_root_id", "syn_count"]], 
        nc
    )
    
    logger.info(f"Initial vectorized length: {current_length:.2f}")
    logger.info(f"Target length: {real_length:.2f}")
    logger.info(f"Initial ratio: {current_length/real_length:.4f}")
    
    min_synapses = 0 if allow_zeros else 1
    
    # Create lookup arrays for faster vectorized operations
    conn_indices = np.arange(len(conns_with_coords))
    distances = conns_with_coords["distance"].values
    syn_counts = conns_with_coords["syn_count"].values.copy()
    
    for i in range(max_iter_fine):
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
            eligible_indices = conn_indices  # All connections are eligible for adding
            
            # Randomly select indices to add synapses to
            selected_indices = np.random.choice(eligible_indices, 
                                               size=min(synapses_to_add, len(eligible_indices)),
                                               replace=True)
            
            # Count occurrences of each index (how many synapses to add per connection)
            unique_indices, counts = np.unique(selected_indices, return_counts=True)
            
            # Update total length (vectorized)
            current_length += np.sum(distances[unique_indices] * counts)
            
            # Update synapse counts (vectorized)
            syn_counts[unique_indices] += counts
        
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
            # We'll use choice with replacement and then count occurrences
            selected_indices = np.random.choice(
                removable_indices, 
                size=min(synapses_to_remove, len(removable_indices)),
                replace=True
            )
            
            # Count occurrences of each index
            unique_indices, counts = np.unique(selected_indices, return_counts=True)
            
            # Compute maximum removable synapses for each selected connection (vectorized)
            max_removable = syn_counts[unique_indices] - min_synapses
            
            # Limit removals to not exceed maximum removable counts (vectorized)
            counts = np.minimum(counts, max_removable)
            
            # Only keep positive counts
            mask = counts > 0
            if np.any(mask):
                # Apply updates vectorially
                valid_indices = unique_indices[mask]
                valid_counts = counts[mask]
                
                # Update total length (vectorized dot product)
                current_length -= np.sum(distances[valid_indices] * valid_counts)
                
                # Update synapse counts (vectorized)
                syn_counts[valid_indices] -= valid_counts
        
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

def match_wiring_length_with_random_pruning(
    connections: pd.DataFrame,
    nc: pd.DataFrame,
    real_length: float,
    tolerance: float = 0.01,
    max_iter: int = 6,
    allow_zeros: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Ajusta un conjunt de connexions 'unconstrained' perquè el wiring length
    coincideixi amb `real_length`, eliminant sinapsis de forma aleatòria i
    sense biaixos (binomial).  La forma de la distribució de distàncies es
    conserva estadísticament.

    Retorna un DataFrame amb els mateixos pre/post però amb syn_count modificat.
    """
    rng = np.random.default_rng(random_state)
    conns = connections.copy()

    # ------------------------------------------------------------
    # 1. Distància d'aquest parell pre–post
    # ------------------------------------------------------------
    coords = add_coords(conns, nc)
    distances = np.linalg.norm(
        coords[["pre_x", "pre_y", "pre_z"]].values -
        coords[["post_x", "post_y", "post_z"]].values,
        axis=1,
    )

    orig_counts = conns["syn_count"].to_numpy()
    unconstrained_len = float(np.sum(distances * orig_counts))

    logger.info(f"[PRUNE] unconstrained_len = {unconstrained_len:,.2f}")
    logger.info(f"[PRUNE] target_len        = {real_length:,.2f}")

    if unconstrained_len <= real_length:
        logger.warning("[PRUNE] la xarxa 'unconstrained' ja és ≤ target; no cal pruning.")
        return conns[["pre_root_id", "post_root_id", "syn_count"]]

    # ------------------------------------------------------------
    # 2. Prova–i-reajusta de la probabilitat p
    # ------------------------------------------------------------
    p = real_length / unconstrained_len    # valor d'arrencada (0<p<1)

    for step in range(1, max_iter + 1):
        new_counts = rng.binomial(orig_counts, p)

        if not allow_zeros:
            zero_mask = (orig_counts > 0) & (new_counts == 0)
            new_counts[zero_mask] = 1     # garantim ≥1 sinapsi per connexió

        new_len = float(np.sum(distances * new_counts))
        ratio   = new_len / real_length
        total_syn = int(new_counts.sum())

        logger.info(
            f"[PRUNE] iter {step:>2}: p = {p:.6f}  "
            f"len = {new_len:,.2f}  "
            f"ratio = {ratio:.4f}  "
            f"synapses = {total_syn:,}"
        )

        if abs(ratio - 1.0) <= tolerance:
            break

        # Ajust multiplicatiu sobre p.  (típic control proporcional)
        p *= real_length / new_len

    conns["syn_count"] = new_counts.astype(int)
    logger.info(
        f"[PRUNE] FINAL : len = {new_len:,.2f}  "
        f"ratio = {ratio:.4f}  "
        f"synapses = {total_syn:,}"
    )
    return conns[["pre_root_id", "post_root_id", "syn_count"]]


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
    pre_neurons = neurons[["root_id", "pos_x", "pos_y", "pos_z"]].copy()
    pre_neurons.columns = ["pre_root_id", "pre_x", "pre_y", "pre_z"]

    post_neurons = neurons[["root_id", "pos_x", "pos_y", "pos_z"]].copy()
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


plots_dir = os.path.join(PROJECT_ROOT, "plots")
def plot_synapse_length_distributions(neuron_coords, conns_dict, plots_dir=plots_dir, use_density=True):

    titles  = conns_dict.keys()
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Pre-calculem distàncies per a cadascun dels quatre dataframes
    dists   = {name: compute_individual_synapse_lengths(df, neuron_coords)
               for name, df in conns_dict.items()}
    weights = {name: df["syn_count"].to_numpy()
               for name, df in conns_dict.items()}

    # Binat com abans (99 % per treure extrems)
    all_d   = np.concatenate(list(dists.values()))
    max_len = np.percentile(all_d, 99)
    bins    = np.linspace(0, max_len, 100)

    # Primera passada per obtenir la y-max comuna
    max_val = 0
    for name in titles:
        hist, _ = np.histogram(dists[name], bins=bins,
                               weights=weights[name], density=use_density)
        max_val = max(max_val, hist.max())
    max_val *= 1.1        # petit marge

    # ——— Figura ———
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True,
                            constrained_layout=True)

    total_mm = {}             # mm totals per a l'annotació

    for ax, title, col in zip(axs, titles, colors):
        w  = weights[title]
        L  = dists[title]
        ax.hist(L, bins=bins, weights=w, density=use_density,
                color=col, alpha=0.7)

        # Mean (ponderat!)
        mean_nm = np.average(L, weights=w)
        ax.axvline(mean_nm, ls='--', c='k', lw=1)
        ax.text(mean_nm*1.05, 0.8*max_val,
                f"Mean: {mean_nm:,.2f} nm", fontsize=9)

        # Total wiring length (mm)
        tot_nm   = float(np.sum(L * w))
        tot_mm   = tot_nm / 1e9
        total_mm[title] = tot_mm
        ax.text(0.95, 0.85, f"Total: {tot_mm:,.1f} mm",
                transform=ax.transAxes, ha='right',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_ylim(0, max_val)
        ax.set_ylabel("Density" if use_density else "Count", fontsize=10)
        ax.set_title(title, fontsize=12)

    axs[-1].set_xlabel("Synapse Length (nm)", fontsize=12)

    plt.savefig(os.path.join(plots_dir, "shuffling_distributions.png"),
                dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(os.path.join(plots_dir, "shuffling_distributions.pdf"),
                bbox_inches="tight")

    return fig, total_mm


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate randomized network connections')
    parser.add_argument('--unconstrained', action='store_true',
                      help='Generate unconstrained randomized network')
    parser.add_argument('--pruned', action='store_true',
                      help='Generate pruned randomized network')
    parser.add_argument('--binned', action='store_true',
                      help='Generate binned randomized network')
    parser.add_argument("--plot_results", action="store_true",
                      help="Plot the results")
    args = parser.parse_args()
    
    # If no arguments provided, run all randomizations
    run_all = not (args.unconstrained or args.pruned or args.binned)
    
    # Load data
    logger.info("Loading data...")
    connections = load_connections()
    neuron_coordinates = load_neuron_coordinates()
    total_length = compute_total_synapse_length(connections, neuron_coordinates)
    logger.info(f"Total wiring length of original network: {total_length:.2f}")

    # Unconstrained randomization
    if run_all or args.unconstrained:
        logger.info("Starting unconstrained randomization...")
        random_unconstrained = shuffle_post_root_id(connections)
        random_unconstrained.to_csv(
            os.path.join(PROJECT_ROOT, "new_data", "connections_random_unconstrained.csv"),
            index=False,
        )
        logger.info("Unconstrained randomization completed")
    else:
        random_unconstrained = None

    # Pruned randomization
    if run_all or args.pruned:
        if random_unconstrained is None and (args.pruned or run_all):
            logger.info("Starting unconstrained randomization for pruned version...")
            random_unconstrained = shuffle_post_root_id(connections)
        
        logger.info("Starting synapse count scaling...")
        random_pruned = match_wiring_length_with_random_pruning(
            random_unconstrained,
            neuron_coordinates,
            total_length,
            tolerance=0.01,
            allow_zeros=True,
        )
        random_pruned.to_csv(
            os.path.join(PROJECT_ROOT, "new_data", "connections_random_pruned.csv"),
            index=False,
        )
        logger.info("Synapse count scaling completed")

    # Binned randomization
    if run_all or args.binned:
        logger.info("Starting length-preserving randomization...")
        random_binned = create_length_preserving_random_network(
            connections, neuron_coordinates, bins=100, tolerance=0.01
        )
        random_binned.to_csv(
            os.path.join(PROJECT_ROOT, "new_data", "connections_random_binned.csv"),
            index=False,
        )
        logger.info("Length-preserving randomization completed")

    # Plot synapse length distributions
    if args.plot_results:
        plot_synapse_length_distributions(neuron_coordinates, {
            "Original": connections,
            "Random unconstrained": random_unconstrained,
            "Random pruned": random_pruned,
            "Random bin-wise": random_binned
        })
