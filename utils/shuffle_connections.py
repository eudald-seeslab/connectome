# Standard library
import logging
import os
import argparse

# Third-party
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local modules
from notebooks.visualization.activation_plots import plot_synapse_length_distributions
from paths import PROJECT_ROOT
from utils.helpers import (
    compute_total_synapse_length,
    add_distance_column,
    load_connections,
    load_neuron_annotations,
    shuffle_post_root_id,
    setup_logging,
)

# Configure logging once for the entire application
setup_logging(level=logging.INFO)

# Module-specific logger
logger = logging.getLogger(__name__)


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
    
    # Add coordinates and a pre-computed distance column
    conns_with_coords = add_distance_column(connections, nc)
    
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
    conns_with_coords = add_distance_column(connections, nc)
    
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
    logger.info("Calculating distances between neurons...")
    connections_with_coords = add_distance_column(conns, nc, distance_col="distance")
    distances = connections_with_coords["distance"].values

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
    connections_with_coords = add_distance_column(connections, neurons, distance_col="distance")

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


def match_wiring_length_with_connection_pruning(
    connections: pd.DataFrame,
    nc: pd.DataFrame,
    real_length: float,
    tolerance: float = 0.01,
    max_iter: int = 100,
    adaptive_batch: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Adjusts a set of connections to match a target wiring length by completely
    removing entire connections with purely random selection (no bias toward
    connection length).
    
    Parameters:
    -----------
    connections : DataFrame
        Contains pre_root_id, post_root_id, and syn_count columns
    nc : DataFrame
        Contains root_id and position coordinates for each neuron
    real_length : float
        Target total wiring length to match
    tolerance : float
        Acceptable relative error for the final result
    max_iter : int
        Maximum number of iterations
    adaptive_batch : bool
        Whether to use adaptive batch sizing for faster convergence
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    DataFrame with pre_root_id, post_root_id, and syn_count columns
    """
    rng = np.random.default_rng(random_state)
    conns = connections.copy()
    
    # Add coordinate data and calculate distances vectorially
    conns_with_coords = add_distance_column(conns, nc)
    
    # Calculate initial wiring length
    syn_counts = conns["syn_count"].values
    current_length = float(np.sum(conns_with_coords["distance"] * syn_counts))
    
    logger.info(f"[CONN_PRUNE] Initial length: {current_length:,.2f}")
    logger.info(f"[CONN_PRUNE] Target length: {real_length:,.2f}")
    
    if current_length <= real_length:
        logger.warning("[CONN_PRUNE] Initial length already <= target; no pruning needed.")
        return conns[["pre_root_id", "post_root_id", "syn_count"]]
    
    # Create a mask to track which connections to keep (initially all True)
    keep_mask = np.ones(len(conns), dtype=bool)
    total_conns = len(conns)
    
    # Calculate connection contribution to total wiring length (for tracking only)
    length_contributions = conns_with_coords["distance"].values * syn_counts
    
    # Start with 0.5% of connections to remove in first iteration
    if adaptive_batch:
        # More aggressive initial batch size
        initial_removal_proportion = min(0.02, 100000 / total_conns)
    else:
        initial_removal_proportion = 0.005  # Fixed 0.5%
    
    for iteration in range(1, max_iter + 1):
        ratio = current_length / real_length
        
        # If we've reached the target length within tolerance, stop
        if abs(ratio - 1.0) <= tolerance:
            logger.info(f"[CONN_PRUNE] Converged with ratio {ratio:.4f}")
            break
            
        # Adaptive batch sizing based on how far we are from target
        if adaptive_batch:
            # Adjust batch size based on ratio, but keep pure random selection
            removal_proportion = min(
                initial_removal_proportion * ratio,  # Linear scaling with ratio
                0.2  # Don't remove more than 20% at once
            )
        else:
            # Fixed proportion
            removal_proportion = initial_removal_proportion
        
        # Calculate actual batch size (number of connections to remove)
        current_batch_size = min(
            int(total_conns * removal_proportion),  # Based on proportion
            int(np.sum(keep_mask) * 0.1)  # Don't remove more than 10% of remaining
        )
        
        # Ensure we're removing at least some connections
        current_batch_size = max(current_batch_size, min(1000, int(np.sum(keep_mask) * 0.01)))
        
        # Make sure batch size isn't larger than remaining connections
        current_batch_size = min(current_batch_size, int(np.sum(keep_mask)))
        
        if current_batch_size <= 0:
            logger.warning("[CONN_PRUNE] No more connections can be removed.")
            break
            
        # PURELY RANDOM selection of connections to remove
        candidate_indices = np.where(keep_mask)[0]
        to_remove_indices = rng.choice(
            candidate_indices,
            size=current_batch_size,
            replace=False
        )
        
        # Update the keep mask
        keep_mask[to_remove_indices] = False
        
        # Update current length
        length_reduction = np.sum(length_contributions[to_remove_indices])
        current_length -= length_reduction
        
        # Log progress
        remaining_conns = int(np.sum(keep_mask))
        removed_pct = (total_conns - remaining_conns) / total_conns * 100
        
        logger.info(
            f"[CONN_PRUNE] Iter {iteration:>2}: removed {len(to_remove_indices):,} connections, "
            f"length = {current_length:,.2f}, "
            f"ratio = {current_length/real_length:.4f}, "
            f"remaining = {remaining_conns:,}/{total_conns:,} ({100-removed_pct:.1f}%)"
        )
        
        # If we've gone below the target, restore some connections
        if current_length < real_length:
            logger.info("[CONN_PRUNE] Overshot target length, restoring connections randomly...")
            
            # Get all removed connections
            removed_mask = ~keep_mask
            removed_indices = np.where(removed_mask)[0]
            
            # If there are no removed connections, break
            if len(removed_indices) == 0:
                break
            
            # PURELY RANDOM shuffling of removed indices for unbiased restoration
            shuffled_indices = rng.permutation(removed_indices)
            
            # Add connections back until we're above the target again
            connections_restored = 0
            for idx in shuffled_indices:
                keep_mask[idx] = True
                current_length += length_contributions[idx]
                connections_restored += 1
                
                if current_length >= real_length:
                    break
                    
            # Log the adjustment
            remaining_conns = int(np.sum(keep_mask))
            removed_pct = (total_conns - remaining_conns) / total_conns * 100
            
            logger.info(
                f"[CONN_PRUNE] Randomly restored {connections_restored} connections: "
                f"length = {current_length:,.2f}, "
                f"ratio = {current_length/real_length:.4f}, "
                f"remaining = {remaining_conns:,}/{total_conns:,} ({100-removed_pct:.1f}%)"
            )
    
    # Create final pruned connections dataframe
    pruned_conns = conns[keep_mask].copy()
    
    logger.info(
        f"[CONN_PRUNE] FINAL: length = {current_length:,.2f}, "
        f"ratio = {current_length/real_length:.4f}, "
        f"connections = {len(pruned_conns):,}/{total_conns:,} ({len(pruned_conns)/total_conns*100:.1f}%)"
    )
    
    return pruned_conns[["pre_root_id", "post_root_id", "syn_count"]]


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate randomized network connections')
    parser.add_argument('--unconstrained', action='store_true',
                      help='Generate unconstrained randomized network')
    parser.add_argument('--pruned', action='store_true',
                      help='Generate pruned randomized network')
    parser.add_argument('--conn_pruned', action='store_true',
                      help='Generate connection-pruned randomized network (removes entire connections)')
    parser.add_argument('--binned', action='store_true',
                      help='Generate binned randomized network')
    parser.add_argument("--plot_results", action="store_true",
                      help="Plot the results")
    args = parser.parse_args()
    
    # If no arguments provided, run all randomizations
    run_all = not (args.unconstrained or args.pruned or args.conn_pruned or args.binned)
    
    # Load data
    logger.info("Loading data...")
    connections = load_connections()
    neuron_coordinates = load_neuron_annotations()
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

    # Pruned randomization (synapse count scaling)
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
    
    # Connection-wise pruned randomization
    if run_all or args.conn_pruned:
        if random_unconstrained is None and (args.conn_pruned or run_all):
            logger.info("Starting unconstrained randomization for connection-pruned version...")
            random_unconstrained = shuffle_post_root_id(connections)
        
        logger.info("Starting connection-wise pruning...")
        random_conn_pruned = match_wiring_length_with_connection_pruning(
            random_unconstrained,
            neuron_coordinates,
            total_length,
            tolerance=0.01,
            max_iter=100,
            adaptive_batch=True,
            random_state=42
        )
        random_conn_pruned.to_csv(
            os.path.join(PROJECT_ROOT, "new_data", "connections_random_conn_pruned.csv"),
            index=False,
        )
        logger.info("Connection-wise pruning completed")
    else:
        random_conn_pruned = None

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
    else:
        random_binned = None

    # Plot synapse length distributions
    if args.plot_results:
        conns_to_plot = {"Original": connections}
        if random_unconstrained is not None:
            conns_to_plot["Random unconstrained"] = random_unconstrained
        if args.pruned or run_all:
            conns_to_plot["Random pruned"] = random_pruned
        if args.conn_pruned or run_all:
            conns_to_plot["Random conn. pruned"] = random_conn_pruned
        if args.binned or run_all:
            conns_to_plot["Random bin-wise"] = random_binned

        fig1, fig2 = plot_synapse_length_distributions(neuron_coordinates, conns_to_plot, use_density=False)
        plots_path = os.path.join(PROJECT_ROOT, "utils", "plots")
        os.makedirs(plots_path, exist_ok=True)
        fig1.savefig(os.path.join(plots_path, "synapse_length_distributions.png"), dpi=300)
        fig2.savefig(os.path.join(plots_path, "synapse_length_distributions_density.png"), dpi=300)
