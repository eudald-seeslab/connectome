from utils.helpers import add_distance_column, get_logger


import numpy as np
import pandas as pd

# Get logger for this module
logger = get_logger(__name__)


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