# ------------------------------------------------------------
# NumPy/Numba utilities
# ------------------------------------------------------------

import numpy as np
import pandas as pd
from utils.helpers import get_logger
from utils.randomizers.numba_helpers import euclidean_rows

# ------------------------------------------------------------

# Get module-level logger
logger = get_logger(__name__)

def match_wiring_length_with_connection_pruning(
    connections: pd.DataFrame,
    nc: pd.DataFrame,
    real_length: float,
    tolerance: float = 0.01,
    max_iter: int = 100,
    adaptive_batch: bool = True,
    random_state: int | None = None,
    *,
    silent: bool = False,
    fast: bool = True,
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
    silent : bool
        Whether to suppress logging
    fast : bool
        Whether to use fast mode (single-pass removal)

    Returns:
    --------
    DataFrame with pre_root_id, post_root_id, and syn_count columns
    """
    rng = np.random.default_rng(random_state)
    conns = connections.copy()

    prev_disabled = logger.disabled
    logger.disabled = silent or prev_disabled

    # ------------------------------------------------------------
    # NumPy distance vector (no pandas merge)
    # ------------------------------------------------------------

    pre_ids  = conns["pre_root_id"].to_numpy(dtype=np.int64)
    post_ids = conns["post_root_id"].to_numpy(dtype=np.int64)
    syn_counts = conns["syn_count"].to_numpy(dtype=np.int32)

    roots = nc["root_id"].to_numpy(dtype=np.int64)
    coords = nc[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float32)

    sort_idx = np.argsort(roots)
    roots_sorted = roots[sort_idx]
    coords_sorted = coords[sort_idx]

    idx_pre = np.searchsorted(roots_sorted, pre_ids)
    idx_post = np.searchsorted(roots_sorted, post_ids)

    pre_xyz = coords_sorted[idx_pre]
    post_xyz = coords_sorted[idx_post]

    distances = euclidean_rows(pre_xyz, post_xyz)

    # Calculate initial wiring length
    current_length = float(np.sum(distances * syn_counts))

    # Store per-connection contribution for fast updates
    length_contributions = distances * syn_counts

    if current_length <= real_length:
        logger.warning("[CONN_PRUNE] Initial length already <= target; no pruning needed.")
        return conns[["pre_root_id", "post_root_id", "syn_count"]]

    total_conns = len(conns)

    # --------------------------------------------------------------------
    # Fast single-pass removal (random order) – skips iterative loop
    # --------------------------------------------------------------------
    if fast:
        # ----------------------------------------------------------------
        # Uniform Bernoulli sampling – each connection kept with prob q so
        # selection is unbiased with respect to length.
        # ----------------------------------------------------------------

        q = real_length / current_length  # keep probability ≈ desired ratio
        keep_mask = rng.random(total_conns) < q

        current_length = float(np.sum(length_contributions[keep_mask]))

        # Quick stochastic fine-tuning: add/remove random edges until within tol
        for _ in range(5):  # at most 5 lightweight adjustments
            error_ratio = (current_length - real_length) / real_length

            if abs(error_ratio) <= tolerance:
                break

            if error_ratio > 0:  # still too long → drop more
                candidates = np.where(keep_mask)[0]
                # probability proportional to required reduction
                p = min(1.0, error_ratio)
                drop = rng.random(candidates.size) < p
                if drop.any():
                    keep_mask[candidates[drop]] = False
                    current_length -= float(np.sum(length_contributions[candidates[drop]]))
            else:  # too short → restore some removed edges
                candidates = np.where(~keep_mask)[0]
                p = min(1.0, -error_ratio)
                add = rng.random(candidates.size) < p
                if add.any():
                    keep_mask[candidates[add]] = True
                    current_length += float(np.sum(length_contributions[candidates[add]]))

        if not silent:
            removed = (~keep_mask).sum()
            logger.info(
                f"[CONN_PRUNE] fast-uniform mode removed {removed:,} connections; len {current_length:,.2f} (ratio {current_length/real_length:.4f})"
            )

        if abs(current_length - real_length) / real_length <= tolerance:
            pruned_conns = conns[keep_mask].copy()

            logger.disabled = prev_disabled
            return pruned_conns[["pre_root_id", "post_root_id", "syn_count"]]

        # Otherwise fall back to iterative fine-tuning below

    # --------------------------------------------------------------------
    # Iterative fine-tuning loop (original algorithm)
    # --------------------------------------------------------------------

    # TODO: we can probably remove this

    keep_mask = np.ones(total_conns, dtype=bool)

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
            if not silent:
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

        # Guarantee progress for tiny graphs – remove at least one connection
        current_batch_size = max(
            current_batch_size,
            1,
            min(1000, int(np.sum(keep_mask) * 0.01)),
        )

        # Make sure batch size isn't larger than remaining connections
        current_batch_size = min(current_batch_size, int(np.sum(keep_mask)))

        if current_batch_size <= 0:
            if not silent:
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

        if not silent:
            logger.info(
                f"[CONN_PRUNE] Iter {iteration:>2}: removed {len(to_remove_indices):,} connections, "
                f"length = {current_length:,.2f}, "
                f"ratio = {current_length/real_length:.4f}, "
                f"remaining = {remaining_conns:,}/{total_conns:,} ({100-removed_pct:.1f}%)"
            )

        # If we've gone below the target, restore some connections
        if current_length < real_length:
            if not silent:
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

            if not silent:
                logger.info(
                    f"[CONN_PRUNE] Randomly restored {connections_restored} connections: "
                    f"length = {current_length:,.2f}, "
                    f"ratio = {current_length/real_length:.4f}, "
                    f"remaining = {remaining_conns:,}/{total_conns:,} ({100-removed_pct:.1f}%)"
                )

    # Create final pruned connections dataframe
    pruned_conns = conns[keep_mask].copy()

    if not silent:
        logger.info(
            f"[CONN_PRUNE] FINAL: length = {current_length:,.2f}, "
            f"ratio = {current_length/real_length:.4f}, "
            f"connections = {len(pruned_conns):,}/{total_conns:,} ({len(pruned_conns)/total_conns*100:.1f}%)"
        )

    result = pruned_conns[["pre_root_id", "post_root_id", "syn_count"]]

    logger.disabled = prev_disabled
    return result