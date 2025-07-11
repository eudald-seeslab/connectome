from utils.helpers import get_logger
from utils.randomizers.randomizers_helpers import compute_total_synapse_length


import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.randomizers.numba_helpers import euclidean_rows

# Get logger for this module
logger = get_logger(__name__)


def create_length_preserving_random_network(
    connections,
    neurons,
    bins: int = 100,
    tolerance: float = 0.1,
    *,
    silent: bool = False,
):
    """
    Create a randomized network that preserves the total wiring length.
    Process bin by bin to reduce memory usage.
    """
    connections = connections.copy()
    neurons = neurons.copy()

    # ------------------------------------------------------------------
    # Optional silence – temporarily disable this module's logger so that
    # benchmark runs remain clean.  We restore the previous *disabled*
    # state on exit so other calls are unaffected.
    # ------------------------------------------------------------------
    prev_disabled = logger.disabled
    logger.disabled = silent or prev_disabled

    if not silent:
        logger.info("Starting length-preserving network randomization...")

    # Convert to NumPy arrays
    pre_ids  = connections["pre_root_id"].to_numpy(dtype=np.int64)
    post_ids = connections["post_root_id"].to_numpy(dtype=np.int64)
    syn_cnt  = connections["syn_count"].to_numpy(dtype=np.int32)

    roots   = neurons["root_id"].to_numpy(dtype=np.int64)
    coords  = neurons[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float32)

    order   = np.argsort(roots)
    roots_s = roots[order]
    coords_s = coords[order]

    idx_pre  = np.searchsorted(roots_s, pre_ids)
    idx_post = np.searchsorted(roots_s, post_ids)

    pre_xyz  = coords_s[idx_pre]
    post_xyz = coords_s[idx_post]

    if not silent:
        logger.info("Computing distances …")

    dist_vec = euclidean_rows(pre_xyz, post_xyz)

    # ------------------------------------------------------------------
    # Quantile bin edges via NumPy (avoids pandas qcut)
    # ------------------------------------------------------------------
    if not silent:
        logger.info(f"Creating {bins} quantile bins …")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(dist_vec, quantiles)
    edges = np.unique(edges)  # drop duplicates

    if edges.size < 2:
        raise ValueError("Distance distribution degenerate – cannot bin.")

    n_bins = edges.size - 1

    bin_ids = np.searchsorted(edges, dist_vec, side="right") - 1
    bin_ids[bin_ids == n_bins] = n_bins - 1  # rightmost edge fix

    # ------------------------------------------------------------------
    # Shuffle connections **within** each distance bin.
    # Old implementation built a boolean mask for every bin which is
    # O(N × bins).  We now sort by *bin_id* once, shuffle contiguous
    # slices, then restore the original order – equivalent statistically
    # but only O(N log N + N).
    # ------------------------------------------------------------------

    # Stable sort so that the relative order of rows belonging to the same
    # bin is kept before shuffling – this has no statistical effect but
    # avoids an extra permutation when we write back to the original order.
    order_bins = np.argsort(bin_ids, kind="stable")

    pre_sorted     = pre_ids[order_bins]
    post_sorted    = post_ids[order_bins]
    bin_ids_sorted = bin_ids[order_bins]

    # Boundaries (start, end) of each bin slice in the *sorted* arrays
    boundaries = np.flatnonzero(np.diff(bin_ids_sorted)) + 1
    starts     = np.concatenate(([0], boundaries))
    ends       = np.concatenate((boundaries, [len(bin_ids_sorted)]))

    rng = np.random.default_rng()  # local RNG; preserves global state semantics

    for s, e in zip(starts, ends):
        n = e - s
        if n <= 1:
            continue  # nothing to shuffle in bins of size 0/1
        perm = rng.permutation(n)
        # Apply the *same* permutation to pre and post so syn_count rows stay aligned
        pre_sorted[s:e]  = pre_sorted[s:e][perm]
        post_sorted[s:e] = post_sorted[s:e][perm]

    # Undo the bin sort: place shuffled values back into original positions
    shuffled_pre          = np.empty_like(pre_ids)
    shuffled_post         = np.empty_like(post_ids)
    shuffled_pre[order_bins]  = pre_sorted
    shuffled_post[order_bins] = post_sorted

    final_connections = pd.DataFrame(
        {
            "pre_root_id": shuffled_pre,
            "post_root_id": shuffled_post,
            "syn_count": syn_cnt,
        }
    )

    # ------------------------------------------------------------------
    # Validate wiring length – reuse *dist_vec* for the original distances
    # and compute the new ones with NumPy only (no pandas merges).
    # ------------------------------------------------------------------

    if not silent:
        logger.info("Validating total wiring length…")

    # Map *post_root_id* → coordinates for the **shuffled** posts
    idx_post_new = np.searchsorted(roots_s, shuffled_post)
    post_xyz_new = coords_s[idx_post_new]
    dist_vec_new = euclidean_rows(pre_xyz, post_xyz_new)

    real_length  = float(np.sum(dist_vec * syn_cnt, dtype=np.float64))
    final_length = float(np.sum(dist_vec_new * syn_cnt, dtype=np.float64))
    dif_ratio    = abs(final_length - real_length) / real_length

    if not silent:
        logger.info(f"Original wiring length:   {real_length:.2f}")
        logger.info(f"Randomised wiring length: {final_length:.2f}")
        logger.info(
            f"Difference ratio: {dif_ratio:.4f} (should be < {tolerance})"
        )

    assert (
        dif_ratio < 1 + tolerance
    ), f"Final length differs by {dif_ratio:.4f}, exceeding tolerance of {tolerance}"

    # Restore logger state before returning
    logger.disabled = prev_disabled

    return final_connections