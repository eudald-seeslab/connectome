import time
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.helpers import get_logger
from utils.randomizers.randomizers_helpers import add_coords, add_distance_column

# Get logger for this module
logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Optional Numba support
# -----------------------------------------------------------------------------

try:
    from numba import njit, prange  # type: ignore

    _HAS_NUMBA = True

    @njit(fastmath=True, nogil=True)
    def _numba_shuffle_post_ids(
        post_sorted: np.ndarray,
        dist_sorted: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        bins: int,
        min_conn: int,
    ) -> None:
        """In-place bin-aware shuffling compiled with Numba (single-thread)."""

        for idx in range(starts.size):
            s = starts[idx]
            e = ends[idx]
            n = e - s
            if n < 2:
                continue

            # Few connections → plain Fisher-Yates shuffle
            if n < min_conn:
                for j in range(n - 1, 0, -1):
                    k = np.random.randint(j + 1)
                    tmp = post_sorted[s + j]
                    post_sorted[s + j] = post_sorted[s + k]
                    post_sorted[s + k] = tmp
                continue

            # Rank distances, compute bin ids
            # Numba lacks argsort that returns float32, so cast to float64
            order_local = np.argsort(dist_sorted[s:e].astype(np.float64))
            ranks = np.empty(n, dtype=np.int64)
            for r in range(n):
                ranks[order_local[r]] = r
            bin_ids = (ranks * bins) // n

            # Shuffle within each bin
            for b in range(bins):
                # Collect indices of this bin into a temporary list
                m = 0
                for j in range(n):
                    if bin_ids[j] == b:
                        m += 1
                if m < 2:
                    continue
                idx_buf = np.empty(m, dtype=np.int64)
                k = 0
                for j in range(n):
                    if bin_ids[j] == b:
                        idx_buf[k] = j
                        k += 1

                # Fisher-Yates shuffle restricted to idx_buf
                for j in range(m - 1, 0, -1):
                    p = np.random.randint(j + 1)
                    a = idx_buf[j]
                    bidx = idx_buf[p]
                    tmp = post_sorted[s + a]
                    post_sorted[s + a] = post_sorted[s + bidx]
                    post_sorted[s + bidx] = tmp

except ImportError:  # pragma: no cover – numba optional
    _HAS_NUMBA = False


def mantain_neuron_wiring_length(
    connections: pd.DataFrame,
    nc: pd.DataFrame,
    bins: int = 20,
    min_connections_for_binning: int = 10,
    random_state: int | None = None,
    *,
    silent: bool = False,
    use_numba: bool = True,
) -> pd.DataFrame:
    """
    Create a randomized network that preserves per-neuron outgoing wiring length distributions.
    
    This follows the same approach as create_length_preserving_random_network but applies
    it per neuron instead of globally. For each neuron:
    1. Calculates distances for all its outgoing connections
    2. Bins these connections by distance 
    3. Shuffles post_root_ids within each distance bin for that neuron
    
    This preserves each neuron's outgoing wiring length distribution while randomizing 
    the specific targets. Since each neuron's outgoing wiring is preserved, the total
    wiring length is automatically preserved as well.
    
    Fully vectorized implementation using pandas groupby operations with progress tracking.
    
    Parameters:
    -----------
    connections : DataFrame
        Contains pre_root_id, post_root_id, and syn_count columns
    nc : DataFrame
        Contains root_id and position coordinates for each neuron
    bins : int
        Number of distance bins to use for each neuron (default: 20)
    min_connections_for_binning : int
        Minimum outgoing connections a neuron needs to apply binning (default: 10)
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    DataFrame with shuffled connections maintaining per-neuron outgoing wiring distributions
    """

    # ------------------------------------------------------------------
    # Pure NumPy implementation – no pandas inside the heavy loop.
    # ------------------------------------------------------------------

    rng = np.random.default_rng(random_state)

    prev_disabled = logger.disabled
    logger.disabled = silent or prev_disabled

    if not silent:
        logger.info("Starting NumPy per-neuron length-preserving randomization …")

    t0_total = time.perf_counter()

    # --- 1. Prepare NumPy arrays ------------------------------------------------
    pre_ids  = connections["pre_root_id"].to_numpy(dtype=np.int64)
    post_ids = connections["post_root_id"].to_numpy(dtype=np.int64)
    syn_cnt  = connections["syn_count"].to_numpy(dtype=np.int32)

    roots    = nc["root_id"].to_numpy(dtype=np.int64)
    coords   = nc[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float32)

    # Map root_id → index
    order          = np.argsort(roots)
    roots_sorted   = roots[order]
    coords_sorted  = coords[order]

    idx_pre  = np.searchsorted(roots_sorted, pre_ids)
    idx_post = np.searchsorted(roots_sorted, post_ids)

    pre_xyz  = coords_sorted[idx_pre]
    post_xyz = coords_sorted[idx_post]
    dist_vec = np.linalg.norm(pre_xyz - post_xyz, axis=1)

    if not silent:
        logger.info("Distances computed (NumPy) – %.2f s", time.perf_counter() - t0_total)

    # --- 2. Group by pre-synaptic neuron ---------------------------------------
    # Sort by *inv_pre* so that all connections of a neuron form one contiguous
    # slice – avoids repeatedly building boolean masks.

    inv_pre = np.zeros_like(pre_ids, dtype=np.int32)
    unique_pre, inv_pre = np.unique(pre_ids, return_inverse=True)

    order_idx = np.argsort(inv_pre, kind="mergesort")  # stable

    pre_sorted     = pre_ids[order_idx]
    post_sorted    = post_ids.copy()[order_idx]
    dist_sorted    = dist_vec[order_idx]

    # Boundaries of each neuron block in the sorted arrays
    boundaries = np.flatnonzero(np.diff(pre_sorted)) + 1
    starts     = np.concatenate(([0], boundaries))
    ends       = np.concatenate((boundaries, [len(pre_sorted)]))

    if _HAS_NUMBA and use_numba:
        if not silent:
            logger.info("Running Numba-accelerated shuffling …")

        # Call JIT kernel (in-place modification of post_sorted)
        _numba_shuffle_post_ids(post_sorted, dist_sorted, starts, ends, bins, min_connections_for_binning)

    else:
        # Python fallback – iterate slices
        for start, end in tqdm(zip(starts, ends), total=len(starts), disable=silent):
            size = end - start
            if size < 2:
                continue

            slice_post = post_sorted[start:end]

            if size < min_connections_for_binning:
                rng.shuffle(slice_post)
                continue

            order_local = np.argsort(dist_sorted[start:end])
            ranks       = np.empty_like(order_local)
            ranks[order_local] = np.arange(size)
            bin_ids = (ranks * bins) // size

            for b in range(bins):
                sel = np.nonzero(bin_ids == b)[0]
                if sel.size > 1:
                    rng.shuffle(slice_post[sel])

            # slice_post modifications apply to post_sorted in-place

    # Reconstruct final shuffled_post in original order
    shuffled_post = np.empty_like(post_ids)
    shuffled_post[order_idx] = post_sorted

    if not silent:
        logger.info("Shuffling done – %.2f s", time.perf_counter() - t0_total)

    # --- 3. Assemble result DataFrame ------------------------------------------
    res_df = pd.DataFrame({
        "pre_root_id": pre_sorted,
        "post_root_id": shuffled_post,
        "syn_count": syn_cnt[order_idx],
    })

    logger.disabled = prev_disabled

    return res_df


def validate_per_neuron_outgoing_wiring_preservation(original_conns, shuffled_conns, nc, tolerance=0.05):
    """
    Validate that per-neuron outgoing wiring lengths are preserved within tolerance.
    
    Parameters:
    -----------
    original_conns : DataFrame
        Original connections
    shuffled_conns : DataFrame
        Shuffled connections to validate
    nc : DataFrame
        Neuron coordinates
    tolerance : float
        Acceptable relative error
        
    Returns:
    --------
    dict with validation statistics
    """
    
    def calculate_outgoing_wiring(conns, coords):
        conns_with_coords = add_coords(conns, coords)
        distances = np.sqrt(
            (conns_with_coords["pre_x"] - conns_with_coords["post_x"])**2 +
            (conns_with_coords["pre_y"] - conns_with_coords["post_y"])**2 +
            (conns_with_coords["pre_z"] - conns_with_coords["post_z"])**2
        )
        conns_with_coords["wiring_contrib"] = distances * conns_with_coords["syn_count"]
        outgoing = conns_with_coords.groupby("pre_root_id")["wiring_contrib"].sum()
        return outgoing
    
    orig_outgoing = calculate_outgoing_wiring(original_conns, nc)
    shuf_outgoing = calculate_outgoing_wiring(shuffled_conns, nc)
    
    # Align indices
    shuf_outgoing = shuf_outgoing.reindex(orig_outgoing.index, fill_value=0)
    
    # Calculate errors
    outgoing_errors = np.abs(shuf_outgoing / (orig_outgoing + 1e-10) - 1.0)
    outgoing_violations = (outgoing_errors > tolerance).sum()
    
    # Calculate total wiring length preservation
    orig_total = orig_outgoing.sum()
    shuf_total = shuf_outgoing.sum()
    total_wiring_error = abs(shuf_total - orig_total) / orig_total
    
    return {
        "outgoing_violations": outgoing_violations,
        "total_neurons_with_outgoing": len(orig_outgoing),
        "mean_outgoing_error": outgoing_errors.mean(),
        "max_outgoing_error": outgoing_errors.max(),
        "outgoing_within_tolerance": (outgoing_violations == 0),
        "total_wiring_error": total_wiring_error,
        "original_total_wiring": orig_total,
        "shuffled_total_wiring": shuf_total,
    }