import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.helpers import add_coords, get_logger, add_distance_column

# Get logger for this module
logger = get_logger(__name__)


def mantain_neuron_wiring_length(
    connections: pd.DataFrame,
    nc: pd.DataFrame,
    bins: int = 20,
    min_connections_for_binning: int = 10,
    random_state: int | None = None,
    tolerance: float = 0.01,
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
    import time
    import gc
    try:
        import psutil
        monitor_memory = True
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        logger.info(f"Initial memory usage: {initial_memory:.2f} GB")
    except ImportError:
        monitor_memory = False
        logger.info("psutil not available - memory monitoring disabled")
    
    start_time = time.time()
    
    np.random.seed(random_state)
    logger.info("Starting vectorized per-neuron length-preserving randomization...")
    logger.info(f"Processing {len(connections):,} connections for {connections['pre_root_id'].nunique():,} neurons")
    
    # Calculate distances for all connections (vectorized) using helper
    step_start = time.time()
    connections_with_coords = add_distance_column(connections, nc, distance_col="distance")
    distances = connections_with_coords["distance"]  # view only; avoids extra memory
    logger.info(
        f"Distance calculation + coordinate attachment completed in {time.time() - step_start:.2f}s"
    )
    if monitor_memory:
        current_memory = process.memory_info().rss / 1024 / 1024 / 1024
        logger.info(f"Memory usage after distance calculation: {current_memory:.2f} GB")
    
    # Count connections per neuron (vectorized)
    connection_counts = connections_with_coords.groupby('pre_root_id').size()
    
    # Identify neurons with enough connections for binning
    neurons_for_binning = connection_counts[connection_counts >= min_connections_for_binning].index
    neurons_for_simple = connection_counts[connection_counts < min_connections_for_binning].index
    
    logger.info(f"Neurons using binning: {len(neurons_for_binning)}")
    logger.info(f"Neurons using simple shuffle: {len(neurons_for_simple)}")
    
    # ------------------------------------------------------------------
    # Vectorized distance-bin creation (replaces slow Python for-loops)
    # ------------------------------------------------------------------
    logger.info(
        f"Creating distance bins for {len(neurons_for_binning):,} neurons using vectorised ranking…"
    )
    step_start = time.time()

    # Initialise distance_bin to 0 (neurons with too few connections keep this)
    connections_with_coords["distance_bin"] = 0

    if len(neurons_for_binning):
        # Compute the percentile rank of each distance **within each neuron**
        mask_binned = connections_with_coords["pre_root_id"].isin(neurons_for_binning)
        # rank(pct=True) gives values in (0,1]; multiply by bins and cast to int
        pct_rank = connections_with_coords.loc[mask_binned].groupby("pre_root_id")[
            "distance"
        ].rank(method="first", pct=True)
        distance_bin = np.minimum((pct_rank * bins).astype(int), bins - 1)
        # Assign back
        connections_with_coords.loc[mask_binned, "distance_bin"] = distance_bin.values

    logger.info(f"Binning completed in {time.time() - step_start:.2f}s")
    # ------------------------------------------------------------------
    
    # Force garbage collection after memory-intensive binning operation
    gc.collect()
    if monitor_memory:
        current_memory = process.memory_info().rss / 1024 / 1024 / 1024
        logger.info(f"Memory usage after binning: {current_memory:.2f} GB")
    
    # Split into binned and simple shuffle groups
    connections_binned = connections_with_coords[
        connections_with_coords['pre_root_id'].isin(neurons_for_binning)
    ]
    
    # For neurons with few connections, they already have distance_bin = 0
    connections_simple = connections_with_coords[
        connections_with_coords['pre_root_id'].isin(neurons_for_simple)
    ]
    
    # All connections now have distance_bin assigned
    all_connections = connections_with_coords
    # Note: all_connections is just a reference to connections_with_coords
    # We'll delete all_connections later to free the memory
    
    logger.info("Shuffling post_root_ids within (neuron, distance_bin) groups…")
    
    step_start = time.time()
    # Keep only required columns for shuffling to save memory
    shuffle_df = connections_with_coords[["pre_root_id", "distance_bin", "post_root_id", "syn_count"]].copy()

    # Build the groupby object only once so we can access the number of groups
    groups = shuffle_df.groupby(["pre_root_id", "distance_bin"])["post_root_id"]    

    pbar = tqdm(total=groups.ngroups, desc="Shuffling groups", disable=False)

    def _permute_with_progress(series):
        """Helper that shuffles a series and updates the progress-bar."""
        pbar.update(1)
        return np.random.permutation(series.values)

    shuffle_df["post_root_id"] = groups.transform(_permute_with_progress)
    pbar.close()

    logger.info(f"Shuffling completed in {time.time() - step_start:.2f}s")
    
    # ------------------------------------------------------------------
    # Prepare return value & clean-up
    # ------------------------------------------------------------------
    final_result = shuffle_df[["pre_root_id", "post_root_id", "syn_count"]].copy()
    del shuffle_df, connections_with_coords  # free memory
    gc.collect()
    
    logger.info("Validating wiring length preservation...")
    
    # Quick validation of total wiring length preservation
    orig_total_wiring = np.sum(distances * connections["syn_count"])
    
    # Calculate final distances and wiring
    validation_start = time.time()
    final_coords = add_coords(final_result, nc)
    final_distances = np.sqrt(
        (final_coords["pre_x"] - final_coords["post_x"])**2 +
        (final_coords["pre_y"] - final_coords["post_y"])**2 +
        (final_coords["pre_z"] - final_coords["post_z"])**2
    )
    final_total_wiring = np.sum(final_distances * final_result["syn_count"])
    
    wiring_error = abs(final_total_wiring - orig_total_wiring) / orig_total_wiring
    
    total_time = time.time() - start_time
    
    logger.info(f"Validation completed in {time.time() - validation_start:.2f}s")
    logger.info(f"Total processing time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    logger.info(f"Total wiring length preservation error: {wiring_error:.8f}")
    logger.info(f"Original total wiring: {orig_total_wiring:.2f}")
    logger.info(f"Final total wiring: {final_total_wiring:.2f}")
    
    if wiring_error > 1e-10:
        logger.warning(f"Wiring length error {wiring_error:.2e} is larger than expected for binned approach")
    
    # Detailed per-neuron validation (optional for large networks to avoid additional memory usage)
    if len(connections) < 5000000:  # Only for networks with < 5M connections
        logger.info("Running detailed per-neuron validation...")
        detailed_validation = validate_per_neuron_outgoing_wiring_preservation(
            connections, final_result, nc, tolerance=tolerance
        )
        logger.info(f"Detailed validation results: {detailed_validation}")
    else:
        logger.info("Skipping detailed per-neuron validation for large network (>5M connections)")
    
    return final_result


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