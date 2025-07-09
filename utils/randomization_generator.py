import pandas as pd
import numpy as np
from pathlib import Path

from paths import PROJECT_ROOT
from utils.helpers import (
    load_neuron_coordinates,
)
from utils.randomizers.binned import create_length_preserving_random_network
from utils.randomizers.connection_pruning import match_wiring_length_with_connection_pruning
from utils.randomizers.pruning import match_wiring_length_with_random_pruning
from utils.randomizers.mantain_neuron_wiring_length import mantain_neuron_wiring_length
from utils.randomizers.randomizers_helpers import compute_total_synapse_length, shuffle_post_root_id


def generate_random_connectome(u_config):
    """Create a randomised version of the connectome according to
    *u_config.randomization_strategy* and *u_config.random_seed*.

    The file is saved to disk following the naming convention expected by
    DataProcessor._get_connections so that downstream code can load it
    transparently (e.g., ``connections_random_binned.csv`` in the appropriate
    data folder).
    """

    if u_config.randomization_strategy is None:
        # No randomisation requested → nothing to do
        return

    data_dir = "new_data" if u_config.new_connectome else "adult_data"
    base_file = (
        "connections_refined.csv" if u_config.refined_synaptic_data else "connections.csv"
    )
    base_path = Path(PROJECT_ROOT) / data_dir / base_file
    if not base_path.exists():
        raise FileNotFoundError(f"Base connectome not found: {base_path}")

    dtype_spec = {
        "pre_root_id": "string",
        "post_root_id": "string",
        "syn_count": "int32",
    }
    connections = (
        pd.read_csv(base_path, dtype=dtype_spec)
        .groupby(["pre_root_id", "post_root_id"], as_index=False)
        .sum("syn_count")
        .sort_values(["pre_root_id", "post_root_id"])
    )

    neuron_coords = load_neuron_coordinates(root_dir=PROJECT_ROOT)

    seed = u_config.random_seed
    strategy = u_config.randomization_strategy

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------
    if strategy == "unconstrained":
        randomized = shuffle_post_root_id(connections, random_state=seed)

    elif strategy == "pruned":
        total_len = compute_total_synapse_length(connections, neuron_coords)
        rand_unconstrained = shuffle_post_root_id(connections, random_state=seed)
        randomized = match_wiring_length_with_random_pruning(
            rand_unconstrained, neuron_coords, total_len, tolerance=0.01, allow_zeros=True
        )

    elif strategy == "conn_pruned":
        total_len = compute_total_synapse_length(connections, neuron_coords)
        rand_unconstrained = shuffle_post_root_id(connections, random_state=seed)
        randomized = match_wiring_length_with_connection_pruning(
            rand_unconstrained,
            neuron_coords,
            total_len,
            tolerance=0.01,
            max_iter=100,
            adaptive_batch=True,
            random_state=seed,
        )

    elif strategy == "binned":
        np.random.seed(seed)
        randomized = create_length_preserving_random_network(
            connections, neuron_coords, bins=100, tolerance=0.01
        )

    elif strategy == "neuron_binned":
        randomized = mantain_neuron_wiring_length(
            connections,
            neuron_coords,
            bins=20,
            min_connections_for_binning=10,
            random_state=seed,
            tolerance=0.01,
        )
    else:
        raise ValueError(f"Unknown randomization strategy '{strategy}'.")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    suffix = f"_random_{strategy}.csv"
    out_file = (
        f"connections_refined{suffix}" if u_config.refined_synaptic_data else f"connections{suffix}"
    )
    out_path = Path(PROJECT_ROOT) / data_dir / out_file
    randomized.to_csv(out_path, index=False) 