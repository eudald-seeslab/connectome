from pathlib import Path

from joblib import Memory

from utils.helpers import load_neuron_coordinates
from notebooks.visualization.activations_funcs import (
    get_all_connections,
    get_activation_dictionnary,
)

# ---------------------------------------------------------------------------
# Joblib cache setup
# ---------------------------------------------------------------------------

# Cache directory lives next to this file: notebooks/.joblib_cache
cache_dir = Path(__file__).resolve().parent / ".joblib_cache"
memory = Memory(location=cache_dir, verbose=0)


@memory.cache
def connections_cached():
    """Disk-cached version of get_all_connections()."""
    return get_all_connections()


@memory.cache
def activations_cached(num_passes: int = 4):
    """Disk-cached wrapper around get_activation_dictionnary()."""
    conn_dict = connections_cached()
    return get_activation_dictionnary(conn_dict, num_passes)


def main(num_passes: int = 4):
    """Entry point for quick testing/CLI usage."""
    # Lightweight data
    neuron_pos = load_neuron_coordinates()

    # Heavy cached computations
    conn_dict = connections_cached()
    activations_dict = activations_cached(num_passes)

    # Example: report shapes
    for name, df in activations_dict.items():
        print(f"{name}: {df.shape}")


if __name__ == "__main__":
    main()