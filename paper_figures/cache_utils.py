"""Stable joblib caches for notebook visualization data."""

from pathlib import Path

from joblib import Memory

from notebooks.visualization.activations_funcs import (
    get_activation_dictionnary,
    get_all_connections,
)


_CACHE_DIR = Path(__file__).resolve().parent / "data" / ".joblib_cache"
_MEMORY = Memory(location=_CACHE_DIR, verbose=0)


@_MEMORY.cache
def connections_cached():
    """Return cached connection variants."""
    return get_all_connections()


@_MEMORY.cache
def activations_cached(num_passes: int = 4):
    """Return cached activation propagation results."""
    conn_dict = connections_cached()
    return get_activation_dictionnary(conn_dict, num_passes)
