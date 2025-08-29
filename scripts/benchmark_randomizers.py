import argparse
import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from utils.randomizers import (
    binned,
    connection_pruning,
    mantain_neuron_wiring_length as mnwl,
    pruning,
)
from utils.randomizers.randomizers_helpers import compute_total_synapse_length

RandomizerFn = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


# -----------------------------------------------------------------------------
# Synthetic-data generator
# -----------------------------------------------------------------------------

def generate_synthetic_network(
    n_neurons: int,
    n_connections: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (connections, neuron_coords) dataframes for benchmarking."""
    rng = np.random.default_rng(seed)

    # Neuron coordinates in a unit cube → ensures a wide range of distances
    coords = pd.DataFrame(
        {
            "root_id": np.arange(n_neurons, dtype="int64"),
            "pos_x": rng.uniform(0, 1, size=n_neurons).astype("float32"),
            "pos_y": rng.uniform(0, 1, size=n_neurons).astype("float32"),
            "pos_z": rng.uniform(0, 1, size=n_neurons).astype("float32"),
        }
    )

    # Random pairs with replacement (allow self-loops; inconsequential here)
    pre = rng.integers(0, n_neurons, size=n_connections, dtype="int64")
    post = rng.integers(0, n_neurons, size=n_connections, dtype="int64")
    syn_count = rng.integers(1, 8, size=n_connections, dtype="int32")

    conns = pd.DataFrame(
        {
            "pre_root_id": pre,
            "post_root_id": post,
            "syn_count": syn_count,
        }
    )
    return conns, coords


# -----------------------------------------------------------------------------
# Benchmark helpers
# -----------------------------------------------------------------------------

def _bench(fn: RandomizerFn, conns: pd.DataFrame, coords: pd.DataFrame) -> float:
    t0 = time.perf_counter()
    fn(conns, coords)
    return time.perf_counter() - t0


def _wrap_randomizers(target_ratio: float = 0.8) -> Dict[str, RandomizerFn]:
    """Return dict of name→callable with bound parameters for benchmarking."""

    def _binned(c, n):
        return binned.create_length_preserving_random_network(c, n, bins=50, silent=True)

    def _mnwl(c, n):
        return mnwl.mantain_neuron_wiring_length(c, n, bins=20, min_connections_for_binning=10, silent=True)

    def _pruning(c, n):
        real_len = compute_total_synapse_length(c, n)
        target_len = target_ratio * real_len
        return pruning.match_wiring_length_with_random_pruning(
            c, n, real_length=target_len, tolerance=0.05, max_iter=10, silent=True
        )

    def _conn_prune(c, n):
        real_len = compute_total_synapse_length(c, n)
        target_len = target_ratio * real_len
        return connection_pruning.match_wiring_length_with_connection_pruning(
            c, n, real_length=target_len, tolerance=0.05, max_iter=50, silent=True
        )

    return {
        "binned": _binned,
        "mantain_neuron_wiring_length": _mnwl,
        "random_pruning": _pruning,
        "connection_pruning": _conn_prune,
    }


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark randomizer utilities")
    parser.add_argument("--n-neurons", type=int, default=10_000, help="Number of neurons in synthetic graph")
    parser.add_argument("--n-conns", type=int, default=200_000, help="Number of connections (rows)")
    parser.add_argument("--iters", type=int, default=3, help="Iterations per randomizer")
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated list of methods to benchmark (all, binned, mantain_neuron_wiring_length, random_pruning, connection_pruning)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-ratio", type=float, default=0.8, help="Target wiring length ratio for pruning methods")

    args = parser.parse_args()

    conns, coords = generate_synthetic_network(args.n_neurons, args.n_conns, args.seed)

    methods = _wrap_randomizers(args.target_ratio)

    selected: List[str]
    if args.methods == "all":
        selected = list(methods.keys())
    else:
        selected = [m.strip() for m in args.methods.split(",") if m.strip() in methods]
        if not selected:
            raise ValueError("No valid methods selected.")

    print(f"Benchmarking on synthetic network: {args.n_neurons:,} neurons, {args.n_conns:,} connections")
    print(f"Running {args.iters} iterations per method…\n")

    for name in selected:
        fn = methods[name]
        times = [_bench(fn, conns, coords) for _ in range(args.iters)]
        print(
            f"{name:>28}: mean {np.mean(times):.3f} s  (std {np.std(times):.3f}, min {np.min(times):.3f}, max {np.max(times):.3f})"
        )


if __name__ == "__main__":
    main() 