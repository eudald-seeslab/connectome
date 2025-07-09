# Standard library
import logging
import os
import argparse

# Third-party

# Local modules
from notebooks.visualization.activation_plots import plot_synapse_length_distributions
from paths import PROJECT_ROOT
from utils.helpers import (
    load_connections,
    load_neuron_coordinates,
    setup_logging,
    get_logger,
)
from utils.randomizers.binned import create_length_preserving_random_network
from utils.randomizers.connection_pruning import match_wiring_length_with_connection_pruning
from utils.randomizers.pruning import match_wiring_length_with_random_pruning
from utils.randomizers.mantain_neuron_wiring_length import mantain_neuron_wiring_length
from utils.randomizers.randomizers_helpers import compute_total_synapse_length, shuffle_post_root_id

# Configure logging once for the entire application
setup_logging(level=logging.INFO)

# Module-specific logger
logger = get_logger(__name__)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate randomized network connections')
    parser.add_argument('-t', '--tolerance', type=float, default=0.01,
                      help='Tolerance for randomization')
    parser.add_argument('--unconstrained', action='store_true',
                      help='Generate unconstrained randomized network')
    parser.add_argument('--pruned', action='store_true',
                      help='Generate pruned randomized network')
    parser.add_argument('--conn_pruned', action='store_true',
                      help='Generate connection-pruned randomized network (removes entire connections)')
    parser.add_argument('--binned', action='store_true',
                      help='Generate binned randomized network')
    parser.add_argument('--mantain_neuron_wiring_length', action='store_true',
                      help='Generate mantain neuron wiring length randomized network')
    parser.add_argument("--plot_results", action="store_true",
                      help="Plot the results")
    args = parser.parse_args()
    
    # If no arguments provided, run all randomizations
    run_all = not (args.unconstrained or args.pruned or args.conn_pruned or args.binned or args.mantain_neuron_wiring_length)
    
    # Load data
    logger.info("Loading data...")
    connections = load_connections()
    neuron_coordinates = load_neuron_coordinates()
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
            tolerance=args.tolerance,
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
            tolerance=args.tolerance,
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
            connections, neuron_coordinates, bins=100, tolerance=args.tolerance
        )
        random_binned.to_csv(
            os.path.join(PROJECT_ROOT, "new_data", "connections_random_binned.csv"),
            index=False,
        )
        logger.info("Length-preserving randomization completed")
    else:
        random_binned = None

    # Mantain neuron wiring length
    if run_all or args.mantain_neuron_wiring_length:
        logger.info("Starting mantain neuron wiring length randomization...")
        random_mantain_neuron_wiring_length = mantain_neuron_wiring_length(
            connections,
            neuron_coordinates,
            bins=20,
            min_connections_for_binning=10,
            random_state=1234,
            tolerance=args.tolerance,
        )   
        random_mantain_neuron_wiring_length.to_csv(
            os.path.join(PROJECT_ROOT, "new_data", "connections_random_mantain_neuron_wiring_length.csv"),
            index=False,
        )
        logger.info("Mantain neuron wiring length completed")
    else:
        random_mantain_neuron_wiring_length = None

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
        if args.mantain_neuron_wiring_length or run_all:
            conns_to_plot["Random mantain neuron wiring length"] = random_mantain_neuron_wiring_length

        fig1, fig2 = plot_synapse_length_distributions(neuron_coordinates, conns_to_plot, use_density=False)
        plots_path = os.path.join(PROJECT_ROOT, "utils", "plots")
        os.makedirs(plots_path, exist_ok=True)
        fig1.savefig(os.path.join(plots_path, "synapse_length_distributions.png"), dpi=300)
        fig2.savefig(os.path.join(plots_path, "synapse_length_distributions_density.png"), dpi=300)
