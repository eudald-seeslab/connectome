# %%
import argparse
import itertools
from complete_training import main
from wandb_logger import WandBLogger
import numpy as np


project_name = "Complete_v2"

params = {
    "NUM_CONNECTOME_PASSES": [3, 4, 5, 6],
    "base_lr": np.linspace(1e-5, 1e-2, num=5).tolist(),
    "neurons": ["selected", "all"],
    "voronoi_criteria": ["R7", "all"],
    "random_synapses": [True, False],
}

param_names = sorted(params.keys())
combinations = list(itertools.product(*(params[name] for name in param_names)))

# %%
class SweepConfig:
    def __init__(
        self, neurons, voronoi_criteria, random_synapses, base_lr, NUM_CONNECTOME_PASSES
    ):
        self.neurons = neurons
        self.voronoi_criteria = voronoi_criteria
        self.random_synapses = random_synapses
        self.base_lr = base_lr
        self.NUM_CONNECTOME_PASSES = NUM_CONNECTOME_PASSES


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process sweep parameters.")
    parser.add_argument("NUM_CONNECTOME_PASSES", type=int, help="Number of connectome passes")
    parser.add_argument("base_lr", type=float, help="Base learning rate")
    parser.add_argument("neurons", type=str, help="Type of neurons (selected or all)")
    parser.add_argument("random_synapses", type=str, help="Use random synapses (True or False)")
    parser.add_argument("voronoi_criteria", type=str, help="Voronoi criteria (R7 or all)")

    # Parse the arguments
    args = parser.parse_args()

    # Create a SweepConfig object from the parsed arguments
    sweep_config = SweepConfig(
        neurons=args.neurons,
        voronoi_criteria=args.voronoi_criteria,
        random_synapses=args.random_synapses == "True",
        base_lr=args.base_lr,
        NUM_CONNECTOME_PASSES=args.NUM_CONNECTOME_PASSES
    )
    print("Running with configuration:")
    print(f"Neurons: {sweep_config.neurons}")
    print(f"Voronoi Criteria: {sweep_config.voronoi_criteria}")
    print(f"Random Synapses: {sweep_config.random_synapses}")
    print(f"Base Learning Rate: {sweep_config.base_lr}")
    print(f"Number of Connectome Passes: {sweep_config.NUM_CONNECTOME_PASSES}")

    wandb_logger = WandBLogger("adult_complete")
    wandb_logger.initialize_run(group="cluster_sweep")

    
    main(wandb_logger, sweep_config=sweep_config)
