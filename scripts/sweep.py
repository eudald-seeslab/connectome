import argparse
import wandb
import multiprocessing
import pandas as pd

from configs import config
from scripts.train import main
from connectome.tools.wandb_logger import WandBLogger


project_name = "architecture"

sweep_config1 = {
    "method": "bayes",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "NUM_CONNECTOME_PASSES": {"values": [3, 4, 5, 6]},
        "neurons": {"values": ["selected", "all"]},
        "voronoi_criteria": {"values": ["R7", "all"]},
        "random_synapses": {"values": [True, False]},
        "eye": {"values": ["left", "right"]},
        "train_edges": {"values": [True, False]},
        "train_neurons": {"values": [True, False]},
        "final_layer": {"values": ["mean", "nn"]},
    },
}

cts = pd.read_csv("adult_data/cell_types.csv")
cts = cts[cts["count"] > 1000]
rational_cell_types = pd.read_csv("adult_data/rational_cell_types.csv", index_col=0).index.tolist()
forbidden_cell_types = rational_cell_types + ["R8", "R7", "R1-6"]
cell_types = [x for x in cts["cell_type"].values if x not in forbidden_cell_types]

sweep_config2 = {
    "method": "bayes",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "filtered_celltypes": {"values": cell_types},
    }
}

sweep_config3 = {
    "method": "random",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "filtered_fraction": {"values": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    },
}

sweep_config4 = {
    "method": "random",
    "metric": {"name": "Validation accuracy", "goal": "maximize"},
    "parameters": {
        "neuron_dropout": {"values": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        "decision_dropout": {"values": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    },
}

sweep_config5 = {
    "method": "random",
    "metric": {"name": "Validation accuracy", "goal": "maximize"},
    "parameters": {
        "train_neurons": {"values": [True, False]},
        "train_edges": {"values": [True, False]},
        "refined_synaptic_data": {"values": [True, False]},
        "final_layer": {"values": ["mean", "nn"]},
    },
}


def train(config=None):
    wandb_logger = WandBLogger(project_name)
    with wandb.init(config=config):
        main(wandb_logger, wandb.config)


def run_agent(sweep_id):
    wandb.agent(sweep_id=sweep_id, function=train, project=project_name, count=60)


if __name__ == "__main__":
    # argparse sweep id
    parser = argparse.ArgumentParser(
        description="Run a sweep."
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Sweep id if you have started the sweep elsewhere.",
    )
    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep_config5, project=project_name)

    if config.device_type == "cpu":
        num_agents = 4
        processes = []
        for _ in range(num_agents):
            p = multiprocessing.Process(target=run_agent, args=(sweep_id,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        run_agent(sweep_id)
