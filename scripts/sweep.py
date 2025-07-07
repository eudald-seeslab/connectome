import argparse
import wandb
import multiprocessing
import pandas as pd
import os
import hashlib, json

from configs import config as base_config
from scripts.train import main
from connectome.tools.wandb_logger import WandBLogger
from utils.randomization_generator import generate_random_connectome
from connectome.core.utils import update_config_with_sweep


project_name = base_config.wandb_project

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

def _load_cell_type_lists():
    """Return *cell_types* and *rational_cell_types* lists, using whatever
    files are available. If the expected CSVs are missing, return empty lists
    so that sweeps that do not depend on them can still run."""

    adult_dir = "adult_data"
    ct_path = os.path.join(adult_dir, "cell_types.csv")
    rat_path = os.path.join(adult_dir, "rational_cell_types.csv")

    if not os.path.exists(ct_path) or not os.path.exists(rat_path):
        return [], []

    cts_df = pd.read_csv(ct_path)
    cts_df = cts_df[cts_df["count"] > 1000]
    rational = pd.read_csv(rat_path, index_col=0).index.tolist()
    forbidden = rational + ["R8", "R7", "R1-6"]
    cell_types = [x for x in cts_df["cell_type"].values if x not in forbidden]
    return cell_types, rational


# -------------------------------------------------------------
# Cell-type dependent sweep config (only built if data available)
# -------------------------------------------------------------
_cell_types, _rational_cell_types = _load_cell_type_lists()

if _cell_types:
    sweep_config2 = {
        "method": "bayes",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "filtered_celltypes": {"values": _cell_types},
        },
    }
else:
    sweep_config2 = None  # data not available; skip this sweep

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

seeds = list(range(10))
sweep_config_seeds = {
    "method": "grid",
    "parameters": {
        "random_seed": {"values": seeds},
    },
}


def train(sweep_cfg=None):
    wandb_logger = WandBLogger(project_name)
    with wandb.init(config=sweep_cfg):
        # Merge sweep hyper-parameters into the global config so the generator
        # knows about the seed/strategy values.
        u_config = update_config_with_sweep(base_config, wandb.config)

        # Create the randomised dataset (if requested) before training.
        generate_random_connectome(u_config)

        connections = pd.read_csv(f"new_data/connections_random_{u_config.randomization_strategy}.csv")
        checksum = hashlib.md5(connections.to_json().encode()).hexdigest()
        wandb.log({"connectome_md5": checksum})

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
    parser.add_argument(
        "--seeds_sweep",
        action="store_true",
        help="Run sweep that iterates over random seeds and uses on-the-fly randomised connectomes.",
    )
    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        if args.seeds_sweep:
            sweep_id = wandb.sweep(sweep_config_seeds, project=project_name)
        else:
            sweep_id = wandb.sweep(sweep_config5, project=project_name)

    run_agent(sweep_id)
