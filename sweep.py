import wandb
from complete_training import main

from wandb_logger import WandBLogger


sweep_config = {
    "method": "bayes",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "NUM_CONNECTOME_PASSES": {"values": [3, 4, 5, 6]},
        "base_lr": {"distribution": "uniform", "min": 1e-6, "max": 1e-3},
        "neurons": {"values": ["selected", "all"]},
        "voronoi_criteria": {"values": ["R7", "all"]},
        "random_synapses": {"values": [True, False]},
    },
}

project_name = "Complete_v1"
sweep_id = wandb.sweep(sweep_config, project=project_name)


def train(config=None):
    wandb_logger = WandBLogger(project_name)
    with wandb.init(config=config):
        main(wandb_logger, wandb.config)


wandb.agent(sweep_id, train, count=20)
