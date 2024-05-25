import wandb
import multiprocessing
import config
from complete_training import main
from wandb_logger import WandBLogger


project_name = "Complete_v2"

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

def train(config=None):
    wandb_logger = WandBLogger(project_name)
    with wandb.init(config=config):
        main(wandb_logger, wandb.config)


def run_agent(sweep_id):
    wandb.agent(sweep_id=sweep_id, function=train, project=project_name, count=20)


sweep_id = wandb.sweep(sweep_config, project=project_name)


if __name__ == "__main__":
    num_agents = 4 if config.device_type == "cpu" else 1
    processes = []

    for _ in range(num_agents):
        p = multiprocessing.Process(target=run_agent, args=(sweep_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

wandb.agent(sweep_id=sweep_id, function=train, project=project_name, count=20)
