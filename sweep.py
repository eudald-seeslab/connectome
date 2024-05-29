import argparse
import wandb
import multiprocessing
import config
from complete_training import main
from wandb_logger import WandBLogger


project_name = "Complete_v3"

sweep_config = {
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
        sweep_id = wandb.sweep(sweep_config, project=project_name)

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
