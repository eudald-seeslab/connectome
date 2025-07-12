import argparse
import wandb
import pandas as pd
import os
import hashlib

from configs import config as base_config
from scripts.train import main
from connectome.tools.wandb_logger import WandBLogger
from utils.randomization_generator import generate_random_connectome
from connectome.core.utils import update_config_with_sweep

# New imports after refactor
from configs.sweep_definitions import SWEEP_DEFS
from scripts.sweep_utils import validate_sweep_config


project_name = base_config.wandb_project


def train(sweep_cfg=None):
    wandb_logger = WandBLogger(project_name)
    with wandb.init(config=sweep_cfg):
        # Merge sweep hyper-parameters into the global config so the generator
        # knows about the seed/strategy values.
        u_config = update_config_with_sweep(base_config, wandb.config)

        # Log full config early so that the run's Config panel is complete. We
        # only push serialisable primitives, so this won't interfere with the
        # training loop that consumes *wandb.config* later on.
        wandb_logger.update_full_config(u_config)

        # Create the randomised dataset (if requested) before training.
        generate_random_connectome(u_config)

        dataset_name = f"connections_random_{u_config.randomization_strategy}" if u_config.randomization_strategy is not None else "connections"
        connections = pd.read_csv(os.path.join("new_data", f"{dataset_name}.csv"))
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
    # Choose a sweep by name (defined in configs.sweep_definitions)
    parser.add_argument(
        "--sweep",
        default="regularisation",
        help=f"Sweep definition to use. Available: {', '.join(SWEEP_DEFS.keys())}",
    )

    # Allow the user to bypass pre-flight sanity checks (useful for quick local
    # experimentation).
    parser.add_argument(
        "--skip_checks",
        action="store_true",
        help="Skip pre-flight checks (debugging, small_length, etc.) and run the sweep anyway.",
    )
    args = parser.parse_args()

    # Pre-flight sanity checks
    validate_sweep_config(base_config, args.skip_checks)

    # Run / resume the sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_name = args.sweep

        try:
            sweep_cfg_dict = SWEEP_DEFS[sweep_name]
        except KeyError as exc:
            available = ", ".join(SWEEP_DEFS)
            raise SystemExit(f"Unknown sweep '{sweep_name}'. Available: {available}") from exc

        sweep_id = wandb.sweep(sweep_cfg_dict, project=project_name)

    run_agent(sweep_id)
