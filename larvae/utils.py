import pandas as pd
import torch
import wandb
import logging
from random import getrandbits
import matplotlib.pyplot as plt
import seaborn as sns
from model_config_manager import ModelConfigManager


def log_training_images(
    images: torch.Tensor, labels: torch.Tensor, outputs: torch.Tensor
) -> None:
    # Don't always log, or we saturate wandb
    # Substitute for a number smaller or equal to batch size if you want more
    #  images
    num_images_to_log = getrandbits(1)
    for i in range(num_images_to_log):
        # Convert image and label to numpy for visualization
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        label = labels[i].cpu().numpy()
        predicted_label = torch.argmax(outputs[i]).cpu().numpy()

        # There is a bug in wandb that prevents logging images
        try:
            wandb.log(
                {
                    "Training Image": wandb.Image(
                        image,
                        caption=f"True Label: {label}, Predicted Label: {predicted_label}",
                    )
                }
            )
        except FileNotFoundError:
            logger = logging.getLogger("training_log")
            logger.warning(f"Could not log training image")


def print_run_details(
    config_manager: ModelConfigManager,
    debug: bool,
    images_fraction: float,
    continue_training: bool,
) -> None:
    logger = logging.getLogger("training_log")
    config_manager.output_model_details()
    if debug:
        logger.warning("WARNING: Running on DEBUG mode, so using 10% of the images")
    elif images_fraction < 1:
        logger.warning(f"WARNING: Using {images_fraction * 100}% of the images")
    if continue_training:
        logger.warning("Warning: I'm training and already trained model")


def handle_log_configs(debug: bool) -> logging.Logger:
    logging.basicConfig(
        filename="training_log.log",
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("raining_log")
    # In the sweeps, this gets called multiple times
    if not len(logger.handlers):
        logger.addHandler(logging.FileHandler("training_log.log"))
        logger.addHandler(logging.StreamHandler())

    return logger


def preliminary_checks(
    debug: bool,
    continue_training: bool,
    plot_weber: bool,
    wb: bool,
    dev: torch.device,
    logger: logging.Logger,
) -> None:
    if debug and continue_training:
        raise ValueError("Can't continue training in DEBUG mode")
    if plot_weber and not wb:
        raise ValueError("Can't log Weber fraction plot without wandb")
    if dev.type == "cpu":
        logger.warning("WARNING: Running on CPU, so it might be slow")


def flush_cuda_memory():
    import gc
    import torch

    torch.cuda.empty_cache()
    gc.collect()
