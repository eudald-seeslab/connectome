# main run

import pandas as pd
from torch import nn, optim, device, cuda
import torch

import yaml
from tqdm.auto import tqdm, trange

import wandb
from torch.utils.tensorboard import SummaryWriter

from data_parser import adj_matrix, nodes
from image_parser import train_loader, test_loader, debug_loader
from run_functions import run_test_epoch, run_train_epoch
from utils import (
    plot_weber_fraction,
    print_run_details,
    handle_log_configs,
    get_image_names,
    preliminary_checks,
)

from models import CombinedModel
from model_config_manager import ModelConfigManager
from model_manager import ModelManager


def main(sweep_config=None):
    config = yaml.safe_load(open("config.yml"))
    DEBUG = config["DEBUG"]
    epochs = config["EPOCHS"] if not DEBUG else 2
    RETINA_MODEL = config["RETINA_MODEL"]
    images_fraction = config["IMAGES_FRACTION"]
    continue_training = config["CONTINUE_TRAINING"]
    saved_model_path = config["SAVED_MODEL_PATH"]
    save_every = config["SAVE_EVERY"]
    wb = config["WANDB"]
    plot_weber = config["PLOT_WEBER_FRACTION"]
    dev = device("cuda" if cuda.is_available() else "cpu")

    # FIXME: this is terrible
    if sweep_config is not None:
        config["CONNECTOME_LAYER_NUMBER"] = sweep_config["connectome_layer_number"]

    loader = debug_loader if DEBUG else train_loader
    logger = handle_log_configs(DEBUG)

    preliminary_checks(DEBUG, continue_training, plot_weber, wb, dev, logger)

    # Create the ModelConfigManager and load configurations from YAML files
    config_manager = ModelConfigManager(config)

    # Get a specific configuration by model name
    config_manager.set_model_config(RETINA_MODEL)

    combined_model = CombinedModel(
        adj_matrix,
        neurons=nodes,
        model_config=config_manager.current_model_config,
        general_config=config,
    )

    # Saving and loading manager
    model_manager = ModelManager(config, save_dir="models", clean_previous=True)

    if continue_training:
        # If we want to continue training a saved model
        model_manager.load_model(combined_model, saved_model_path)

    # Model details
    combined_model = combined_model.to(dev)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=0.00001)

    # Logs
    # wandb sometimes screws up, so we might want to disable it (in config.yml)
    if wb:
        # This might be called from a sweep run, so init is elsewhere
        if sweep_config is None:
            project_name = f"connectome{'-test' if DEBUG else ''}"
            wandb.init(project=project_name, config=config_manager.current_model_config)
        _ = wandb.watch(combined_model, criterion, log="all")
        wandb.log({"Connectome layer number": config["CONNECTOME_LAYER_NUMBER"]})

        # Save config info
        wandb.config.update(config)

    # I also want the model architecture
    tensorboard_writer = SummaryWriter()
    mock_image = torch.rand(1, 3, 512, 512).to(dev)
    tensorboard_writer.add_graph(combined_model, mock_image)

    print_run_details(config_manager, DEBUG, images_fraction, continue_training)

    # Training loop
    for epoch in trange(epochs, position=1, leave=True, desc="Epochs"):
        running_loss = 0
        correct_predictions = 0

        for images, labels in tqdm(loader, position=0, leave=True, desc="Batches"):
            running_loss += run_train_epoch(
                combined_model, criterion, optimizer, images, labels, epoch, dev
            )

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

        # Save model
        if (epoch + 1) % save_every == 0:
            model_manager.save_model(combined_model, epoch)

    # Clean up intermediate models
    model_manager.clean_previous_runs()

    # Testing
    correct = 0
    total = 0
    test_results_df = pd.DataFrame(
        columns=["Image", "Real Label", "Predicted Label", "Correct Prediction"]
    )

    # Training loop
    with torch.no_grad():
        j = 0
        for images, labels in tqdm(test_loader):
            # Get the correct image names
            image_names, j = get_image_names(j, test_loader)
            correct, total, batch_df = run_test_epoch(
                combined_model, images, labels, image_names, total, correct, dev
            )
            # Append the batch DataFrame to the list
            test_results_df = pd.concat([test_results_df, batch_df], ignore_index=True)

    logger.info(f"Accuracy on the {total} test images: {100 * correct / total}%")

    # Store to wandb
    if wb:
        wandb.log({"Test accuracy": correct / total})

    if plot_weber and wb:
        weber_plot = plot_weber_fraction(test_results_df, model_manager.model_dir)
        wandb.log({"Weber Fraction Plot": wandb.Image(weber_plot)})

    # Close logs
    if wb:
        wandb.finish()
    tensorboard_writer.close()


if __name__ == "__main__":
    main()
