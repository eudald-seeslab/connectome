import pandas as pd
from torch import nn, optim, device, cuda
import torch

import yaml
from tqdm.auto import tqdm, trange

import wandb

from torch.utils.tensorboard import SummaryWriter

from data_parser import adj_matrix, nodes
from image_parser import train_loader, test_loader, validation_loader, debug_loader
from run_functions import run_validation_epoch, run_train_epoch, calculate_test_accuracy
from utils import (
    plot_weber_fraction,
    print_run_details,
    handle_log_configs,
    preliminary_checks,
)
from early_stopper import EarlyStopper
from models import CombinedModel
from model_config_manager import ModelConfigManager
from model_manager import ModelManager


def main(sweep_config=None):
    """
    Main function to run the training and testing of the model.
    Parameters
    ----------
    sweep_config: wandb sweep config object
    """

    config = yaml.safe_load(open("config.yml"))
    debug = config["DEBUG"]
    epochs = config["EPOCHS"] if not debug else 2
    retina_model = config["RETINA_MODEL"]
    batch_size = config["BATCH_SIZE"]
    images_fraction = config["IMAGES_FRACTION"]
    learning_rate = config["LEARNING_RATE"]
    continue_training = config["CONTINUE_TRAINING"]
    saved_model_path = config["SAVED_MODEL_PATH"]
    save_every = config["SAVE_EVERY"]
    wb = config["WANDB"]
    plot_weber = config["PLOT_WEBER_FRACTION"]
    dev = device("cuda" if cuda.is_available() else "cpu")

    if sweep_config is not None:
        config["CONNECTOME_LAYER_NUMBER"] = sweep_config["connectome_layer_number"]

    loader = debug_loader if debug else train_loader
    logger = handle_log_configs(debug)

    preliminary_checks(debug, continue_training, plot_weber, wb, dev, logger)

    # Create the ModelConfigManager and load configurations from YAML files
    config_manager = ModelConfigManager(config)

    # Get a specific configuration by model name
    config_manager.set_model_config(retina_model)
    if config_manager.model_type == "pretrained" and batch_size > 8:
        # If it's a pretrained model, we need to be careful about batch size
        raise ValueError("Pretrained models can't handle batch sizes larger than 8")
    if config_manager.model_type == "pretrained":
        # If it's a pretrained model, save every epoch
        save_every = 1

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
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=1, verbose=True
    )
    early_stopper = EarlyStopper(patience=30, min_delta=0.2)

    # Logs
    # wandb sometimes screws up, so we might want to disable it (in config.yml)
    if wb:
        # This might be called from a sweep run, so init is elsewhere
        if sweep_config is None:
            project_name = f"connectome{'-test' if debug else ''}"
            wandb.init(project=project_name, config=config_manager.current_model_config)
        _ = wandb.watch(combined_model, criterion, log="all")
        wandb.log({"Connectome layer number": config["CONNECTOME_LAYER_NUMBER"]})

        # Save config info
        wandb.config.update(config)

    # I also want the model architecture
    tensorboard_writer = SummaryWriter()
    mock_image = torch.rand(1, 3, 512, 512).to(dev)
    tensorboard_writer.add_graph(combined_model, mock_image)

    print_run_details(config_manager, debug, images_fraction, continue_training)

    # Training loop
    try:
        for epoch in trange(epochs, position=1, leave=True, desc="Epochs"):
            accuracy = 0
            running_loss = 0
            wandb.log({"Epoch": epoch})

            # for images, labels in loader:
            for images, labels in tqdm(loader, position=0, leave=True, desc="Batches"):
                loss, accuracy = run_train_epoch(
                    combined_model, criterion, optimizer, images, labels, dev
                )
                running_loss += loss
            # Run test
            test_accuracy, test_loss = calculate_test_accuracy(
                test_loader, combined_model, criterion, dev
            )
            wandb.log({"Test loss": test_loss, "Test accuracy": test_accuracy})
            scheduler.step(test_loss)

            if early_stopper.early_stop(test_loss):
                logger.info("Early stopping")
                break

            logger.info(
                f"Epoch {epoch+1}/{epochs}, "
                f"\nLoss: {running_loss/len(train_loader)}, Accuracy: {accuracy}"
                f"\nTest loss: {test_loss}, Test accuracy: {test_accuracy}"
            )

            # Save model
            if (epoch + 1) % save_every == 0:
                model_manager.save_model(combined_model, epoch)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        model_manager.save_model(combined_model, epoch)
        logger.info("Continuing to testing...")

    # Clean up intermediate models
    model_manager.clean_previous_runs()

    # Testing
    correct = 0
    total = 0
    validation_results_df = pd.DataFrame(
        columns=["Real Label", "Predicted Label", "Correct Prediction"]
    )

    # Validation loop
    with torch.no_grad():
        # This is overly complicated because I need to get the image names
        #  to compute the weber fraction
        for images, labels in tqdm(validation_loader):
            correct, total, batch_df = run_validation_epoch(
                combined_model, images, labels, total, correct, dev
            )
            # Append the batch DataFrame to the list
            validation_results_df = pd.concat(
                [validation_results_df, batch_df], ignore_index=True
            )
    validation_results_df["Image"] = [a[0] for a in validation_loader.dataset.samples]

    logger.info(f"Accuracy on the {total} test images: {100 * correct / total}%")

    # Store to wandb
    if wb:
        wandb.log({"Validation accuracy": correct / total})

    if plot_weber and wb:
        weber_plot = plot_weber_fraction(validation_results_df)
        try:
            wandb.log({"Weber Fraction Plot": wandb.Image(weber_plot)})
        except FileNotFoundError:
            logger.warning(
                "Could not log Weber fraction plot because wandb screwed up"
            )
            # Save the plot to a file, taking into account the connectome layer number
            model_manager.save_model_plot(weber_plot)

    # Close logs
    if wb:
        wandb.finish()
    tensorboard_writer.close()


if __name__ == "__main__":
    main()
