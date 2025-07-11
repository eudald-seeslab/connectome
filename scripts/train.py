import datetime
import os
from os.path import basename
import random
import traceback
import warnings
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import inspect

from connectome.core.debug_utils import get_logger, model_summary
from connectome.core.graph_models_helpers import EarlyStopping, TrainingError
from configs import config
from connectome.visualization.plots import guess_your_plots, plot_results
from connectome.core.utils import (
    get_image_paths,
    get_iteration_number,
    initialize_results_df,
    process_warnings,
    save_checkpoint,
    select_random_images,
    update_config_with_sweep,
    update_results_df,
    update_running_loss,
    clean_model_outputs
)
from connectome.core.data_processing import DataProcessor
from connectome.core.graph_models import FullGraphModel
from connectome.tools.wandb_logger import WandBLogger

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
    module="wandb.sdk.data_types.image",
)

from torch.optim import AdamW as TorchAdamW


def main(wandb_logger, sweep_config=None):

    u_config = update_config_with_sweep(config, sweep_config)

    random_generator = torch.Generator(device=u_config.DEVICE)
    random_generator.manual_seed(u_config.random_seed)
    np.random.seed(u_config.random_seed)
    random.seed(u_config.random_seed)

    # if it's not a sweep, we need to initialize wandb
    if sweep_config is None:
        wandb_logger.initialize_run(u_config)

    logger = get_logger("ct", u_config.debugging)
    process_warnings(u_config, logger)

    # for saving later
    start_datetime = datetime.datetime.now().isoformat(sep=" ", timespec="minutes")
    dchar = "_DEBUG" if u_config.debugging else ""
    model_name = f"m_{start_datetime}_{wandb_logger.run_id}{dchar}.pth"

    # update batch size number of connectome passes (otherwise we run out of memory)
    batch_size = u_config.batch_size
    batch_size = batch_size // 2 if u_config.NUM_CONNECTOME_PASSES > 5 else batch_size

    # get data and prepare model
    training_images = get_image_paths(u_config.TRAINING_DATA_DIR, u_config.small_length)
    data_processor = DataProcessor(u_config)

    model = FullGraphModel(data_processor, u_config, random_generator).to(u_config.DEVICE)

    # Optimizer: fused AdamW (available with CUDA≥11.4)
    optimizer = TorchAdamW(model.parameters(), lr=u_config.base_lr, fused=True)

    criterion = CrossEntropyLoss()
    # Initialize GradScaler in a version-agnostic way – older PyTorch releases do not
    # accept the `device_type` argument. We therefore inspect the signature and only
    # pass the argument when it is supported to avoid a TypeError.
    if "device_type" in inspect.signature(GradScaler.__init__).parameters:
        scaler = GradScaler(device_type="cuda")
    else:
        scaler = GradScaler()

    total_steps = get_iteration_number(len(training_images), u_config) * u_config.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=u_config.base_lr, total_steps=total_steps
    )

    early_stopping = EarlyStopping(patience=u_config.patience, min_delta=0, target_accuracy=0.99)

    if u_config.resume_checkpoint is not None:
        checkpoint_path = os.path.join("models", u_config.resume_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=u_config.DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Print model details
    model_summary(model)

    # train
    model.train()
    iterations = get_iteration_number(len(training_images), u_config)
    try:
        for ep in range(u_config.num_epochs):

            already_selected = []
            running_loss, total_correct, total = 0, 0, 0
            for i in tqdm(range(iterations)):
                batch_files, already_selected = select_random_images(
                    training_images, batch_size, already_selected
                )
                if u_config.voronoi_criteria == "all":
                    data_processor.voronoi_cells.recreate()
                    data_processor.update_voronoi_state()
                images, labels = data_processor.get_data_from_paths(batch_files)
                if i % u_config.wandb_images_every == 0:
                    p, title = data_processor.plot_input_images(
                        images[0], u_config.voronoi_colour, u_config.voronoi_width
                        )
                    wandb_logger.log_image(p, basename(batch_files[0]), title)

                inputs, labels = data_processor.process_batch(images, labels)

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", dtype=torch.float16):
                    out = model(inputs)
                    loss = criterion(out, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                _, _, _, correct = clean_model_outputs(out, labels)
                running_loss += update_running_loss(loss, inputs)
                total += batch_size
                total_correct += correct.sum()

                wandb_logger.log_metrics(ep, running_loss, total_correct, total)
                if i == 0:
                    first_loss = running_loss
                    if torch.isnan(loss).any():
                        raise TrainingError("Loss is NaN. Training will stop.")
                if i == 100 and running_loss == first_loss:
                    raise TrainingError("Loss is constant. Training will stop.")

            # If epoch is None, it will overwrite the previous checkpoint
            save_checkpoint(
                model,
                optimizer,
                model_name,
                u_config,
                epoch=ep if u_config.save_every_checkpoint else None,
            )
            torch.cuda.empty_cache()

            accuracy = total_correct / total
            logger.info(
                f"Finished epoch {ep + 1} with loss {running_loss / total} "
                f"and accuracy {accuracy}."
            )
            if early_stopping.should_stop(running_loss, accuracy):
                logger.info("Early stopping activated - either perfect accuracy reached or no improvement in loss.")
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Continuing to testing.")

    # test
    testing_images = get_image_paths(u_config.TESTING_DATA_DIR, u_config.small_length)
    already_selected_testing = []
    total_correct, total, running_loss = 0, 0, 0.0
    test_results = initialize_results_df()

    model.eval()
    iterations = get_iteration_number(len(testing_images), u_config)
    with torch.no_grad():
        for _ in tqdm(range(iterations)):
            batch_files, already_selected_testing = select_random_images(
                testing_images, batch_size, already_selected_testing
            )
            images, labels = data_processor.get_data_from_paths(batch_files)
            inputs, labels = data_processor.process_batch(images, labels)
            inputs = inputs.to(u_config.DEVICE)
            labels = labels.to(u_config.DEVICE)

            out = model(inputs)
            loss = criterion(out, labels)

            outputs, predictions, labels_cpu, correct = clean_model_outputs(out, labels)
            test_results = update_results_df(
                test_results, batch_files, outputs, predictions, labels_cpu, correct
            )
            running_loss += update_running_loss(loss, inputs)
            total += batch_size
            total_correct += correct.sum()

    plot_types = guess_your_plots(u_config)
    final_plots = plot_results(
        test_results, plot_types=plot_types, classes=u_config.CLASSES
    )
    wandb_logger.log_validation_stats(
        running_loss, total_correct, total, test_results, final_plots
    )

    logger.info(
        f"Finished testing with loss {running_loss / total} and "
        f"accuracy {total_correct / total}."
    )

    # Free up cached GPU memory so that the next sweep iteration starts from a
    # clean slate.  With *torch.compile* removed we no longer need special
    # Dynamo / Inductor resets.
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":

    logger = get_logger("ct", config.debugging)

    wandb_logger = WandBLogger(
        config.wandb_project, config.wandb_, config.wandb_images_every
    )
    try:
        main(wandb_logger)

    except KeyboardInterrupt:
        logger.error("Testing interrupted by user. Aborting.")

    except Exception:
        error = traceback.format_exc()
        logger.error(error)
        wandb_logger.send_crash(f"Error during training: {error}")

    finally:
        wandb_logger.finish()
