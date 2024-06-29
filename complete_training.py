import datetime
import os
from os.path import basename
import traceback
import warnings
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from debug_utils import get_logger, model_summary
from graph_models_helpers import EarlyStopping, TrainingError
import config
from plots import guess_your_plots, plot_results
from utils import (
    get_image_paths,
    get_iteration_number,
    initialize_results_df,
    process_warnings,
    save_checkpoint,
    select_random_images,
    update_config_with_sweep,
    update_results_df,
    update_running_loss,
)
from complete_training_data_processing import CompleteModelsDataProcessor
from graph_models import FullGraphModel
from utils import (
    clean_model_outputs,
)

from wandb_logger import WandBLogger

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
    module="wandb.sdk.data_types.image",
)

torch.manual_seed(1234)


def main(wandb_logger, sweep_config=None):

    u_config = update_config_with_sweep(config, sweep_config)
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
    data_processor = CompleteModelsDataProcessor(u_config)
    model = FullGraphModel(data_processor, u_config).to(u_config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=u_config.base_lr)
    criterion = CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=u_config.patience, min_delta=0)

    if u_config.resume_checkpoint is not None:
        logger.warning(f"Resuming training from {u_config.resume_checkpoint}")
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
                    # create voronoi cells each batch so they are different
                    data_processor.recreate_voronoi_cells()
                images, labels = data_processor.get_data_from_paths(batch_files)
                if i % u_config.wandb_images_every == 0:
                    p, title = data_processor.plot_input_images(images[0])
                    wandb_logger.log_image(p, basename(batch_files[0]), title)
                    
                inputs, labels = data_processor.process_batch(images, labels)

                optimizer.zero_grad()
                with torch.autocast(u_config.device_type):
                    out = model(inputs)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()

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

            # save checkpoint (overriding the last)
            save_checkpoint(model, optimizer, model_name, u_config)
            torch.cuda.empty_cache()

            logger.info(
                f"Finished epoch {ep + 1} with loss {running_loss / total} "
                f"and accuracy {total_correct / total}."
            )

            if early_stopping.should_stop(running_loss):
                logger.info("Early stopping activated. Continuing to testing.")
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


if __name__ == "__main__":

    logger = get_logger("ct", config.debugging)

    wandb_logger = WandBLogger(config.wandb_project, config.wandb_, config.wandb_images_every)
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
