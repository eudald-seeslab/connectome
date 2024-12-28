from os.path import basename
import datetime
import traceback
import warnings
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from debug_utils import get_logger, model_summary
from graph_models_helpers import EarlyStopping, TrainingError
import config
from config_multitasking_dirs import data_dirs_config
from plots import guess_your_plots, plot_results
from utils import (
    get_image_paths,
    get_iteration_number,
    initialize_results_df,
    process_warnings,
    save_checkpoint,
    select_random_images,
    update_results_df,
    update_running_loss,
)
from data_processing import DataProcessor
from graph_models import FullGraphModel
from utils import clean_model_outputs
from wandb_logger import WandBLogger

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
    module="wandb.sdk.data_types.image",
)

torch.manual_seed(1234)


def main(wandb_logger):
    logger = get_logger("ct", config.debugging)
    process_warnings(config, logger)

    wandb_logger.initialize_run(config)

    # for saving later
    start_datetime = datetime.datetime.now().isoformat(sep=" ", timespec="minutes")
    dchar = "_DEBUG" if config.debugging else ""
    model_name = f"m_{start_datetime}_{wandb_logger.run_id}{dchar}.pth"

    # update batch size number of connectome passes (otherwise we run out of memory)
    batch_size = config.batch_size
    batch_size = batch_size // 2 if config.NUM_CONNECTOME_PASSES > 5 else batch_size

    # Initialize data, models, criteria, and early stopping mechanisms for each task
    data_processors = []
    early_stoppings = []
    criterions = []
    training_images = []

    for data_dir in data_dirs_config:
        training_data_dir = data_dir["TRAINING_DATA_DIR"]
        training_images.append(get_image_paths(training_data_dir, config.small_length))
        data_processors.append(DataProcessor(config, data_dir=training_data_dir))
        criterions.append(CrossEntropyLoss())
        early_stoppings.append(EarlyStopping(patience=config.patience, min_delta=0))

    model = FullGraphModel(data_processors[0], config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr)

    min_images = min(len(images) for images in training_images)
    config_min = config

    # Print model details
    model_summary(model)

    # train
    model.train()
    iterations = get_iteration_number(min_images, config_min)
    try:
        for ep in range(config.num_epochs):
            already_selected = [[] for _ in data_dirs_config]
            running_losses = [0] * len(data_dirs_config)
            total_correct = [0] * len(data_dirs_config)
            totals = [0] * len(data_dirs_config)

            for i in tqdm(range(iterations)):
                inputs_labels = [
                    get_batch_data(
                        config,
                        wandb_logger,
                        batch_size,
                        training_images[idx],
                        data_processors[idx],
                        already_selected[idx],
                        i,
                        idx + 1,
                    )
                    for idx in range(len(data_dirs_config))
                ]
                inputs_labels = list(zip(*inputs_labels))

                inputs = inputs_labels[1]
                labels = inputs_labels[2]
                already_selected = inputs_labels[3]

                optimizer.zero_grad()
                with torch.autocast(config.device_type):
                    losses = []
                    outputs = []
                    for idx, (inp, lbl, crit) in enumerate(
                        zip(inputs, labels, criterions)
                    ):
                        out = model(inp)
                        loss = crit(out, lbl)
                        loss.backward()
                        outputs.append(out)
                        losses.append(loss)
                    optimizer.step()

                for idx, (out, loss, inp, lbl) in enumerate(
                    zip(outputs, losses, inputs, labels)
                ):
                    _, _, _, correct = clean_model_outputs(out, lbl)
                    running_losses[idx] += update_running_loss(loss, inp)
                    totals[idx] += batch_size
                    total_correct[idx] += correct.sum()

                    wandb_logger.log_metrics(
                        ep,
                        running_losses[idx],
                        total_correct[idx],
                        totals[idx],
                        idx + 1,
                    )

                    if i == 0:
                        first_loss = running_losses[idx]
                        if torch.isnan(loss).any():
                            raise TrainingError("Loss is NaN. Training will stop.")
                    if i == 100 and running_losses[idx] == first_loss:
                        raise TrainingError("Loss is constant. Training will stop.")

            # save checkpoint (overriding the last)
            save_checkpoint(model, optimizer, model_name, config)
            torch.cuda.empty_cache()

            for idx in range(len(data_dirs_config)):
                logger.info(
                    f"Finished epoch {ep + 1} with loss {running_losses[idx] / totals[idx]} "
                    f"and accuracy {total_correct[idx] / totals[idx]}."
                )

            if all(
                early_stoppings[idx].should_stop(running_losses[idx])
                for idx in range(len(data_dirs_config))
            ):
                logger.info("Early stopping activated. Continuing to testing.")
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Continuing to testing.")

    # test
    logger.info("Starting testing.")
    testing_images = [
        get_image_paths(data_dir["TESTING_DATA_DIR"], config.small_length)
        for data_dir in data_dirs_config
    ]
    already_selected_testing = [[] for _ in data_dirs_config]
    total_correct = [0] * len(data_dirs_config)
    totals = [0] * len(data_dirs_config)
    running_losses = [0.0] * len(data_dirs_config)
    test_results = [initialize_results_df() for _ in data_dirs_config]

    min_images = min(len(images) for images in testing_images)

    model.eval()
    iterations = get_iteration_number(min_images, config)
    with torch.no_grad():
        for _ in tqdm(range(iterations)):
            batch_data = [
                get_batch_data(
                    config,
                    wandb_logger,
                    batch_size,
                    testing_images[idx],
                    data_processors[idx],
                    already_selected_testing[idx],
                    i,
                    idx + 1,
                )
                for idx in range(len(data_dirs_config))
            ]
            batch_data = list(zip(*batch_data))

            batch_files = batch_data[0]
            inputs = batch_data[1]
            labels = batch_data[2]
            already_selected_testing = batch_data[3]

            for idx, (inp, lbl, crit) in enumerate(zip(inputs, labels, criterions)):
                out = model(inp)
                loss = crit(out, lbl)

                outputs, predictions, labels_cpu, correct = clean_model_outputs(
                    out, lbl
                )
                test_results[idx] = update_results_df(
                    test_results[idx],
                    batch_files[idx],
                    outputs,
                    predictions,
                    labels_cpu,
                    correct,
                )
                running_losses[idx] += update_running_loss(loss, inp)
                totals[idx] += batch_size
                total_correct[idx] += correct.sum()

        for idx, data_dir in enumerate(data_dirs_config):
            plot_types = guess_your_plots(config)
            final_plots = plot_results(
                test_results[idx], plot_types=plot_types, classes=config.CLASSES
            )
            wandb_logger.log_validation_stats(
                running_losses[idx],
                total_correct[idx],
                totals[idx],
                test_results[idx],
                final_plots,
                idx + 1,
            )

            logger.info(
                f"Finished testing with loss {running_losses[idx] / totals[idx]} and "
                f"accuracy {total_correct[idx] / totals[idx]}."
            )


def get_batch_data(
    config_,
    wandb_logger,
    batch_size,
    training_images,
    data_processor,
    already_selected,
    i,
    task,
):
    batch_files, already_selected = select_random_images(
        training_images, batch_size, already_selected
    )
    if config_.voronoi_criteria == "all":
        # create voronoi cells each batch so they are different
        data_processor.recreate_voronoi_cells()
    inputs, labels = data_processor.get_data_from_paths(batch_files)
    if i % config_.wandb_images_every == 0:
        p, title = data_processor.plot_input_images(inputs[0])
        wandb_logger.log_image(p, basename(batch_files[0]), title, task)

    inputs, labels = data_processor.process_batch(inputs, labels)

    return batch_files, inputs, labels, already_selected


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
