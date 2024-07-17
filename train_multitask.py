import datetime
from os.path import basename
import traceback
import warnings
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from debug_utils import get_logger, model_summary
from graph_models_helpers import EarlyStopping, TrainingError
import config
import config2
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
from data_processing import CompleteModelsDataProcessor
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

    # get data and prepare model
    training_images1 = get_image_paths(config.TRAINING_DATA_DIR, config.small_length)
    data_processor1 = CompleteModelsDataProcessor(config)
    model = FullGraphModel(data_processor1, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr)
    criterion1 = CrossEntropyLoss()
    early_stopping1 = EarlyStopping(patience=config.patience, min_delta=0)

    training_images2 = get_image_paths(config2.TRAINING_DATA_DIR, config2.small_length)
    data_processor2 = CompleteModelsDataProcessor(config2)
    criterion2 = CrossEntropyLoss()
    early_stopping2 = EarlyStopping(patience=config2.patience, min_delta=0)

    # Print model details
    model_summary(model)

    # train
    model.train()
    iterations = get_iteration_number(len(training_images2), config2)
    try:
        for ep in range(config.num_epochs):

            already_selected1 = []
            already_selected2 = []
            running_loss1, total_correct1, total1 = 0, 0, 0
            running_loss2, total_correct2, total2 = 0, 0, 0
            for i in tqdm(range(iterations)):
                inputs1, labels1, already_selected1 = get_batch_data(
                    config, wandb_logger, batch_size, training_images1, data_processor1, already_selected1, i, 1
                    )
                inputs2, labels2, already_selected2 = get_batch_data(
                    config2, wandb_logger, batch_size, training_images2, data_processor2, already_selected2, i, 2
                    )

                optimizer.zero_grad()
                with torch.autocast(config.device_type):
                    out1 = model(inputs1)
                    loss1 = criterion1(out1, labels1)
                    loss1.backward()

                    out2 = model(inputs2)
                    loss2 = criterion2(out2, labels2)
                    loss2.backward()
                    
                    optimizer.step()

                _, _, _, correct1 = clean_model_outputs(out1, labels1)
                running_loss1 += update_running_loss(loss1, inputs1)
                total1 += batch_size
                total_correct1 += correct1.sum()

                _, _, _, correct2 = clean_model_outputs(out2, labels2)
                running_loss2 += update_running_loss(loss2, inputs2)
                total2 += batch_size
                total_correct2 += correct2.sum()

                wandb_logger.log_metrics(ep, running_loss1, total_correct1, total1, 1)
                wandb_logger.log_metrics(ep, running_loss2, total_correct2, total2, 2)
                if i == 0:
                    first_loss = running_loss1
                    if torch.isnan(loss1).any():
                        raise TrainingError("Loss is NaN. Training will stop.")
                if i == 100 and running_loss1 == first_loss:
                    raise TrainingError("Loss is constant. Training will stop.")

            # save checkpoint (overriding the last)
            save_checkpoint(model, optimizer, model_name, config)
            torch.cuda.empty_cache()

            logger.info(
                f"Finished epoch {ep + 1} with loss {running_loss1 / total1} "
                f"and accuracy {total_correct1 / total1}."
            )
            logger.info(
                f"Finished epoch {ep + 1} with loss {running_loss2 / total2} "
                f"and accuracy {total_correct2 / total2}."
            )

            if early_stopping1.should_stop(running_loss1) and early_stopping2.should_stop(running_loss2):
                logger.info("Early stopping activated. Continuing to testing.")
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Continuing to testing.")

    # test
    testing_images1 = get_image_paths(config.TESTING_DATA_DIR, config.small_length)
    testing_images2 = get_image_paths(config2.TESTING_DATA_DIR, config2.small_length)
    already_selected_testing1 = []
    already_selected_testing2 = []
    total_correct1, total1, running_loss1 = 0, 0, 0.0
    total_correct2, total2, running_loss2 = 0, 0, 0.0
    test_results1 = initialize_results_df()
    test_results2 = initialize_results_df()

    model.eval()
    iterations = get_iteration_number(len(testing_images2), config2)
    with torch.no_grad():
        for _ in tqdm(range(iterations)):
            batch_files, already_selected_testing1 = select_random_images(
                testing_images1, batch_size, already_selected_testing1
            )
            images, labels1 = data_processor1.get_data_from_paths(batch_files)
            inputs1, labels1 = data_processor1.process_batch(images, labels1)
            inputs1 = inputs1.to(config.DEVICE)
            labels1 = labels1.to(config.DEVICE)

            out1 = model(inputs1)
            loss1 = criterion1(out1, labels1)

            outputs1, predictions1, labels_cpu1, correct1 = clean_model_outputs(out1, labels1)
            test_results1 = update_results_df(
                test_results1, batch_files, outputs1, predictions1, labels_cpu1, correct1
            )
            running_loss1 += update_running_loss(loss1, inputs1)
            total1 += batch_size
            total_correct1 += correct1.sum()

            batch_files, already_selected_testing2 = select_random_images(
                testing_images2, batch_size, already_selected_testing2
            )
            images, labels2 = data_processor2.get_data_from_paths(batch_files)
            inputs2, labels2 = data_processor2.process_batch(images, labels2)
            inputs2 = inputs2.to(config.DEVICE)
            labels2 = labels2.to(config.DEVICE)

            out2 = model(inputs2)
            loss2 = criterion2(out2, labels2)

            outputs2, predictions2, labels_cpu2, correct2 = clean_model_outputs(out2, labels2)
            test_results2 = update_results_df(
                test_results2, batch_files, outputs2, predictions2, labels_cpu2, correct2
            )
            running_loss2 += update_running_loss(loss2, inputs2)
            total2 += batch_size
            total_correct2 += correct2.sum()

    plot_types1 = guess_your_plots(config)
    final_plots1 = plot_results(
        test_results1, plot_types=plot_types1, classes=config.CLASSES
    )
    wandb_logger.log_validation_stats(
        running_loss1, total_correct1, total1, test_results1, final_plots1, 1
    )

    plot_types2 = guess_your_plots(config2)
    final_plots2 = plot_results(
        test_results2, plot_types=plot_types2, classes=config2.CLASSES
    )
    wandb_logger.log_validation_stats(
        running_loss2, total_correct2, total2, test_results2, final_plots2, 2
    )

    logger.info(
        f"Finished testing with loss {running_loss1 / total1} and "
        f"accuracy {total_correct1 / total1}."
    )

    logger.info(
        f"Finished testing with loss {running_loss2 / total2} and "
        f"accuracy {total_correct2 / total2}."
    )

def get_batch_data(config_, wandb_logger, batch_size, training_images, data_processor, already_selected, i, task):
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

    return inputs, labels, already_selected

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
