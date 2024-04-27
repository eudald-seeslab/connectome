import traceback
import warnings
import torch
from torch import device, cuda
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

import config
from from_retina_to_connectome_utils import select_random_videos
from from_retina_to_connectome_utils import (
    initialize_results_df,
    predictions_and_corrects_from_model_results,
    update_results_df,
    update_running_loss,
)
from adult_models import FullAdultModel
from wandb_utils import WandBLogger
from full_training_data_processing import FullModelsDataProcessor

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
    module="wandb.sdk.data_types.image",
)

torch.manual_seed(1234)


def main(wandb_logger):

    # get data
    data_processor = FullModelsDataProcessor(wandb_logger=wandb_logger)
    training_videos = data_processor.get_videos(
        config.TRAINING_DATA_DIR, config.small, config.small_length
    )
    validation_videos = data_processor.get_videos(
        config.VALIDATION_DATA_DIR, config.small, config.small_length
    )
    synaptic_matrix = data_processor.synaptic_matrix()
    one_hot_decision_making = data_processor.decision_making_neurons()
    cell_type_indices = data_processor.cell_type_indices()

    # init model
    model = FullAdultModel(
        synaptic_matrix,
        one_hot_decision_making,
        cell_type_indices,
        config.NUM_CONNECTOME_PASSES,
        log_transform_weights=config.log_transform_weights,
        sparse_layout=config.sparse_layout,
        dtype=config.dtype,
        device=config.DEVICE,
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr)
    criterion = BCEWithLogitsLoss()

    # train
    results = initialize_results_df()
    already_selected = []
    running_loss, total_correct, total = 0, 0, 0

    model.train()
    iterations = (
        config.debug_length
        if config.debugging
        else len(training_videos) // config.batch_size
    )
    for i in tqdm(range(iterations)):
        batch_files, already_selected = select_random_videos(
            training_videos, config.batch_size, already_selected
        )
        labels, inputs = data_processor.process_full_models_data(i, batch_files)

        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        # Calculate run parameters
        predictions, labels_cpu, correct = predictions_and_corrects_from_model_results(
            out, labels
        )
        results = update_results_df(
            results, batch_files, predictions, labels_cpu, correct
        )
        running_loss += update_running_loss(loss, inputs)
        total += config.batch_size
        total_correct += correct.sum()

        wandb_logger.log_metrics(i, running_loss, total_correct, total, results)

    print(
        f"Finished training with loss {running_loss / total} and accuracy {total_correct / total}"
    )
    torch.cuda.empty_cache()

    # test
    already_selected_validation = []
    total_correct, total, running_loss = 0, 0, 0.0
    validation_results = initialize_results_df()

    # FIXME: this clashes with the small_length in the validation_videos
    validation_iterations = (
        config.validation_length
        if config.validation_length is not None
        else len(validation_videos) // config.batch_size
    )
    for j in tqdm(range(validation_iterations)):
        batch_files, already_selected_validation = select_random_videos(
            validation_videos, config.batch_size, already_selected_validation
        )

        labels, inputs = data_processor.process_full_models_data(
            j, batch_files
        )

        with torch.no_grad():
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            predictions, batch_labels_cpu, correct = (
                predictions_and_corrects_from_model_results(outputs, labels)
            )
            validation_results = update_results_df(
                validation_results, batch_files, predictions, batch_labels_cpu, correct
            )
            running_loss += update_running_loss(loss, inputs)
            total += batch_labels_cpu.shape[0]
            total_correct += correct.sum().item()

    wandb_logger.log_metrics(0, running_loss, total_correct, total, validation_results)
    wandb_logger.finish()

    print(
        f"Validation Loss: {running_loss / total}, "
        f"Validation Accuracy: {total_correct / total}"
    )


if __name__ == "__main__":

    wandb_logger = WandBLogger("adult_connectome")
    wandb_logger.initialize()
    try:
        main(wandb_logger)
    except Exception:
        error = traceback.format_exc()
        print(error)
        wandb_logger.send_crash(f"Error during training: {error}")
