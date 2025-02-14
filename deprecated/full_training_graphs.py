import traceback
import warnings
import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from configs import config
from deprecated.from_retina_to_connectome_utils import (
    get_decision_making_neurons,
)
from full_training_data_processing import FullModelsDataProcessor
from connectome import FullGraphModel
from connectome.visualization.plots import plot_weber_fraction
from connectome import (
    clean_model_outputs,
    initialize_results_df,
    select_random_images,
    update_results_df,
    update_running_loss,
)
from wandb_utils import WandBLogger

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
    training_videos = data_processor.get_images(
        config.TRAINING_DATA_DIR, config.small, config.small_length
    )
    validation_videos = data_processor.get_images(
        config.VALIDATION_DATA_DIR,
        config.validation_length is not None,
        config.validation_length,
    )
    # TODO: move this into data_processor
    decision_making_vector = get_decision_making_neurons(config.dtype)

    model = FullGraphModel(
        input_shape=data_processor.synaptic_matrix.shape[0],
        num_connectome_passes=config.NUM_CONNECTOME_PASSES,
        decision_making_vector=decision_making_vector,
        log_transform_weights=config.log_transform_weights,
        batch_size=config.batch_size,
        dtype=config.dtype,
        cell_type_indices=data_processor.cell_type_indices(),
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
        batch_files, already_selected = select_random_images(
            training_videos, config.batch_size, already_selected
        )

        inputs, labels = data_processor.process_full_models_graph_data(i, batch_files)
        inputs = inputs.to(config.DEVICE)

        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        # Calculate run parameters
        outputs, predictions, labels_cpu, correct = clean_model_outputs(out, labels)
        results = update_results_df(
            results, batch_files, outputs, predictions, labels_cpu, correct
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
    model.eval()
    already_selected_validation = []
    total_correct, total, running_loss = 0, 0, 0.0
    validation_results = initialize_results_df()

    for j in tqdm(range(len(validation_videos) // config.batch_size)):
        batch_files, already_selected_validation = select_random_images(
            validation_videos, config.batch_size, already_selected_validation
        )

        inputs, labels = data_processor.process_full_models_graph_data(j, batch_files)
        inputs = inputs.to(config.DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            outputs, predictions, batch_labels_cpu, correct = clean_model_outputs(
                outputs, labels
            )
            validation_results = update_results_df(
                validation_results,
                batch_files,
                outputs,
                predictions,
                batch_labels_cpu,
                correct,
            )
            running_loss += update_running_loss(loss, inputs)
            total += batch_labels_cpu.shape[0]
            total_correct += correct.sum().item()

    print(
        f"Validation Loss: {running_loss / total}, "
        f"Validation Accuracy: {total_correct / total}"
    )

    weber_plot = plot_weber_fraction(validation_results)
    wandb_logger.log_validation_stats(
        running_loss, total_correct, total, validation_results, weber_plot
    )
    wandb_logger.finish()


if __name__ == "__main__":

    wandb_logger = WandBLogger("adult_connectome")
    wandb_logger.initialize()
    try:
        main(wandb_logger)
    except Exception:
        error = traceback.format_exc()
        print(error)
        wandb_logger.send_crash(f"Error during training: {error}")
