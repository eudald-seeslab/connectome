import traceback
import warnings
import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

import config
from utils import get_image_paths, get_iteration_number, plot_weber_fraction
from complete_training_data_processing import CompleteModelsDataProcessor
from graph_models import FullGraphModel
from from_retina_to_connectome_utils import (
    select_random_images,
    initialize_results_df,
    clean_model_outputs,
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
    # get data and prepare model
    training_images = get_image_paths(config.TRAINING_DATA_DIR, config.small, config.small_length)
    data_processor = CompleteModelsDataProcessor(config.log_transform_weights)

    model = FullGraphModel(
        input_shape=data_processor.number_of_synapses,
        num_connectome_passes=config.NUM_CONNECTOME_PASSES,
        decision_making_vector=data_processor.decision_making_vector,
        batch_size=config.batch_size,
        dtype=config.dtype,
        edge_weights=data_processor.synaptic_matrix.data,
        device=config.DEVICE,
        retina_connection=False,
    ).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr)
    criterion = BCEWithLogitsLoss()

    # train
    model.train()
    results = initialize_results_df()
    running_loss, total_correct, total = 0, 0, 0

    iterations = get_iteration_number(len(training_images), config.batch_size)
    for ep in range(config.num_epochs):
        already_selected = []
        for i in tqdm(range(iterations)):
            batch_files, already_selected = select_random_images(
                training_images, config.batch_size, already_selected
            )
            # create voronoi cells each batch so they are different
            data_processor.create_voronoi_cells()
            inputs, labels = data_processor.process_batch(batch_files)
            
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
            f"Finished epoch {ep} with loss {running_loss / total} and accuracy {total_correct / total}"
        )
        torch.cuda.empty_cache()

    try:
        weber_plot = plot_weber_fraction(results)
    except:
        weber_plot = None
    wandb_logger.log_validation_stats(
        running_loss, total_correct, total, results, weber_plot
    )

    print(
        f"Finished training with loss {running_loss / total} and accuracy {total_correct / total}"
    )


if __name__ == "__main__":

    wandb_logger = WandBLogger("adult_complete")
    wandb_logger.initialize()
    try:
        main(wandb_logger)

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    except Exception:
        error = traceback.format_exc()
        print(error)
        wandb_logger.send_crash(f"Error during training: {error}")

    finally:
        wandb_logger.finish()