from os.path import basename
import traceback
import warnings
from matplotlib import pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from adult_models_helpers import TrainingError
import config
from utils import (
    get_image_paths,
    get_iteration_number,
    initialize_results_df,
    plot_results,
    select_random_images,
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

    if sweep_config is None:
        neurons = config.neurons
        voronoi_criteria = config.voronoi_criteria
        random_synapses = config.random_synapses
        base_lr = config.base_lr
        NUM_CONNECTOME_PASSES = config.NUM_CONNECTOME_PASSES
    else:
        neurons = sweep_config.neurons
        voronoi_criteria = sweep_config.voronoi_criteria
        random_synapses = sweep_config.random_synapses
        base_lr = sweep_config.base_lr
        NUM_CONNECTOME_PASSES = sweep_config.NUM_CONNECTOME_PASSES

    # update batch size number of connectome passes (otherwise we run out of memory)
    batch_size = (
        config.batch_size // 2 if NUM_CONNECTOME_PASSES > 5 else config.batch_size
    )

    # get data and prepare model
    training_images = get_image_paths(
        config.TRAINING_DATA_DIR, config.small, config.small_length
    )
    data_processor = CompleteModelsDataProcessor(
        neurons=neurons,
        voronoi_criteria=voronoi_criteria,
        random_synapses=random_synapses,
        log_transform_weights=config.log_transform_weights
        )

    model = FullGraphModel(
        input_shape=data_processor.number_of_synapses,
        num_connectome_passes=NUM_CONNECTOME_PASSES,
        decision_making_vector=data_processor.decision_making_vector,
        batch_size=batch_size,
        dtype=config.dtype,
        edge_weights=data_processor.synaptic_matrix.data,
        device=config.DEVICE,
        num_classes=len(config.CLASSES),
    ).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    criterion = CrossEntropyLoss()

    # train
    model.train()
    results = initialize_results_df()

    iterations = get_iteration_number(len(training_images), batch_size)
    for ep in range(config.num_epochs):

        already_selected = []
        running_loss, total_correct, total = 0, 0, 0
        for i in tqdm(range(iterations)):
            batch_files, already_selected = select_random_images(
                training_images, batch_size, already_selected
            )
            # create voronoi cells each batch so they are different
            data_processor.create_voronoi_cells()
            images, labels = data_processor.get_data_from_paths(batch_files)
            if i % config.wandb_images_every == 0:
                p1 = data_processor.plot_voronoi_cells_with_image(images[0])
                wandb_logger.log_image(p1, basename(batch_files[0]), "Original")
                p2 = data_processor.plot_voronoi_cells_with_neurons()
                wandb_logger.log_image(p2, "Voronoi cells", "Voronoi")
                plt.close("all")

            inputs, labels = data_processor.process_batch(images, labels)

            optimizer.zero_grad()
            with torch.autocast(config.device_type):
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
            total += batch_size
            total_correct += correct.sum()

            wandb_logger.log_metrics(ep, i, running_loss, total_correct, total)
            if i == 0:
                first_loss = running_loss
            if i == 100 and running_loss == first_loss:
                raise TrainingError("Loss is constant. Training will stop.")

        wandb_logger.log_dataframe(results, "Training results")
        print(
            f"Finished epoch {ep + 1} with loss {running_loss / total} "
            f"and accuracy {total_correct / total}."
        )
        torch.cuda.empty_cache()

    # test
    testing_images = get_image_paths(
        config.TESTING_DATA_DIR, config.small, config.small_length
    )
    already_selected_testing = []
    total_correct, total, running_loss = 0, 0, 0.0
    test_results = initialize_results_df()

    model.eval()
    iterations = get_iteration_number(len(testing_images), batch_size)
    for _ in tqdm(range(iterations)):
        batch_files, already_selected_testing = select_random_images(
            testing_images, batch_size, already_selected_testing
        )
        images, labels = data_processor.get_data_from_paths(batch_files)
        inputs, labels = data_processor.process_batch(images, labels)
        inputs = inputs.to(config.DEVICE)

        out = model(inputs)
        loss = criterion(out, labels)

        # Calculate run parameters
        outputs, predictions, labels_cpu, correct = clean_model_outputs(out, labels)
        test_results = update_results_df(
            test_results, batch_files, outputs, predictions, labels_cpu, correct
        )
        running_loss += update_running_loss(loss, inputs)
        total += batch_size
        total_correct += correct.sum()

    final_plots = plot_results(test_results, plot_types=config.plot_types)
    wandb_logger.log_validation_stats(
        running_loss, total_correct, total, test_results, final_plots
    )

    print(
        f"Finished testing with loss {running_loss / total} and "
        f"accuracy {total_correct / total}."
    )


if __name__ == "__main__":

    wandb_logger = WandBLogger("adult_complete")
    wandb_logger.initialize_run()
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