import warnings
import pandas as pd
import torch
from torch import device, cuda
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from random import sample
from scipy.sparse import load_npz

import flyvision
from flyvision_ans import DECODING_CELLS, FINAL_CELLS
from from_retina_to_connectome_funcs import get_cell_type_indices
from from_retina_to_connectome_utils import (
    get_files_from_directory,
    select_random_videos,
)
from from_retina_to_connectome_utils import (
    initialize_results_df,
    predictions_and_corrects_from_model_results,
    update_results_df,
    update_running_loss,
    get_decision_making_neurons,
    vector_to_one_hot,
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
dtype = torch.float32

device_type = "cuda" if cuda.is_available() else "cpu"
device_type = "cpu"
DEVICE = device(device_type)
sparse_layout = torch.sparse_coo

TRAINING_DATA_DIR = "images/easy_v2"
TESTING_DATA_DIR = "images/easy_images"
VALIDATION_DATA_DIR = "images/easyval_images"

debugging = False
debug_length = 100
validation_length = 50
wandb_ = False
wandb_images_every = 100
small = True
small_length = 400

num_epochs = 1
batch_size = 1

dropout = 0.1
max_lr = 0.01
base_lr = 0.00001
weight_decay = 0.0001
NUM_CONNECTOME_PASSES = 4
final_retina_cells = FINAL_CELLS
normalize_voronoi_cells = True

model_config = {
    "debugging": debugging,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "dropout": dropout,
    "base_lr": base_lr,
    "max_lr": max_lr,
    "weight_decay": weight_decay,
    "num_connectome_passes": NUM_CONNECTOME_PASSES,
}


def main(wandb_logger):
    # TODO: move this to the data processing file
    extent, kernel_size = 15, 13
    decision_making_vector = get_decision_making_neurons(dtype)
    receptors = flyvision.rendering.BoxEye(extent=extent, kernel_size=kernel_size)
    network_view = flyvision.NetworkView(flyvision.results_dir / "opticflow/000/0000")
    network = network_view.init_network(chkpt="best_chkpt")
    classification = pd.read_csv("adult_data/classification_clean.csv")
    root_id_to_index = pd.read_csv("adult_data/root_id_to_index.csv")

    cell_type_indices = get_cell_type_indices(
        classification, root_id_to_index, final_retina_cells
    )

    training_videos = get_files_from_directory(TRAINING_DATA_DIR)
    test_videos = get_files_from_directory(TESTING_DATA_DIR)
    validation_videos = get_files_from_directory(TESTING_DATA_DIR)

    if small:
        training_videos = sample(training_videos, small_length)
        test_videos = sample(test_videos, small_length)
        validation_videos = sample(validation_videos, int(small_length / 5))

    if len(training_videos) == 0:
        print("I can't find any training images or videos!")

    synaptic_matrix = load_npz("adult_data/good_synaptic_matrix.npz")
    one_hot_decision_making = vector_to_one_hot(
        decision_making_vector, dtype, sparse_layout
    ).to(DEVICE)

    # init models and data processor
    model = FullAdultModel(
        synaptic_matrix,
        one_hot_decision_making,
        cell_type_indices,
        NUM_CONNECTOME_PASSES,
        log_transform_weights=True,
        sparse_layout=sparse_layout,
        dtype=dtype,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    criterion = BCEWithLogitsLoss()

    # initialize the data processor
    data_processor = FullModelsDataProcessor(
        wandb_logger=wandb_logger,
        receptors=receptors,
        network=network,
        classification=classification,
        final_retina_cells=final_retina_cells,
        normalize_voronoi_cells=normalize_voronoi_cells,
        root_id_to_index=root_id_to_index,
        dtype=dtype,
        DEVICE=DEVICE,
    )

    # train
    results = initialize_results_df()
    already_selected = []
    running_loss, total_correct, total = 0, 0, 0

    model.train()
    iterations = debug_length if debugging else len(training_videos) // batch_size
    for i in tqdm(range(iterations)):
        batch_files, already_selected = select_random_videos(
            training_videos, batch_size, already_selected
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
        total += batch_size
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

    validation_iterations = (
        validation_length
        if validation_length is not None
        else len(validation_videos) // batch_size
    )
    for j in tqdm(range(validation_iterations)):
        batch_files, already_selected_validation = select_random_videos(
            validation_videos, batch_size, already_selected_validation
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
    wandb_logger = WandBLogger(
        "adult_connectome",
        model_config,
        enabled=wandb_,
        log_images_every=wandb_images_every,
        cell_type_plot="TmY18",
        last_good_frame=2,
    )
    wandb_logger.initialize()
    try:
        main(wandb_logger)
    except Exception as e:
        print(f"Error during training: {e}")
        wandb_logger.send_crash(f"Error during training: {e}")
