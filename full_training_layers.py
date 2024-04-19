import warnings
import pandas as pd
import torch
from torch import device, cuda
from torch.cuda.amp import GradScaler
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import wandb
from random import sample
from scipy.sparse import load_npz

import flyvision
from flyvision_ans import DECODING_CELLS
from flyvision.utils.activity_utils import LayerActivity
from from_image_to_video import image_paths_to_sequences
from from_retina_to_connectome_funcs import get_cell_type_indices, compute_voronoi_averages, from_retina_to_connectome
from logs_to_wandb import log_images_to_wandb, log_running_stats_to_wandb
from from_video_to_training_batched_funcs import get_files_from_directory, select_random_videos, paths_to_labels
from from_retina_to_connectome_utils import (
    hex_to_square_grid,
    initialize_results_df,
    predictions_and_corrects_from_model_results,
    update_results_df,
    update_running_loss,
    get_decision_making_neurons,
    vector_to_one_hot,
)
from adult_models import FullAdultModel


warnings.filterwarnings(
    'ignore',
    message='invalid value encountered in cast',
    category=RuntimeWarning,
    module='wandb.sdk.data_types.image'
)

torch.manual_seed(1234)
dtype = torch.float32

device_type = "cuda" if cuda.is_available() else "cpu"
# device_type = "cpu"
DEVICE = device(device_type)
sparse_layout = torch.sparse_coo

TRAINING_DATA_DIR = "images/easy_v2"
TESTING_DATA_DIR = "images/easy_images"
VALIDATION_DATA_DIR = "images/easyval_images"

debugging = False
debug_length = 100
validation_length = 50
wandb_ = True
wandb_images_every = 100
small = False
small_length = 1000

num_epochs = 1
batch_size = 1

dropout = .1
max_lr = 0.01
base_lr = 0.0001
weight_decay = 0.0001
NUM_CONNECTOME_PASSES = 15

use_one_cycle_lr = False

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

# init stuff
extent, kernel_size = 15, 13
decision_making_vector = get_decision_making_neurons(dtype)
receptors = flyvision.rendering.BoxEye(extent=extent, kernel_size=kernel_size)
network_view = flyvision.NetworkView(flyvision.results_dir / "opticflow/000/0000")
network = network_view.init_network(chkpt="best_chkpt")
classification = pd.read_csv("adult_data/classification_clean.csv")
root_id_to_index = pd.read_csv("adult_data/root_id_to_index.csv")
dt = 1 / 100  # some parameter from flyvision
last_good_frame = 2
cell_type_plot = "TmY18"

cell_type_indices = get_cell_type_indices(
    classification, root_id_to_index, DECODING_CELLS
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

synaptic_matrix = load_npz("adult_data/synaptic_matrix_sparse.npz")
one_hot_decision_making = vector_to_one_hot(
    decision_making_vector, dtype, sparse_layout
).to(DEVICE)

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
scaler = GradScaler()

# Initialize the loss function
criterion = BCEWithLogitsLoss()

if wandb_:
    wandb.init(project="adult_connectome", config=model_config)

model.train()

results = initialize_results_df()
probabilities = []
accuracies = []
already_selected = []
running_loss = 0.0
total_correct = 0
total = 0

iterations = debug_length if debugging else len(training_videos) // batch_size

for i in tqdm(range(iterations)):
    batch_files, already_selected = select_random_videos(
        training_videos, batch_size, already_selected
    )
    labels = paths_to_labels(batch_files)
    batch_sequences = image_paths_to_sequences(batch_files)
    rendered_sequences = receptors(batch_sequences)

    layer_activations = []
    for rendered_sequence in rendered_sequences:
        # rendered sequences are in RGB; move it to 0-1 for better training
        rendered_sequence = torch.div(rendered_sequence, 255)
        simulation = network.simulate(rendered_sequence[None], dt)
        layer_activations.append(
            LayerActivity(simulation, network.connectome, keepref=True)
        )

    if wandb_ and i % wandb_images_every == 0:
        try:
            la_0 = (
                hex_to_square_grid(
                    layer_activations[0][cell_type_plot]
                    .squeeze()[-last_good_frame]
                    .cpu()
                    .numpy()
                ),
            )
            log_images_to_wandb(
                batch_sequences[0],
                rendered_sequences[0],
                la_0,
                batch_files[0],
                frame=last_good_frame,
                cell_type=cell_type_plot,
            )
        except Exception as e:
            print(f"Error logging to wandb: {e}. Continuing...")

    voronoi_averages_df = compute_voronoi_averages(
        layer_activations,
        classification,
        DECODING_CELLS,
        last_good_frame=last_good_frame,
    )
    # normalize column wise (except last column)
    values_cols = voronoi_averages_df.columns != "index_name"
    voronoi_averages_df.loc[:, values_cols] = voronoi_averages_df.loc[
        :, values_cols
    ].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    activation_df = from_retina_to_connectome(
        voronoi_averages_df, classification, root_id_to_index
    )
    del layer_activations, rendered_sequences, rendered_sequence, simulation
    torch.cuda.empty_cache()

    optimizer.zero_grad()

    inputs = torch.tensor(activation_df.values, dtype=dtype, device=DEVICE)
    labels = torch.tensor(labels, dtype=dtype, device=DEVICE)

    out = model(inputs)
    loss = criterion(out, labels)
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    loss.backward()
    optimizer.step()

    # Calculate run parameters
    predictions, labels_cpu, correct = predictions_and_corrects_from_model_results(
        out, labels
    )
    results = update_results_df(results, batch_files, predictions, labels_cpu, correct)
    running_loss += update_running_loss(loss, inputs)
    total += batch_size
    total_correct += correct.sum()
    print(
        f"Loss: {running_loss / total}, Accuracy: {total_correct / total}"
    )
    print(predictions, labels_cpu, correct)

    if wandb_:
        try:
            log_running_stats_to_wandb(0, i, running_loss, total_correct, total, results)
        except Exception as e:
            print(f"Error logging to wandb: {e}. Continuing...")

print(
    f"Finished training with loss {running_loss / total} and accuracy {total_correct / total}"
)
torch.cuda.empty_cache()

# Validation

already_selected_validation = []
total_correct = 0
total = 0
running_loss = 0.0
validation_results = initialize_results_df()

validation_iterations = (
    validation_length
    if validation_length is not None
    else len(validation_videos) // batch_size
)
for _ in tqdm(range(validation_iterations)):
    batch_files, already_selected_validation = select_random_videos(
        validation_videos, batch_size, already_selected_validation
    )

    labels = paths_to_labels(batch_files)
    batch_sequences = image_paths_to_sequences(batch_files)
    rendered_sequences = receptors(batch_sequences)
    layer_activations = []
    for rendered_sequence in rendered_sequences:
        simulation = network.simulate(rendered_sequence[None], dt)
        layer_activations.append(
            LayerActivity(simulation, network.connectome, keepref=True)
        )

    voronoi_averages_df = compute_voronoi_averages(
        layer_activations,
        classification,
        DECODING_CELLS,
        last_good_frame=last_good_frame,
    )
    # normalize column wise (except last column)
    values_cols = voronoi_averages_df.columns != "index_name"
    voronoi_averages_df.loc[:, values_cols] = voronoi_averages_df.loc[
        :, values_cols
    ].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    activation_df = from_retina_to_connectome(
        voronoi_averages_df, classification, root_id_to_index
    )
    del layer_activations, rendered_sequences, rendered_sequence, simulation
    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(activation_df.values, dtype=dtype, device=DEVICE)
        labels = torch.tensor(labels, dtype=dtype, device=DEVICE)

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

if wandb_:
    try:
        log_running_stats_to_wandb(
            0, 0, running_loss, total_correct, total, validation_results
        )
    except Exception as e:
        print(f"Error logging validation stats: {e}. Continuing...")


print(
    f"Validation Loss: {running_loss / total}, "
    f"Validation Accuracy: {total_correct / total}"
)
