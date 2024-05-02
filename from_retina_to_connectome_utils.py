import os
import random
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn

from flyvision import utils


def get_decision_making_neurons(dtype):
    # Note: this is only run once
    # get a dataframe indicating which neurons will be used to classify
    rational_neurons = pd.read_csv("adult_data/rational_neurons.csv", index_col=0)
    return torch.tensor(rational_neurons.values.squeeze(), dtype=dtype).detach()


def get_tensor_items(x):
    return [a.item() for a in x]


def compute_accuracy(probabilities, labels):

    # Convert probabilities to binary predictions
    predictions = (probabilities > 0.5).float()

    # Calculate accuracy
    return np.where(predictions == labels, 1, 0).float().mean()


def hex_to_square_grid(color, hex_size=15):
    # Coordinate mapping
    u, v = utils.hex_utils.get_hex_coords(hex_size)

    grid_size = hex_size * 2 + 1
    square_x = u + hex_size
    square_y = hex_size - v
    grid = np.full((grid_size, grid_size), np.nan)

    for hex_u, hex_v, hex_color in zip(square_x, square_y, color):
        grid[hex_u, hex_v] = hex_color

    # remove borders, which are not correctly activated
    return grid[1:-1, 1:-1]


def activation_vector_to_image(das):
    square_das = np.apply_along_axis(hex_to_square_grid, axis=1, arr=das.cpu().numpy())
    # replace NaNs with the mean of each layer activation
    layer_means = np.nanmean(square_das, axis=(1, 2), keepdims=True)
    return np.nan_to_num(square_das, nan=layer_means)


def layer_activations_to_decoding_images(la, frame, decoding_cells):
    # only decoding activations
    da = [a[decoding_cells][0, -frame, :, :] for a in la]
    # decoded images to squares without NaNs
    return np.array([activation_vector_to_image(a) for a in da])


def clean_model_outputs(outputs_, batch_labels_):
    outputs_ = torch.sigmoid(outputs_.squeeze().detach().cpu().float())
    predictions_ = torch.round(outputs_).numpy()
    batch_labels_cpu = batch_labels_.detach().cpu().float().numpy()
    correct_ = np.where(predictions_ == batch_labels_cpu, 1, 0)

    return outputs_.numpy(), predictions_, batch_labels_cpu, correct_


def update_running_loss(loss_, inputs_):
    return loss_.item() * inputs_.size(0)


def update_results_df(
    results_, batch_files_, outputs_, predictions_, batch_labels_, correct_
):
    return pd.concat(
        [
            results_,
            pd.DataFrame(
                {
                    "Image": batch_files_,
                    "Model outputs": outputs_,
                    "Prediction": predictions_,
                    "True label": batch_labels_,
                    "Is correct": correct_,
                }
            ),
        ]
    )


def initialize_results_df():
    return pd.DataFrame(
        columns=["Image", "Model outputs", "Prediction", "True label", "Is correct"]
    )


def create_csr_input(activation_df, dtype, device):
    csr_matrix = scipy.sparse.coo_matrix(activation_df.values).tocsr()
    crow_indices = csr_matrix.indptr
    col_indices = csr_matrix.indices
    values = csr_matrix.data
    shape = csr_matrix.shape

    return torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=shape,
        dtype=dtype,
        device=device,
    )


def vector_to_one_hot(vec, dtype, sparse_layout):
    return (
        nn.functional.one_hot(
            torch.nonzero(vec).squeeze().cpu(), num_classes=vec.size(0)
        )
        .to(dtype)
        .to_sparse(layout=sparse_layout)
    )


def select_random_images(all_files, batch_size, already_selected):
    # Filter out files that have already been selected
    remaining_files = [f for f in all_files if f not in already_selected]

    # Select batch_size random videos from the remaining files
    selected_files = random.sample(
        remaining_files, min(batch_size, len(remaining_files))
    )

    # Update the already_selected list
    already_selected.extend(selected_files)

    return selected_files, already_selected


def get_label(name):
    x = os.path.basename(os.path.dirname(name))
    if x == "yellow":
        return 1
    if x == "blue":
        return 0
    raise ValueError(f"Unexpected directory label {x}")


def paths_to_labels(paths):
    return [get_label(a) for a in paths]


def load_custom_sequences(video_paths):
    videos = [np.load(a) for a in video_paths]
    # Assuming videos are numpy arrays with shape (n_frames, height, width, channels)
    return np.array([np.mean(a, axis=3) for a in videos])
