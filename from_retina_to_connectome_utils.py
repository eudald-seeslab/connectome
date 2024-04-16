import numpy as np
import pandas as pd
import torch

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


def predictions_and_corrects_from_model_results(outputs_, batch_labels_):
    predictions_ = torch.round(torch.sigmoid(outputs_).squeeze()).detach().cpu().float().numpy()
    batch_labels_cpu = batch_labels_.detach().cpu().float().numpy()
    correct_ = np.where(predictions_ == batch_labels_cpu, 1, 0)

    return predictions_, batch_labels_cpu, correct_


def update_running_loss(loss_, inputs_):
    return loss_.item() * inputs_.size(0)


def update_results_df(results_, batch_files_, predictions_, batch_labels_, correct_):
    return pd.concat(
        [
            results_,
            pd.DataFrame(
                {
                    "Image": batch_files_,
                    "Prediction": predictions_,
                    "True label": batch_labels_,
                    "Is correct": correct_,
                }
            ),
        ]
    )


def initialize_results_df():
    return pd.DataFrame(columns=["Image", "Prediction", "True label", "Is correct"])


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

