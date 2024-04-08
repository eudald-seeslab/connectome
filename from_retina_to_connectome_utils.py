import numpy as np
import pandas as pd
import torch

from logs_to_wandb import hex_to_square_grid


def get_decision_making_neurons():
    # Note: this is only run once
    # get a dataframe indicating which neurons will be used to classify
    rational_neurons = pd.read_csv("adult_data/rational_neurons.csv", index_col=0)
    return torch.tensor(rational_neurons.values.squeeze(), dtype=torch.float16).detach()


def get_tensor_items(x):
    return [a.item() for a in x]


def compute_accuracy(probabilities, labels):

    # Convert probabilities to binary predictions
    predictions = (probabilities > 0.5).float()

    # Calculate accuracy
    return np.where(predictions == labels, 1, 0).float().mean()


def activation_vector_to_image(da):
    images = np.apply_along_axis(hex_to_square_grid, axis=1, arr=da.cpu().numpy())
    # TODO: think about how to treat the missings
    return np.nan_to_num(images)


def layer_activations_to_decoding_images(la, frame, decoding_cells):
    # only decoding activations
    da = [a[decoding_cells][0, -frame, :, :] for a in la]
    # decoded images
    return np.array([activation_vector_to_image(a) for a in da])


def predictions_and_corrects_from_model_results(outputs_, batch_labels_):
    predictions_ = torch.round(torch.sigmoid(outputs_).squeeze()).detach().cpu().numpy()
    batch_labels_cpu = batch_labels_.detach().cpu().numpy()
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
