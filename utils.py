import json
import os
import random
import numpy as np
import pandas as pd
import sys
from scipy.sparse import coo_matrix
import torch
import config


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def get_files_from_directory(directory_path):
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith((".npy", ".png", ".jpg", ".jpeg")):
                files.append(os.path.join(root, filename))
    return files


def get_image_paths(images_dir, small, small_length):
    images = get_files_from_directory(images_dir)
    assert len(images) > 0, f"No images found in {images_dir}."

    if small:
        try:
            images = random.sample(images, small_length)
        except ValueError:
            print(
                f"Not enough videos in {images_dir} to sample {small_length}."
                f"Continuing with {len(images)}."
            )

    return images


def synapses_to_matrix_and_dict(synapses):
    # Unique root_ids in synapse_df (both pre and post)
    neurons_synapse_pre = pd.unique(synapses["pre_root_id"])
    neurons_synapse_post = pd.unique(synapses["post_root_id"])
    all_neurons = np.unique(np.concatenate([neurons_synapse_pre, neurons_synapse_post]))

    # Map neuron root_ids to matrix indices
    root_id_to_index = {root_id: index for index, root_id in enumerate(all_neurons)}

    # Convert root_ids in filtered_synapse_df to matrix indices
    pre_indices = synapses["pre_root_id"].map(root_id_to_index).values
    post_indices = synapses["post_root_id"].map(root_id_to_index).values

    # Use syn_count as the data for the non-zero elements of the matrix
    data = synapses["syn_count"].values

    # Create the sparse matrix
    matrix = coo_matrix(
        (data, (pre_indices, post_indices)),
        shape=(len(all_neurons), len(all_neurons)),
        dtype=np.int64,
    )

    return matrix, root_id_to_index


def get_iteration_number(im_num, batch_size):
    if config.debugging:
        return config.debug_length
    if config.small and im_num > config.small_length:
        return config.small_length // batch_size
    return im_num // batch_size


def get_label(name):
    x = os.path.basename(os.path.dirname(name))
    try:
        return config.CLASSES.index(x)
    except ValueError:
        raise ValueError(f"Unexpected directory label {x}")


def paths_to_labels(paths):
    return [get_label(a) for a in paths]


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


def compute_accuracy(probabilities, labels):

    # Convert probabilities to binary predictions
    predictions = (probabilities > 0.5).float()

    # Calculate accuracy
    return np.where(predictions == labels, 1, 0).float().mean()


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
                    "Model outputs": list(outputs_),
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


def clean_model_outputs(outputs_, batch_labels_):
    probabilities_ = torch.softmax(outputs_.detach().cpu().float(), dim=1).numpy()
    predictions_ = np.argmax(probabilities_, axis=1)
    batch_labels_cpu = batch_labels_.detach().cpu().numpy()
    correct_ = np.where(predictions_ == batch_labels_cpu, 1, 0)

    return probabilities_, predictions_, batch_labels_cpu, correct_


def save_model(model_, optimizer_, model_name, config_, sweep_config_):
    # create 'models' directory if it doesn't exist
    path_ = os.path.join(os.getcwd(), "models")
    os.makedirs(path_, exist_ok=True)
    torch.save(
        {"model": model_.state_dict(), "optimizer": optimizer_.state_dict()},
        os.path.join(path_, model_name),
    )
    # create an accompanying config file
    config_dict = {
        key: str(value)
        for key, value in config_.__dict__.items()
        if not key.startswith("__") and not "module" in str(value)
    }
    if sweep_config_ is not None:
        config_dict.update(sweep_config_)

    config_path = os.path.join(path_, f"{model_name}_config.txt")
    with open(config_path, "w", encoding='utf-8') as f: 
        json.dump(config_dict, f, ensure_ascii=False, indent=4)

def guess_your_plots():
    classes = config.CLASSES
    # if there is a colour class, it's either weber or colour. One of the plots will be 
    #  useless, but it won't crash, just don't look at it
    if any([x in classes for x in ["blue", "yellow", "green", "red"]]):
        return ["weber", "colour"]
    # if there are geometry classes, it's radius, distance and contingency
    if any([x in classes for x in ["circle", "square", "triangle", "star"]]):
        return ["radius", "distance", "contingency"]
    # if there are numbers bigger than 10 in the classes, they will be angles, so it's stripes
    if any([int(x) > 10 for x in classes]):
        return ["stripes"]
    # if there are numbers smaller than 10, it's guess the number of points
    if any([int(x) < 10 for x in classes]):
        return ["point_num"]
    return []
