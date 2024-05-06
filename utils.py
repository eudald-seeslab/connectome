import os
import random
import numpy as np
import pandas as pd
import sys
from scipy.sparse import coo_matrix
import config
from plots import plot_accuracy_per_value, plot_weber_fraction


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def plot_results(results_, plot_types):
    plots = []
    try:
        for plot_type in plot_types:
            if plot_type == "weber":
                plots.append([plot_weber_fraction(results_)])
            elif plot_type == "radius":
                plots.append(plot_accuracy_per_value(results_, "radius"))
            elif plot_type == "distance":
                plots.append(plot_accuracy_per_value(results_, "distance"))
    except Exception as e:
        print(f"Error plotting results: {e}")

    return plots


def get_files_from_directory(directory_path):
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(".npy") or filename.endswith(".png"):
                files.append(os.path.join(root, filename))
    return files


def get_image_paths(images_dir, small, small_length):
    images = get_files_from_directory(images_dir)
    assert len(images) > 0, f"No videos found in {images_dir}."

    if small:
        try:
            images = random.sample(images, small_length)
        except ValueError:
            print(
                f"Not enough videos in {images_dir} to sample {small_length}."
                f"Continuing with {len(images)}."
            )

    return images


def synapses_to_matrix_and_dict(right_synapses):
    # Unique root_ids in synapse_df (both pre and post)
    neurons_synapse_pre = pd.unique(right_synapses["pre_root_id"])
    neurons_synapse_post = pd.unique(right_synapses["post_root_id"])
    all_neurons = np.unique(np.concatenate([neurons_synapse_pre, neurons_synapse_post]))

    # Map neuron root_ids to matrix indices
    root_id_to_index = {root_id: index for index, root_id in enumerate(all_neurons)}

    # Convert root_ids in filtered_synapse_df to matrix indices
    pre_indices = right_synapses["pre_root_id"].map(root_id_to_index).values
    post_indices = right_synapses["post_root_id"].map(root_id_to_index).values

    # Use syn_count as the data for the non-zero elements of the matrix
    data = right_synapses["syn_count"].values

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
    if config.small:
        return config.small_length // batch_size
    return im_num // batch_size
