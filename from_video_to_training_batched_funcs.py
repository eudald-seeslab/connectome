import os
import random

import numpy as np


def get_files_from_directory(directory_path):
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(".npy"):
                files.append(os.path.join(root, filename))
    return files


def select_random_videos(all_files, batch_size, already_selected):
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
    x = name.split("/")[2]
    if x == "yellow":
        return 1
    if x == "blue":
        return 0
    return np.nan


def paths_to_labels(paths):
    return [get_label(a) for a in paths]


def load_custom_sequences(video_paths):
    videos = [np.load(a) for a in video_paths]
    # Assuming videos are numpy arrays with shape (n_frames, height, width, channels)
    return np.array([np.mean(a, axis=3) for a in videos])
