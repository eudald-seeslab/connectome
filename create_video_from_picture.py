from pathlib import Path

import numpy as np
import os
import glob
import cv2

# Constants
N_BLACKS = 1
N_IMAGES = 2
BLACK_SQUARE = np.zeros((512, 512, 3), dtype=np.uint8)
SEQUENCE_DIRECTORY = "videos"
IMAGE_DIRECTORY = os.path.join("", "images", "random_set")

# Functions
def repeat_image(image_, times):
    return np.array([image_ for _ in range(times)])


def create_sequence_from_image(input_image):
    return np.concatenate((repeat_image(BLACK_SQUARE, N_BLACKS), repeat_image(input_image, N_IMAGES),
                           repeat_image(BLACK_SQUARE, N_BLACKS)), axis=0)


def process_directory(directory):

    target_dir = os.path.join(SEQUENCE_DIRECTORY, directory)
    # create the target dir if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    for image_path in glob.glob(os.path.join(IMAGE_DIRECTORY, directory, "*.png")):
        image = cv2.imread(image_path)
        sequence = create_sequence_from_image(image)

        npy_filename = os.path.join(target_dir,
                                    os.path.basename(image_path).split('.')[0] + "_sequence.npy")
        np.save(npy_filename, sequence)


# Main processing
directories = ["blue", "yellow"]
for dir_ in directories:
    process_directory(dir_)
