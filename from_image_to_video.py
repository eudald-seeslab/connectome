from pathlib import Path

import numpy as np
import os
import glob
import cv2
from tqdm import tqdm

# Constants
N_BLACKS = 1
N_IMAGES = 4
BLACK_SQUARE = np.zeros((512, 512, 3), dtype=np.uint8)
SEQUENCE_DIRECTORY = "easy_videosv2"
IMAGE_DIRECTORY = os.path.join("", "images", "easy_v2")


# Functions
def repeat_image(image_, times):
    return np.array([image_ for _ in range(times)])


def create_sequence_from_image(input_image):
    return np.concatenate(
        (
            repeat_image(BLACK_SQUARE, N_BLACKS),
            repeat_image(input_image, N_IMAGES),
            repeat_image(BLACK_SQUARE, N_BLACKS),
        ),
        axis=0,
    )


def process_directory(directory):

    target_dir = os.path.join(SEQUENCE_DIRECTORY, directory)
    # create the target dir if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(
        glob.glob(os.path.join(IMAGE_DIRECTORY, directory, "*.png"))
    ):
        image = cv2.imread(image_path)
        sequence = create_sequence_from_image(image)

        npy_filename = os.path.join(
            target_dir, os.path.basename(image_path).split(".")[0] + "_sequence.npy"
        )
        np.save(npy_filename, sequence)


def image_paths_to_sequences(image_path_list):
    sequences = []
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        sequences.append(create_sequence_from_image(image))

    return np.array([np.mean(a, axis=3) for a in sequences])


def main():
    directories = ["blue", "yellow"]
    for dir_ in directories:
        process_directory(dir_)


if __name__ == "__main__":
    main()
    print("done")
