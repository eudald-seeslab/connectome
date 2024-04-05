import numpy as np
import os
import glob
import cv2
import torch
from tqdm import tqdm

from flyvision_ans import (
    DECODING_CELLS,
    SerializedResponseProcessor,
)

# Constants for video creation
N_BLACKS = 1
N_IMAGES = 2
BLACK_SQUARE = np.zeros((512, 512, 3), dtype=np.uint8)
IMAGE_DIRECTORY = os.path.join("images")


# Function to repeat image N times
def repeat_image(image_, times):
    return np.array([image_ for _ in range(times)])


# Function to create a sequence from a single image
def create_sequence_from_image(input_image):
    return np.concatenate(
        (
            repeat_image(BLACK_SQUARE, N_BLACKS),
            repeat_image(input_image, N_IMAGES),
            repeat_image(BLACK_SQUARE, N_BLACKS),
        ),
        axis=0,
    )


# Function to process a single image through the retina model and associate it with a label
def process_image_through_retina(image_path, label):
    image = cv2.imread(image_path)
    sequence = create_sequence_from_image(image)

    # Process the sequence through the retina model
    response_processor = SerializedResponseProcessor(extent=15, kernel_size=13)
    with torch.no_grad():
        layer_activations = response_processor.compute_layer_activations(sequence)

    decoding_activations = []
    for layer in layer_activations:
        decoding_activations.append({cell: layer[cell] for cell in DECODING_CELLS})

    return decoding_activations, label


# Main function to process all images in a directory and keep track of labels
def process_directory(directory):
    label = 1 if directory == "blue" else 0
    all_activations = []
    all_labels = []

    for image_path in tqdm(
        glob.glob(os.path.join(IMAGE_DIRECTORY, directory, "*.png"))
    ):
        decoding_activations, image_label = process_image_through_retina(
            image_path, label
        )
        all_activations.append(decoding_activations)
        all_labels.append(image_label)

        del decoding_activations
        torch.cuda.empty_cache()

    return all_activations, all_labels


if __name__ == "__main__":
    # Example usage for "blue" and "yellow" directories
    all_data = []
    all_labels = []

    directories = ["blue", "yellow"]
    for dir_ in directories:
        activations, labels = process_directory(dir_)
        all_data.extend(activations)
        all_labels.extend(labels)
