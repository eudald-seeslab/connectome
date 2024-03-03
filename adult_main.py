from from_image_to_video import process_directory
from retina_processing import process_retina
from from_retina_to_connectome_funcs import prepare_connectome_input
from train_connectome_model import train_model, model, device


BATCH_SIZE = 10


def main():
    # 1. Convert images to videos
    all_data = []
    all_labels = []

    directories = ["blue", "yellow"]
    for dir_ in directories:
        activations, labels = process_directory(dir_)
        all_data.extend(activations)
        all_labels.extend(labels)

    # 2. Process each directory through the retina model
    decoding_activations = {
        dir_: process_retina(f"videos/{dir_}") for dir_ in directories
    }

    # 3. Prepare the data for the connectome model
    # This step will likely involve merging the outputs for 'blue' and 'yellow' directories
    # and then preparing the tensors for PyTorch Geometric
    data_loader, decision_making_vector = prepare_connectome_input(
        decoding_activations, labels, device, batch_size=BATCH_SIZE
    )

    # 4. Train the connectome model
    train_model(data_loader, model, device_type, device)


if __name__ == "__main__":
    main()
