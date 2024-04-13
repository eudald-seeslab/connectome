import random

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
import yaml

config = yaml.safe_load(open("config.yml"))
images_fraction = config["IMAGES_FRACTION"]
debug = config["DEBUG"]
train_images_dir = config["TRAIN_IMAGES_DIR"]
validation_images_dir = config["VALIDATION_IMAGES_DIR"]

# Define a transform to normalize the data
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),  # resize images to 512x512
        transforms.ToTensor(),  # convert image to PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # normalize image data
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Load images from the "images" directory
dataset = datasets.ImageFolder(train_images_dir, transform=transform)

# Create three sets: train, test and validation
train_size = int(len(dataset) * config["TRAIN_SPLIT"])
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size]
)

validation_dataset = datasets.ImageFolder(validation_images_dir, transform=transform)

# Create a data loader
# I created the training images artificially, and maybe there are too many.
#  Let's try to use a fraction of images to speed up training
# TODO: think about what to do with this
# train_indices = random.sample(
#     range(len(train_dataset)), int(len(train_dataset) * images_fraction)
# )
#train_subset = Subset(train_dataset, train_indices)

train_loader = DataLoader(
    train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=config["BATCH_SIZE"], shuffle=True
)
validation_loader = DataLoader(
    validation_dataset, batch_size=config["BATCH_SIZE"], shuffle=True
)

if debug:
    # Create a data loader for debugging
    debug_indices = random.sample(
        range(len(train_dataset)), int(len(train_dataset) * 0.1)
    )
    debug_subset = Subset(train_dataset, debug_indices)
    debug_loader = DataLoader(
        debug_subset, batch_size=config["BATCH_SIZE"], shuffle=True
    )
else:
    debug_loader = None
