import os
import torch
from torch import cuda, device
from utils import debugger_is_active

# Check for CUDA availability and set device
device_type = "cuda" if cuda.is_available() else "cpu"
# device_type = "cpu"
DEVICE = device(device_type)

# Directory paths relative to the project root
TRAINING_DATA_DIR = "images/five_to_fifteen/train"
TESTING_DATA_DIR = "images/five_to_fifteen/test"
# not used
VALIDATION_DATA_DIR = "images/big_pointsval"
# get directory names from the training data directory
CLASSES = sorted(os.listdir(TRAINING_DATA_DIR))

# Neural data
neurons = "all" # "selected" or "all"
voronoi_criteria = "R7" #  "R7" or "all"
random_synapses = False


# Debugging and logging
debugging = False
debug_length = 2
validation_length = 400
wandb_ = True
wandb_images_every = 20
small = False
small_length = 4000

# Training configuration
num_epochs = 20
batch_size = 32
dropout = 0.1
base_lr = 0.01
weight_decay = 0.0001
NUM_CONNECTOME_PASSES = 4
log_transform_weights = False
plot_types = ["weber"]  # "radius", "contingency", "distance" or "weber"

# sparse stuff is generaly not implemented in half...
dtype = torch.float32
sparse_layout = torch.sparse_coo

# small checks so that i don't screw up
wandb_ = False if debugger_is_active() else wandb_
wandb_ = False if debugging else wandb_
validation_length = validation_length if small else None
