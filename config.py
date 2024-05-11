import torch
from torch import cuda, device
from utils import debugger_is_active

# Check for CUDA availability and set device
device_type = "cuda" if cuda.is_available() else "cpu"
# device_type = "cpu"
DEVICE = device(device_type)

# Directory paths relative to the project root
TRAINING_DATA_DIR = "images/arthropods/train"
TESTING_DATA_DIR = "images/arthropods/test"
# not used
VALIDATION_DATA_DIR = "images/big_pointsval"

# Neural data
neurons = "selected" # or "selected"
voronoi_criteria = "all" # or R7
random_synapses = False

# Data
SHAPE = "square"
TRAIN_NUM = 500
TEST_NUM = 100
MIN_RADIUS = 80
MAX_RADIUS = 110
JITTER = False

# Debugging and logging
debugging = True
debug_length = 2
validation_length = 400
wandb_ = True
wandb_images_every = 20
small = False
small_length = 4000

# Training configuration
num_epochs = 1
batch_size = 32
dropout = 0.1
base_lr = 0.00005
weight_decay = 0.0001
NUM_CONNECTOME_PASSES = 5
log_transform_weights = False
plot_types = []  # "radius", "distance" or "weber"

# sparse stuff is generaly not implemented in half...
dtype = torch.float32
sparse_layout = torch.sparse_coo

# small checks so that i don't screw up
wandb_ = False if debugger_is_active() else wandb_
wandb_ = False if debugging else wandb_
validation_length = validation_length if small else None
