import torch
from torch import cuda, device

from flyvision_ans import FINAL_CELLS, DECODING_CELLS
from utils import debugger_is_active

# Check for CUDA availability and set device
device_type = "cuda" if cuda.is_available() else "cpu"
# device_type = "cpu"
DEVICE = device(device_type)

# sparse stuff is generaly not implemented in half...
dtype = torch.float32
sparse_layout = torch.sparse_coo

# Directory paths relative to the project root
TRAINING_DATA_DIR = "images/super_easy"
TESTING_DATA_DIR = "images/big_pointsval"
VALIDATION_DATA_DIR = "images/big_pointsval"

# Debugging and logging
debugging = True
debug_length = 2
validation_length = 400
wandb_ = True
wandb_images_every = 100
small = True
small_length = 2400

# Training configuration
num_epochs = 1
batch_size = 32
dropout = 0.1
base_lr = 0.005
weight_decay = 0.0001
NUM_CONNECTOME_PASSES = 5
normalize_voronoi_cells = True
log_transform_weights = False

# Data specs
dt = 1 / 100
last_good_frame = 2
final_retina_cells = FINAL_CELLS # or DECODING_CELLS
cell_type_plot = "TmY18"

# small checks so that i don't screw up
wandb_ = False if debugger_is_active() else wandb_
wandb_ = False if debugging else wandb_
validation_length = validation_length if small else None
