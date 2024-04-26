import torch
from torch import cuda, device

from flyvision_ans import FINAL_CELLS, DECODING_CELLS

# Check for CUDA availability and set device
device_type = "cuda" if cuda.is_available() else "cpu"
device_type = "cpu"
DEVICE = device(device_type)

# Constants for torch
dtype = torch.float32
sparse_layout = torch.sparse_coo

# Directory paths relative to the project root
TRAINING_DATA_DIR = "images/easy_v2"
TESTING_DATA_DIR = "images/easy_images"
VALIDATION_DATA_DIR = "images/easyval_images"

# Debugging and logging
debugging = True
debug_length = 2
validation_length = 50
wandb_ = False
wandb_images_every = 100
small = True
small_length = 400

# Training configuration
num_epochs = 1
batch_size = 1
dropout = 0.1
base_lr = 0.00001
weight_decay = 0.0001
NUM_CONNECTOME_PASSES = 4
normalize_voronoi_cells = True
log_transform_weights = True

# Data specs
dt = 1 / 100
last_good_frame = 2
final_retina_cells = FINAL_CELLS # or DECODING_CELLS
cell_type_plot = "TmY18"
