import os
import torch
from torch import cuda, device
from torch.nn.functional import leaky_relu
from PIL import Image
from debug_utils import debugger_is_active

# Check for CUDA availability and set device
device_type = "cuda" if cuda.is_available() else "cpu"
# device_type = "cpu"
DEVICE = device(device_type)

# Directory paths relative to the project root
TRAINING_DATA_DIR = "images/two_shapes/train"
TESTING_DATA_DIR = "images/two_shapes/test"
# get directory names from the training data directory
CLASSES = sorted(os.listdir(TRAINING_DATA_DIR))
# get one sample of one class to get the image size
sample_image = os.listdir(os.path.join(TRAINING_DATA_DIR, CLASSES[0]))[0]
image_size = Image.open(os.path.join(TRAINING_DATA_DIR, CLASSES[0], sample_image)).size[0]

# Models
# None if you want to start from scratch
resume_checkpoint = None # "model_2024-05-19 16:16:58.pth"

# Neural data
neurons = "all"  # "selected" or "all"
voronoi_criteria = "R7"  #  "R7" or "all"
random_synapses = False
train_edges = True
train_neurons = False
final_layer = "mean"  # "mean" or "nn"
# node embedding activation function, as in
# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html
# only for training neurons
lambda_func = leaky_relu  # torch activation function
# Shut off some neurons based on their cell_type
# You can find all the cell types in the adult_data/cell_types.csv
filtered_celltypes = []
# You can also filter a fraction of the neurons
# Note: it's a fraction of the number of neurons after filtering by cell type
# and also after removing the protected cell types (R1-6, R7, R8, and the rational cell types)
# None if you don't want to filter
filtered_fraction = None
# Updated synaptic data taking into account the excitatory or inhibitory nature of the synapse
refined_synaptic_data = False
# droputs: there is a dropout for the neuron activations to simulate that, for some reason 
#  (oscillations, the neuron having fired too recently, etc) the neuron does not fire
neuron_dropout = 0.1
# decision dropout: there is a dropout for the decision making vector to simulate that
#  the decision making process also has some neurons not available all the time
decision_dropout = 0.2

# Debugging and logging
debugging = False
debug_length = 2
small_length = None
validation_length = 400
wandb_ = True
wandb_images_every = 400
wandb_group = "weber"

# Training configuration
num_epochs = 50
batch_size = 32
base_lr = 0.001
NUM_CONNECTOME_PASSES = 4
log_transform_weights = False
eye = "right"  # "left" or "right"
# "radius", "contingency", "distance", "point_num", "stripes", "weber", "colour"
# if empty, I will try to guess the plots from the classes
# If None, no plots will be generated
plot_types = []

# sparse stuff is generaly not implemented in half
dtype = torch.float32
sparse_layout = torch.sparse_coo

# small checks so that i don't screw up
wandb_ = False if debugger_is_active() else wandb_
wandb_ = False if debugging else wandb_
validation_length = validation_length if small_length is not None else None
num_epochs = 1 if debugging else num_epochs
