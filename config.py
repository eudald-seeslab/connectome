import os
import torch
from torch import cuda, device
from torch.nn.functional import leaky_relu
from PIL import Image
from debug_utils import debugger_is_active

# Data
data_type = "one_to_ten"
TRAINING_DATA_DIR = os.path.join("images", data_type, "train")
TESTING_DATA_DIR = os.path.join("images", data_type, "test")
CLASSES = sorted(os.listdir(TRAINING_DATA_DIR))
# get one sample of one class to get the image size
sample_image = os.listdir(os.path.join(TRAINING_DATA_DIR, CLASSES[0]))[0]
image_size = Image.open(os.path.join(TRAINING_DATA_DIR, CLASSES[0], sample_image)).size[0]

# Training configuration
num_epochs = 100
batch_size = 16
base_lr = 0.0003
patience = 2

# Checkpoint
# None if you want to start from scratch
resume_checkpoint = None # "m_2024-07-10 18:08_1tn2z4xj.pth"
save_every_checkpoint = False

# Model architecture and biological parameters
NUM_CONNECTOME_PASSES = 3
neurons = "all"  # "selected" or "all"
voronoi_criteria = "R7"  #  "R7" or "all"
random_synapses = False
train_edges = False
train_neurons = True
final_layer = "mean"  # "mean" or "nn"
# Some papers use a subset of neurons to compute the final decision (e.g. https://www-science.org/doi/full/10.1126/sciadv.abq7592)
# If None, all neurons are used
num_decision_making_neurons = None  # None or a number
eye = "right"  # "left" or "right"
# node embedding activation function, as in
# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html
# only for training neurons
lambda_func = leaky_relu  # any torch activation function
# we need to normalize the output of the connectome to avoid exploding gradients
# before applying the activation function (above)
neuron_normalization = "min_max"  # "log1p" or "min_max"
# Shut off some neurons based on their cell_type
# You can find all the cell types in adult_data/cell_types.csv
filtered_celltypes = []
# You can also filter a fraction of the neurons
# Note: it's a fraction of the number of neurons after filtering by cell type
# and also after removing the protected cell types (R1-6, R7, R8, and the rational cell types)
# None if you don't want to filter
filtered_fraction = None
# Updated synaptic data taking into account the excitatory or inhibitory nature of the synapse
refined_synaptic_data = False
# Do you want to clip the weights of the connectome, so that they are between 0 and 1?
synaptic_limit = True
# droputs: there is a dropout for the neuron activations to simulate that, for some reason
#  (oscillations, the neuron having fired too recently, etc) the neuron does not fire
neuron_dropout = 0
# decision dropout: there is a dropout for the decision making vector to simulate that
#  the decision making process also has some neurons not available all the time
decision_dropout = 0
# This is to avoid exploding gradients, but I'm not sure it's a great idea, and there are other ways of doing it
log_transform_weights = False
# According to the literature (see https://www.cell.com/cell/fulltext/S0092-8674(17)31498-8), in the fly's retina,
#  the R7 and R8 neurons inhibit each other. Set to true if you want to simulate this behaviour
inhibitory_r7_r8 = False
# As of October 2024, there is a new version of the connectome, do you want to use it?
new_connectome = True
# You can choose the cell types used to compute the final decision
# If None, the ones in adult_data/rational_cell_types.csv will be used
cluster1 = ["Tm16", "Mi4", "Pm13", "TmY10", "TmY11", "Pm14", "Li11", "Tm36", "MLt2", "Sm04", "Sm32", "Li03", "Li13", "Li23", "Li28", "Li27", "Tm7", "LLPt", "Li32", "Li29", 
            "Tm5f", "Mlt5", "Tm34", "Sm13", "Tm7", "Li02", "Li30", "Li05", "Li06", "Tm35", "Tm8b", "Li07", "Li04", "Li33", "TmY31", "Tm5c", "Tm20", "Tm33", "Li01", "Tm37", "Tm8a", "Li10",
            "Tm5d", "Tm32", "Tm31", "Tm5b", "Tm5a", "Sm31", "Li09", "Li12"]
original_rational = ["KCapbp-m", "KCapbp-ap2", "KCapbp-ap1"]
rational_cell_types = original_rational

# CUDA stuff
device_type = "cuda" if cuda.is_available() else "cpu"
# device_type = "cpu"
DEVICE = device(device_type)
# Random seed (it can be set to None)
randdom_seed = 1714

# Debugging and logging
debugging = False
debug_length = 2
small_length = None
validation_length = 400
wandb_ = True
wandb_images_every = 400
wandb_project = "no_synaptic_limit"
wandb_group = data_type     # you can put something else here

# Plots
# "radius", "contingency", "distance", "point_num", "stripes", "weber", "colour"
# if empty list, I will try to guess the plots from the classes
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
