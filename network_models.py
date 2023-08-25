import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch import where, from_numpy, rand, randn, transpose
from torchvision import models

# Import config
import yaml

# Import model config manager
from model_config_manager import ModelConfigManager

# TODO: this shouldn't be here
config = yaml.safe_load(open("config.yml"))
SOFTMAX_SIZE = config["SOFTMAX_SIZE"]
LAYER_NUM = config["LAYER_NUM"]
IMAGE_SIZE = config["IMAGE_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]
RETINA_MODEL = config["RETINA_MODEL"]


def retina_weight_mask(input_size, output_size, mask):
    connections = np.full((output_size, input_size), False)
    connections[mask, :] = True
    return nn.Parameter(
        where(
            from_numpy(connections),
            rand(output_size, input_size),
            0,
        )
    )


def rational_weight_mask(input_size, output_size, mask):
    connections = np.full((output_size, input_size), False)
    # Note that the mask is in the columns now
    connections[:, mask] = True
    return nn.Parameter(
        where(
            from_numpy(connections),
            rand(output_size, input_size),
            0,
        )
    )


def create_retina_layer(row_size, col_size, mask):
    layer = nn.Linear(row_size, col_size)
    del layer.weight

    layer.weight = retina_weight_mask(row_size, col_size, mask)
    return layer


def create_rational_layer(row_size, col_size, mask):
    layer = nn.Linear(row_size, col_size)
    del layer.weight

    layer.weight = rational_weight_mask(row_size, col_size, mask)
    return layer


class ConnectomeNetwork(nn.Module):
    def __init__(self, adjacency_matrix, nodes, retina_output_size):
        super(ConnectomeNetwork, self).__init__()

        self.neuron_count = nodes.shape[0]
        visual_indices = nodes[nodes["visual"]].index

        # Create a linear layer for the retina input
        self.retina_layer = create_retina_layer(
            retina_output_size[1], self.neuron_count, visual_indices
        )

        # Create a dictionary that packs the neuron layers, which are equivalent
        #  to signals advancing through the connectome
        self.neuron_layer_dict = {}
        for i in range(LAYER_NUM):
            self.neuron_layer_dict[i] = nn.Linear(
                self.neuron_count, self.neuron_count, bias=False
            )
            # Remove the weights as we override them in the forward
            # so that they don't show up when calling .parameters()
            del self.neuron_layer_dict[i].weight

        # Set the weights for non-connected neurons to 0 and initialize the rest randomly
        # TODO: make sure this is correct and the transpose is needed
        self.shared_weights = nn.Parameter(
            where(
                transpose(from_numpy(adjacency_matrix.values == 0), 0, 1),
                0,
                rand(self.neuron_count, self.neuron_count),
            )
        )
        rational_indices = nodes[nodes["rational"]].index
        self.rational_layer = create_rational_layer(
            self.neuron_count, SOFTMAX_SIZE, rational_indices
        )

    def forward(self, x):
        # Pass the input through the retina layer
        out = self.retina_layer(x)

        # Pass the input through the layer with shared weights
        for _, layer in self.neuron_layer_dict.items():
            # Set the weights for all layers to be the same
            layer.weight = self.shared_weights
            # Then do the forward pass
            out = layer(out)

        # Pass the input through the rational layer
        return self.rational_layer(out)


class RetinaModel(nn.Module):
    def __init__(self, config):
        super(RetinaModel, self).__init__()
        # Define the single convolutional layer
        self.conv_layer = nn.Conv2d(
            in_channels=3,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            padding=config.padding,
            stride=config.stride,
        )
        self.activation = nn.ReLU(inplace=True)
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply the convolutional layer
        x = self.conv_layer(x)
        # Apply the activation function
        x = self.activation(x)
        return x


def get_retina_model(model_config):
    if model_config.model_name == "vgg16":
        # Load the pre-trained MobileNetV2 model and remove the last fully connected layer
        # retina_model = hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        retina_model = models.vgg16(pretrained=True)
        retina_model = nn.Sequential(*list(retina_model.children())[:-1])
        retina_model.eval()

        return retina_model
    else:
        return RetinaModel(model_config)


def get_retina_output_shape(model, input_shape):
    # Get a sample input
    sample_input = randn(*input_shape)

    # Pass the sample input through the model
    output = model.retina_model(sample_input)

    output = output.view(output.size(0), -1)

    # Return the output shape
    return output.size()


# Define the combined model
class CombinedModel(nn.Module):
    def __init__(self, adjacency_matrix, neurons, model_config):
        super().__init__()
        self.retina_model = get_retina_model(model_config=model_config)

        # Get the output shape of the retina model
        retina_output_shape = get_retina_output_shape(
            self, (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        )
        self.connectome_model = ConnectomeNetwork(
            adjacency_matrix, neurons, retina_output_shape
        )

    def forward(self, x):
        x = self.retina_model(x)
        x = x.view(x.size(0), -1)
        x = self.connectome_model(x)
        # TODO: explore whether we can mask already the softmax, so we can skip
        #  the rational layer
        return F.log_softmax(x, dim=1)
