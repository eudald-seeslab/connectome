import torch.nn.functional as F
import torch.nn as nn
from torch import where, from_numpy, rand, transpose
from torchvision import models

# Import config
import yaml

# Import model config manager
from model_config_manager import PRETRAINED_MODELS
from model_helpers import (
    create_retina_layer,
    create_rational_layer,
    get_retina_output_shape,
)


class CombinedModel(nn.Module):
    def __init__(self, adjacency_matrix, neurons, model_config, general_config):
        super().__init__()
        self.retina_model = get_retina_model(model_config=model_config)
        batch_size = general_config["BATCH_SIZE"]
        image_size = general_config["IMAGE_SIZE"]

        # Get the output shape of the retina model
        retina_output_shape = get_retina_output_shape(
            self, (batch_size, 3, image_size, image_size)
        )
        self.connectome_model = ConnectomeNetwork(
            adjacency_matrix, neurons, retina_output_shape, general_config
        )

    def forward(self, x):
        x = self.retina_model(x)
        x = x.view(x.size(0), -1)
        x = self.connectome_model(x)
        # TODO: explore whether we can mask already the softmax, so we can skip
        #  the rational layer
        return F.log_softmax(x, dim=1)


class ConnectomeNetwork(nn.Module):
    def __init__(self, adjacency_matrix, nodes, retina_output_size, general_config):
        super(ConnectomeNetwork, self).__init__()

        connectome_layer_number = general_config["CONNECTOME_LAYER_NUMBER"]

        self.neuron_count = nodes.shape[0]
        visual_indices = nodes[nodes["visual"]].index

        # Create a linear layer for the retina input
        self.retina_layer = create_retina_layer(
            retina_output_size[1], self.neuron_count, visual_indices
        )

        # Create a dictionary that packs the neuron layers, which are equivalent
        #  to signals advancing through the connectome
        self.neuron_layer_dict = {}
        for i in range(connectome_layer_number):
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
            self.neuron_count, 2, rational_indices
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


class CustomRetinaModel(nn.Module):
    def __init__(self, model_config):
        super(CustomRetinaModel, self).__init__()

        self.num_layers = model_config.num_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            # TODO: all layers have to be the same; we might want more flexibility
            conv_layer = nn.Conv2d(
                # The first layer has 3 input channels coming from an RGB image
                in_channels=3 if i == 0 else model_config.out_channels,
                out_channels=model_config.out_channels,
                kernel_size=model_config.kernel_size,
                padding=model_config.padding,
                stride=model_config.stride,
            )
            activation = nn.ReLU(inplace=True)
            pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

            self.layers.append(nn.Sequential(conv_layer, activation, pooling_layer))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_retina_model(model_config):
    # Pretrained models
    if model_config.model_name in PRETRAINED_MODELS:
        # Load a pretrained model from torchvision
        retina_model = models.get_model(model_config.model_name, pretrained=True)
        # Remove the last layer, since we will implement our own softmax
        retina_model = nn.Sequential(*list(retina_model.children())[:-1])
        retina_model.eval()

        return retina_model
    # Our models
    else:
        return CustomRetinaModel(model_config)
