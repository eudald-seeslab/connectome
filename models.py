import torch.nn.functional as F
import torch.nn as nn
from torch import Size, Tensor, where, from_numpy, rand, transpose, matmul, ones
from torchvision import models

# Import model config manager
from model_config_manager import PRETRAINED_MODELS
from model_helpers import (
    create_retina_layer,
    create_rational_layer,
    get_retina_output_shape,
)
from model_config import ModelConfig
from pandas.core.frame import DataFrame
from typing import Dict, Union


class CombinedModel(nn.Module):
    def __init__(
        self,
        adjacency_matrix: DataFrame,
        neurons: DataFrame,
        model_config: ModelConfig,
        general_config: Dict[str, Union[int, float, str, bool]],
    ) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.retina_model(x)
        x = x.view(x.size(0), -1)
        x = self.connectome_model(x)
        # TODO: explore whether we can mask already the softmax, so we can skip
        #  the rational layer
        return F.log_softmax(x, dim=1)


class ConnectomeNetwork(nn.Module):
    def __init__(
        self,
        adjacency_matrix: DataFrame,
        nodes: DataFrame,
        retina_output_size: Size,
        general_config: Dict[str, Union[int, float, str, bool]],
    ) -> None:
        super(ConnectomeNetwork, self).__init__()

        self.connectome_layer_number = general_config["CONNECTOME_LAYER_NUMBER"]

        neuron_count = nodes.shape[0]

        # Create a linear layer for the retina input
        self.retina_layer = create_retina_layer(
            retina_output_size[1], neuron_count, nodes[nodes["visual"]].index
        )

        # These are the shared weights for the connectome layers: Set weights
        #  for non-connected neurons to 0 and initialize the rest randomly
        self.shared_weights = nn.Parameter(
            from_numpy(adjacency_matrix.values).float() * rand(neuron_count, neuron_count)
        )
        self.shared_bias = nn.Parameter(ones(neuron_count))

        self.rational_layer = create_rational_layer(
            neuron_count, 2, nodes[nodes["rational"]].index
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pass the input through the retina layer
        out = self.retina_layer(x)

        # Pass the input through the layer with shared weights
        for _ in range(self.connectome_layer_number):
            # Set the weights for all layers to be the same and do the forward pass
            out = matmul(self.shared_weights, out.t()).t() + self.shared_bias

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
                padding=model_config.kernel_padding,
                stride=model_config.kernel_stride,
            )
            batch_norm = nn.BatchNorm2d(model_config.out_channels)
            activation = nn.ReLU(inplace=True)
            pooling_layer = nn.MaxPool2d(
                kernel_size=model_config.pool_kernel_size,
                stride=model_config.pool_stride,
            )
            dropout = nn.Dropout(p=model_config.dropout)

            self.layers.append(
                nn.Sequential(
                    conv_layer, batch_norm, activation, pooling_layer, dropout
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_retina_model(model_config: ModelConfig):
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
