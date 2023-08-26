import numpy as np
from torch import nn as nn, where, from_numpy, rand, randn


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


def get_retina_output_shape(model, input_shape):
    # Get a sample input
    sample_input = randn(*input_shape)

    # Pass the sample input through the model
    output = model.retina_model(sample_input)

    output = output.view(output.size(0), -1)

    # Return the output shape
    return output.size()
