from unittest import TestCase

import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from typing import Dict, Union


class AdultConnectomeNetwork(nn.Module):
    def __init__(
        self,
        adjacency_matrix,
        neuron_count: int,
        general_config: Dict[str, Union[int, float, str, bool]],
    ):
        super(AdultConnectomeNetwork, self).__init__()

        # Convert the adjacency matrix to a PyTorch sparse tensor
        self.adjacency_matrix_csr = adjacency_matrix.tocoo()
        self.connectome_layer_number = general_config["CONNECTOME_LAYER_NUMBER"]

        # Initialize the shared weights for the connectome layers
        self.shared_weights = self.initialize_sparse_weights(
            adjacency_matrix, neuron_count
        )
        self.shared_bias = nn.Parameter(torch.ones(neuron_count))

    @staticmethod
    def initialize_sparse_weights(adjacency_matrix, neuron_count):
        # Convert the adjacency matrix to COO format for easier processing

        # Generate random weights for existing connections
        weights = torch.rand(len(adjacency_matrix.data))

        # Create sparse weights tensor
        indices = torch.LongTensor([adjacency_matrix.row, adjacency_matrix.col])
        sparse_weights = torch.sparse_coo_tensor(
            indices, weights, (neuron_count, neuron_count)
        )

        return nn.Parameter(sparse_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        indices = torch.LongTensor(
            [self.adjacency_matrix_coo.row, self.adjacency_matrix_coo.col]
        )
        values = torch.FloatTensor(self.adjacency_matrix_coo.data)
        shape = torch.Size(self.adjacency_matrix_coo.shape)

        adj_matrix = torch.sparse_coo_tensor(indices, values, shape).to(x.device)

        # Pass the input through the layer with shared weights
        for _ in range(self.connectome_layer_number):
            # Apply the mask from the adjacency matrix to the shared weights
            masked_weights = torch.sparse.mm(adj_matrix, self.shared_weights)

            # Do the forward pass using sparse matrix multiplication
            x = torch.sparse.mm(masked_weights, x.t()).t() + self.shared_bias

        return x


import torch
import torch.nn as nn
from scipy.sparse import csr_matrix


# Example usage of your network
class AdultConnectomeNetworkTest(TestCase):
    def test_forward(self):
        neuron_count = 100
        general_config = {"CONNECTOME_LAYER_NUMBER": 3}
        adjacency_matrix_csr = csr_matrix((neuron_count, neuron_count)).tocoo()

        model = AdultConnectomeNetwork(
            adjacency_matrix_csr, neuron_count, general_config
        )

        # Create a random input tensor
        batch_size = 1
        input_tensor = torch.rand(batch_size, neuron_count)

        # Test the network
        output = model(input_tensor)
        assert output.shape == (batch_size, neuron_count)


class IntegratedModel(nn.Module):
    def __init__(
        self, temporal_conv_net, adult_connectome_net, synapse_df, neuron_type
    ):
        super(IntegratedModel, self).__init__()
        self.temporal_conv_net = temporal_conv_net
        self.adult_connectome_net = adult_connectome_net

        # Filter the DataFrame for the specific neuron type and create a synaptic count matrix
        filtered_df = synapse_df[synapse_df["cell_type"] == neuron_type]
        self.synaptic_matrix = self.create_synaptic_matrix(filtered_df)

    def create_synaptic_matrix(self, df):
        # Assuming 'pre_root_id' and 'post_root_id' are integer indices.
        # If they are not, you'll need to map them to integer indices.

        # Find the maximum index for matrix dimension
        max_index = max(df["pre_root_id"].max(), df["post_root_id"].max()) + 1

        # Create a CSR matrix from the DataFrame
        synaptic_matrix_csr = csr_matrix(
            (df["syn_count"], (df["pre_root_id"], df["post_root_id"])),
            shape=(max_index, max_index),
        )

        # Convert the CSR matrix to a dense PyTorch tensor
        return torch.tensor(synaptic_matrix_csr.toarray(), dtype=torch.float32)

    def forward(self, x):
        # Pass the input through the TemporalConvNet
        x = self.temporal_conv_net(x)

        # Reshape and multiply with the synaptic count matrix
        x = x.view(x.size(0), -1)  # Flatten if necessary
        x = torch.mm(x, self.synaptic_matrix)

        # Pass through the AdultConnectomeNetwork
        return self.adult_connectome_net(x)


if __name__ == "__main__":

    def get_synapse_df():
        classification = pd.read_csv("adult_data/classification.csv")
        connections = pd.read_csv("adult_data/connections.csv")
        return pd.merge(
            connections,
            classification[["root_id", "cell_type"]],
            left_on="pre_root_id",
            right_on="root_id",
        )

    # Example usage of your network
    import pandas as pd
    from model_helpers import TemporalConvNet

    synapse_df = get_synapse_df()

    neuron_type = "TmY18"

    # Create the integrated model
    temporal_conv_net = TemporalConvNet(
        num_inputs, num_channels, num_outputs, kernel_size=2, dropout=0.2
    )
    adult_connectome_net = AdultConnectomeNetwork(
        adjacency_matrix_csr, neuron_count, general_config
    )
    # synapse_df = pd.DataFrame(...)  # Your synapse DataFrame
    # neuron_type = 'TmY18'

    integrated_model = IntegratedModel(
        temporal_conv_net, adult_connectome_net, synapse_df, neuron_type
    )

    # Test the integrated model
    batch_size = 1
    hexal_size = 721  # Example hexal size
    temporal_size = 84  # Example temporal size
    input_tensor = torch.rand(batch_size, hexal_size, temporal_size)

    output = integrated_model(input_tensor)
    print(output.shape)
