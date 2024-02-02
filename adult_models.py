from typing import Dict, Union
from unittest import TestCase

import torch
import torch.nn as nn

from adult_models_helpers import get_synapse_df


class AdultConnectomeNetwork(nn.Module):
    def __init__(
        self,
        adjacency_matrix,
        neuron_count: int,
        general_config: Dict[str, Union[int, float, str, bool]],
    ):
        super(AdultConnectomeNetwork, self).__init__()

        # Convert the adjacency matrix to a PyTorch sparse tensor once in the initialization
        self.adjacency_matrix_coo = adjacency_matrix.tocoo()
        self.adj_matrix_sparse = torch.sparse_coo_tensor(
            torch.tensor(
                [self.adjacency_matrix_coo.row, self.adjacency_matrix_coo.col]
            ),
            torch.FloatTensor(self.adjacency_matrix_coo.data),
            torch.Size(self.adjacency_matrix_coo.shape),
            device="cuda",
        )

        self.connectome_layer_number = general_config["CONNECTOME_LAYER_NUMBER"]

        # Initialize the shared weights for the connectome layers
        self.shared_weights = self.initialize_sparse_weights(
            adjacency_matrix, neuron_count
        )
        self.shared_bias = nn.Parameter(torch.ones(neuron_count))

    @staticmethod
    def initialize_sparse_weights(adjacency_matrix, neuron_count):
        # Generate random weights for existing connections, ensuring the tensor is on the same device
        weights = torch.rand(
            len(adjacency_matrix.data), device="cuda"
        )  # Specify device here

        # Create sparse weights tensor, ensuring indices are on the same device
        indices = torch.tensor(
            [adjacency_matrix.row, adjacency_matrix.col], device="cuda"
        )  # Specify device here
        sparse_weights = torch.sparse_coo_tensor(
            indices, weights, (neuron_count, neuron_count)
        )

        return nn.Parameter(sparse_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the pre-converted adjacency matrix in sparse format
        adj_matrix = self.adj_matrix_sparse.to(x.device)

        # Pass the input through the layer with shared weights
        for _ in range(self.connectome_layer_number):
            # Apply the mask from the adjacency matrix to the shared weights
            masked_weights = torch.sparse.mm(adj_matrix, self.shared_weights)

            # Do the forward pass using sparse matrix multiplication
            x = torch.sparse.mm(masked_weights, x) + self.shared_bias.unsqueeze(0)

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
    def __init__(self, adult_connectome_net, synapse_df, neuron_type):
        super(IntegratedModel, self).__init__()
        self.adult_connectome_net = adult_connectome_net

        # Filter the DataFrame for the specific neuron type and create a synaptic count matrix
        filtered_df = synapse_df[synapse_df["cell_type"] == neuron_type]
        self.synaptic_matrix = self.create_synaptic_matrix(filtered_df)

    @staticmethod
    def create_synaptic_matrix(df):
        # Assuming 'pre_root_id' and 'post_root_id' are integer indices.
        max_index = max(df["pre_root_id"].max(), df["post_root_id"].max()) + 1
        synaptic_matrix_csr = csr_matrix(
            (df["syn_count"], (df["pre_root_id"], df["post_root_id"])),
            shape=(max_index, max_index),
        )
        return torch.LongTensor(synaptic_matrix_csr.toarray(), dtype=torch.float32)

    def forward(self, x):
        # Pass the input through the TemporalConvNet
        x = self.temporal_conv_net(x)

        # Reshape and multiply with the synaptic count matrix
        x = x.view(x.size(0), -1)  # Flatten if necessary
        x = torch.mm(x, self.synaptic_matrix)

        # Pass through the AdultConnectomeNetwork
        return self.adult_connectome_net(x)


if __name__ == "__main__":

    synapse_df = get_synapse_df()

    neuron_type = "TmY18"

    # Create the integrated model

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
