from typing import Dict, Union
from unittest import TestCase

import torch
import torch.nn as nn

from adult_models_helpers import get_synapse_df

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class AdultConnectomeNetwork(nn.Module):
    def __init__(
        self,
        adjacency_matrix,
        neuron_count: int,
        general_config: Dict[str, Union[int, float, str, bool]],
    ):
        super(AdultConnectomeNetwork, self).__init__()

        self.device = DEVICE

        # Convert the adjacency matrix to a PyTorch sparse tensor once in the initialization
        self.adjacency_matrix_coo = adjacency_matrix.tocoo()
        self.adj_matrix_sparse = torch.sparse_coo_tensor(
            torch.tensor(
                [self.adjacency_matrix_coo.row, self.adjacency_matrix_coo.col]
            ),
            torch.FloatTensor(self.adjacency_matrix_coo.data),
            torch.Size(self.adjacency_matrix_coo.shape),
            device=self.device,
            dtype=torch.float16,
        )

        self.connectome_layer_number = general_config["CONNECTOME_LAYER_NUMBER"]

        # Initialize the shared weights for the connectome layers
        self.shared_weights = self.initialize_sparse_weights(
            adjacency_matrix, neuron_count
        )
        self.shared_bias = nn.Parameter(torch.ones(neuron_count))

    def initialize_sparse_weights(self, adjacency_matrix, neuron_count):
        # Extract the non-zero indices from the adjacency matrix
        non_zero_indices = torch.tensor(
            [adjacency_matrix.row, adjacency_matrix.col],
            device=self.device,
            dtype=torch.long,
        )

        # Generate random weights only for the non-zero connections
        non_zero_weights = torch.rand(
            non_zero_indices.size(1), device=self.device, dtype=torch.float16
        )

        # Create a sparse tensor with the non-zero weights
        sparse_weights = torch.sparse_coo_tensor(
            non_zero_indices,
            non_zero_weights,
            (neuron_count, neuron_count),
            device=self.device,
        )

        return nn.Parameter(sparse_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Pass the input through the layer with shared weights
        for _ in range(self.connectome_layer_number):
            # Apply the mask from the adjacency matrix to the shared weights
            masked_weights = torch.sparse.mm(
                self.adj_matrix_sparse, self.shared_weights
            )

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


# most likely deprecated
def run_batched_inference(
    network: nn.Module, input_tensor: torch.Tensor, batch_size: int
) -> torch.Tensor:
    outputs = []
    for start_idx in range(0, input_tensor.size(0), batch_size):
        end_idx = start_idx + batch_size
        outputs.append(network(input_tensor[start_idx:end_idx, :]))
        del input_tensor[start_idx:end_idx, :]
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)
