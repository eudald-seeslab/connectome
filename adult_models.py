from typing import Dict
from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix


class AdultConnectome(nn.Module):
    def __init__(
        self,
        adjacency_matrix,
        neuron_count: int,
        layer_number: int,
        batch_size,
        dtype,
    ):
        super(AdultConnectome, self).__init__()

        self.connectome_layer_number = layer_number

        self.shared_weights = nn.Parameter(
            self.create_coo_tensor(adjacency_matrix, dtype)
        )
        # self.shared_bias = nn.Parameter(
        #    torch.ones([neuron_count, batch_size], dtype=dtype).to_sparse()
        # )
    def forward(self, x):

        # x = x.unsqueeze(1)

        # Pass the input through the layer with shared weights
        for _ in range(self.connectome_layer_number):
            x = torch.sparse.mm(self.shared_weights, x) #+ self.shared_bias
        return x.transpose(0, 1)

    @staticmethod
    def create_coo_tensor(adjacency_matrix, dtype):
        return torch.sparse_coo_tensor(
            np.array([adjacency_matrix.row, adjacency_matrix.col]),
            adjacency_matrix.data,
            adjacency_matrix.shape,
            dtype=dtype,
            check_invariants=True,
        )

    @staticmethod
    def create_csr_tensor(adjacency_matrix, dtype):
        # Extract CSR components
        crow_indices = adjacency_matrix.indptr
        col_indices = adjacency_matrix.indices
        values = adjacency_matrix.data
        shape = adjacency_matrix.shape

        return torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                values,
                size=shape,
                dtype=dtype,
            )

class FullAdultModel(nn.Module):

    def __init__(
        self,
        adjacency_matrix,
        decision_making_vector,
        neuron_count: int,
        layer_number: int,
        batch_size: int,
        dtype,
    ):
        super(FullAdultModel, self).__init__()

        # get number of non-zero elements in decision_making_vector
        num_decision_making_neurons = decision_making_vector.sum().int().item()

        # TODO: add the permutation layer

        self.connectome = AdultConnectome(
            adjacency_matrix, neuron_count, layer_number, batch_size, dtype
        )

        self.decision_making_vector = decision_making_vector
        self.final_fc = nn.Linear(num_decision_making_neurons, 1, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.connectome(x).to_dense()
        x = x[:, self.decision_making_vector.bool()]
        x = x.view(x.size(0), -1)

        return self.final_fc(x)


# deprecated
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
