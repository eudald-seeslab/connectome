from typing import Dict
from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn
from torch.sparse import to_sparse_semi_structured

from scipy.sparse import csr_matrix


class AdultConnectome(nn.Module):
    def __init__(
        self,
        adjacency_matrix,
        layer_number: int,
        dtype,
    ):
        super(AdultConnectome, self).__init__()

        self.connectome_layer_number = layer_number

        self.shared_weights = nn.Parameter(
            self.create_coo_tensor(adjacency_matrix, dtype)
        )
        # FIXME: This is not correct. We only need biases for the non-zero parameters
        # self.shared_bias = nn.Parameter(
        #    torch.ones(neuron_count, dtype=dtype).to_sparse()
        # )
    def forward(self, x):

        # Pass the input through the layer with shared weights
        for i in range(self.connectome_layer_number):
            x = torch.sparse.mm(self.shared_weights, x) #+ self.shared_bias
        return x

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
        decision_making_tensor,
        layer_number: int,
        dtype,
    ):
        super(FullAdultModel, self).__init__()

        # TODO: add the permutation layer

        self.connectome = AdultConnectome(
            adjacency_matrix, layer_number, dtype
        )

        self.decision_making_tensor = decision_making_tensor
        self.final_fc = nn.Linear(self.decision_making_tensor._nnz(), 1, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.connectome(x)
        x = torch.sparse.mm(self.decision_making_tensor, x)
        x = x.to_dense().squeeze(1)

        return self.final_fc(x)

    @staticmethod
    def create_sparse_decision_vector(decision_vector, batch_size):

        decision_vector = decision_vector.unsqueeze(0).expand(
            batch_size, decision_vector.shape[0], decision_vector.shape[1]
        )

        return decision_vector.to_sparse()
