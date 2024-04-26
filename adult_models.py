import numpy as np
import torch
import torch.nn as nn

from custom_torch_function import SparseMatrixMul
from graph_models import RetinaConnectionLayer


class AdultConnectome(nn.Module):
    def __init__(
        self,
        adjacency_matrix,
        layer_number: int,
        log_transform_weights: bool,
        dtype,
        sparse_layout=torch.sparse_coo,
    ):
        super(AdultConnectome, self).__init__()

        self.layer_number = layer_number

        if log_transform_weights:
            self.shared_weights = nn.Parameter(
                torch.tensor(np.log1p(adjacency_matrix.data), dtype=dtype)
            )
        else:
            self.shared_weights = nn.Parameter(
                torch.tensor(adjacency_matrix.data, dtype=dtype)
            )

        self.indices = torch.tensor(
            np.vstack((adjacency_matrix.row, adjacency_matrix.col)),
            dtype=torch.int64,
        )
        self.shape = adjacency_matrix.shape

    def forward(self, x):

        return SparseMatrixMul.apply(
            self.indices, self.shared_weights, self.shape, self.layer_number, x
        )


class FullAdultModel(nn.Module):

    def __init__(
        self,
        adjacency_matrix,
        decision_making_tensor,
        cell_type_indices,
        layer_number: int,
        log_transform_weights: bool,
        sparse_layout,
        dtype,
    ):
        super(FullAdultModel, self).__init__()

        self.sparse_layout = sparse_layout
        self.retina_connection = RetinaConnectionLayer(cell_type_indices, 1, 1)

        self.connectome = AdultConnectome(
            adjacency_matrix, layer_number, log_transform_weights, dtype
        )

        self.decision_making_tensor = decision_making_tensor
        self.final_fc = nn.Linear(self.decision_making_tensor._nnz(), 1, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.retina_connection(x)
        x = x.to_sparse(layout=self.sparse_layout)
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
