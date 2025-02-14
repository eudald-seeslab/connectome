import numpy as np
import torch
import torch.nn as nn

from deprecated.adult_models_helpers import scipy_coo_to_torch_compressed_sparse
from deprecated.custom_torch_sparse_layer import CompressedSparseMatrixMul, SparseMatrixMul
from connectome import RetinaConnectionLayer


class AdultConnectome(nn.Module):
    def __init__(
        self,
        adj_matrix,
        layer_number: int,
        log_trans: bool,
        dtype,
        device,
        sparse_layout=torch.sparse_coo,
    ):
        super(AdultConnectome, self).__init__()

        self.layer_number = layer_number
        self.sparse_layout = sparse_layout
        weights = np.log1p(adj_matrix.data) if log_trans else adj_matrix.data

        # these are the actual weights of the model, and it's what
        #  pytorch updates in the custom backward function
        shared_weights = nn.Parameter(torch.tensor(weights, dtype=dtype))
        if sparse_layout == torch.sparse_coo:
            self.sparse_tensor = torch.sparse_coo_tensor(
                np.vstack((adj_matrix.row, adj_matrix.col)),
                shared_weights,
                adj_matrix.shape,
                dtype=dtype,
                device=device,
            ).coalesce()
        else:
            self.sparse_tensor = scipy_coo_to_torch_compressed_sparse(
                adj_matrix, shared_weights, sparse_layout, dtype, device
            )

    def forward(self, x):
        if self.sparse_layout == torch.sparse_coo:
            return SparseMatrixMul.apply(self.sparse_tensor, self.layer_number, x)
        else:
            return CompressedSparseMatrixMul.apply(
                self.sparse_tensor, self.layer_number, x, self.sparse_layout
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
        device,
    ):
        super(FullAdultModel, self).__init__()

        self.sparse_layout = sparse_layout
        self.retina_connection = RetinaConnectionLayer(
            cell_type_indices, 1, 1, dtype=dtype
            )     

        self.connectome = AdultConnectome(
            adjacency_matrix, layer_number, log_transform_weights, dtype, device, sparse_layout=sparse_layout
        )

        self.decision_making_tensor = decision_making_tensor
        self.final_fc = nn.Linear(self.decision_making_tensor._nnz(), 1, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.retina_connection(x)
        x = x.to_sparse(layout=self.sparse_layout)
        x = self.connectome(x)
        if x.layout == torch.sparse_coo:
            x = torch.sparse.mm(self.decision_making_tensor, x)
        else:
            x = self.decision_making_tensor.matmul(x)
        x = x.to_dense().squeeze(1)

        return self.final_fc(x)

    @staticmethod
    def create_sparse_decision_vector(decision_vector, batch_size):

        decision_vector = decision_vector.unsqueeze(0).expand(
            batch_size, decision_vector.shape[0], decision_vector.shape[1]
        )

        return decision_vector.to_sparse()
