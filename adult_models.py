import numpy as np
import torch
import torch.nn as nn

from graph_models import RetinaConnectionLayer


class SparseMatrixMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, layer_number, x):
        ctx.layer_number = layer_number
        # Create the sparse tensor only with indices and values, shape is passed separately and not saved
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape, dtype=values.dtype, device=values.device
        )
        result = x
        for _ in range(layer_number):
            result = torch.sparse.mm(sparse_tensor, result)

        # Save tensors that will be needed in the backward pass
        ctx.save_for_backward(indices, values, result)
        ctx.shape = shape  # Just store shape as an attribute, not in the context

        return result

    @staticmethod
    def backward(ctx, grad_output):
        indices, values, result = ctx.saved_tensors
        shape = ctx.shape  # Retrieve the shape from the context attribute
        layer_number = ctx.layer_number

        # Rebuild the sparse tensor for use in the backward pass
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape, dtype=values.dtype, device=values.device
        )
        grad_input = grad_output
        for _ in range(layer_number):
            grad_input = torch.sparse.mm(sparse_tensor.t(), grad_input)

        return None, None, None, None, grad_input


class AdultConnectome(nn.Module):
    def __init__(
        self,
        adjacency_matrix,
        layer_number: int,
        dtype,
    ):
        super(AdultConnectome, self).__init__()

        self.layer_number = layer_number

        self.shared_weights = nn.Parameter(
            torch.tensor(adjacency_matrix.data, dtype=dtype)
        )
        self.indices = torch.tensor(
            np.vstack((adjacency_matrix.row, adjacency_matrix.col)),
            dtype=torch.int64,
        )
        self.shape = adjacency_matrix.shape

    def forward(self, x):

        return SparseMatrixMulFunction.apply(
            self.indices, self.shared_weights, self.shape, self.layer_number, x
            )


class FullAdultModel(nn.Module):

    def __init__(
        self,
        adjacency_matrix,
        decision_making_tensor,
        cell_type_indices,
        layer_number: int,
        sparse_layout,
        dtype,
    ):
        super(FullAdultModel, self).__init__()

        self.sparse_layout = sparse_layout
        self.retina_connection = RetinaConnectionLayer(
            cell_type_indices, 1, 1
        )

        self.connectome = AdultConnectome(
            adjacency_matrix, layer_number, dtype
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
