# Probably deprecated
import torch
from torch.nn import Parameter

class PermutationLayer(torch.nn.Module):
    def __init__(self, max_indices, num_types):
        super(PermutationLayer, self).__init__()
        # Assuming max_indices is the maximum number of nodes for any cell type
        self.max_indices = max_indices
        self.num_types = num_types
        # Initialize permutation indices for each cell type
        self.permutations = Parameter(
            torch.stack(
                [torch.randperm(max_indices) for _ in range(num_types)]
            ).float(),
        )

    def forward(self, x, cell_type_indices):
        # cell_type_indices should be a vector indicating the cell type for each node in x
        permuted_x = x.clone()
        for type_index in range(self.num_types):
            type_mask = cell_type_indices == type_index

            if type_mask.any():
                # Adjust perm_indices to only select as many as needed for this type
                actual_num_nodes = type_mask.sum().item()
                perm_indices = self.permutations[type_index][:actual_num_nodes]

                # We gather the nodes of this cell type, then apply the perm_indices
                permuted_nodes = x[type_mask][perm_indices.long()]

                # Reassign permuted nodes back
                permuted_x[type_mask] = permuted_nodes

        return permuted_x
