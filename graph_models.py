import torch
from torch.nn import functional as F
from torch.nn import Parameter, ParameterList, Module
from torch_geometric.nn import global_mean_pool, GATConv

DROPOUT = 0.0


class GNNModel(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        decision_making_vector,
        num_passes,
        cell_type_indices,
        batch_size,
        num_heads=1,
    ):

        super(GNNModel, self).__init__()
        self.attention_conv = GATConv(
            num_node_features, out_channels=1, heads=num_heads, concat=False
        )
        self.register_buffer("decision_making_vector", decision_making_vector)
        self.num_passes = num_passes
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.norm = torch.nn.BatchNorm1d(num_node_features)

        self.permutation_layer = CustomFullyConnectedLayer(
            cell_type_indices, batch_size, num_node_features
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.permutation_layer(x)

        # Make a copy to be used later
        x_res = x

        # Noise layer for the non-initialized neurons
        x = self.__add_noise(x)

        # Multiple passes through the connectome
        for _ in range(self.num_passes):
            x = self.attention_conv(x, edge_index)
            x = self.norm(x)
            x = self.leaky_relu(x)
            x = F.dropout(x, training=self.training, p=DROPOUT)

            x = x + x_res

        # Apply the decision-making mask
        batch_size = batch.max().item() + 1
        decision_mask = (
            self.decision_making_vector.unsqueeze(0).repeat(batch_size, 1).view(-1, 1)
        )
        x = x * decision_mask

        # Global pooling and return
        return global_mean_pool(x, batch)

    @staticmethod
    def __add_noise(x):
        mask = x == 0
        std_dev = x[x != 0].std().item()
        noise = (std_dev / 100) * torch.randn_like(x)
        return torch.where(mask, noise, x)


class CustomFullyConnectedLayer(Module):
    def __init__(self, cell_type_indices, batch_size, num_features=1):
        super().__init__()
        self.cell_type_indices = cell_type_indices
        # Dictionary: cell type -> count of neurons
        self.neuron_counts = self.get_neuron_counts(cell_type_indices)
        self.num_features = num_features
        self.batch_size = batch_size

        # Initialize weight matrices for each cell type based on neuron_counts
        self.weights = ParameterList(
            [
                Parameter(torch.randn(batch_size, count, count, dtype=torch.float))
                for count in self.neuron_counts.values()
            ]
        )

    def forward(self, x):
        # GNNs work with batches stacked into a unidimensional tensor, but I
        # find this difficult to work with. Here, we widen it to
        # [batch_size, num_neurons, num_features]
        x = x.view(self.batch_size, -1, self.num_features)

        # [batch_size, num_neurons, num_features]
        output = torch.zeros_like(x)
        for type_index in self.neuron_counts.keys():
            # Determine which nodes belong to the current cell type
            # [num_neurons_for_selected_type]
            mask_indices = torch.tensor(
                self.cell_type_indices[self.cell_type_indices == type_index].index,
                dtype=torch.long,
            )

            if len(mask_indices) > 0:

                # To simulate a permutation pairing, apply gumbel softmax in the row dimension for each batch
                soft_weight = F.gumbel_softmax(self.weights[type_index], dim=1)

                # FIXME: with this approach, two visual neurons can map to the same connectome neuron

                # Apply weights to the nodes of this cell type
                # [num_neurons_for_selected_type, num_neurons_for_selected_type] *
                # [batch_size, num_neurons_for_selected_type, num_features] =
                # [batch_size, num_neurons_for_selected_type, num_features]
                output[:, mask_indices] = torch.matmul(
                    soft_weight, torch.index_select(x, 1, mask_indices)
                ).to(output.dtype)

        # We need to return output in the same shape as the input x
        # [num_neurons * batch_size, num_features]
        return output.view(-1, self.num_features)

    @staticmethod
    def get_neuron_counts(cell_type_indices):
        return dict(sorted(cell_type_indices.value_counts().to_dict().items()))


# Probably deprecated
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
                # Adjust perm_indices to only select as many as needed for this type in this batch
                actual_num_nodes = type_mask.sum().item()
                perm_indices = self.permutations[type_index][:actual_num_nodes]

                # We gather the nodes of this cell type, then apply the perm_indices
                permuted_nodes = x[type_mask][perm_indices.long()]

                # Reassign permuted nodes back
                permuted_x[type_mask] = permuted_nodes

        return permuted_x
