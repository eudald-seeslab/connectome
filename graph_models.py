import torch
from torch.nn import functional as F
from torch.nn import Parameter, ParameterList, Module
from torch_geometric.nn import GATConv, global_max_pool, MessagePassing
from torch import nn
from torch_scatter import scatter_mean, scatter_std

DROPOUT = 0.0


class TrainableEdgeConv(MessagePassing):
    def __init__(self, input_shape, num_connectome_passes=1):
        super(TrainableEdgeConv, self).__init__(aggr="add")

        self.num_passes = num_connectome_passes
        self.num_nodes_per_graph = input_shape

    def forward(self, x, edge_index, edge_weight, batch):
        # Start propagating messages.
        size = (x.size(0), x.size(0))
        batch_size = batch.max().item() + 1

        for _ in range(self.num_passes):
            x = self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)
            # Reshape, normalize, and then flatten back
            x = x.view(batch_size, self.num_nodes_per_graph, -1)
            x = x / x.norm(dim=1, keepdim=True)
            x = x.view(-1, x.size(2))

        return x

    def message(self, x_j, edge_weight):
        # Message: edge_weight (learnable) multiplied by node feature of the neighbor
        return x_j * edge_weight.unsqueeze(1)

    def update(self, aggr_out):
        # Each node gets its updated feature as the sum of its neighbor contributions.
        return aggr_out


class FullGraphModel(nn.Module):

    def __init__(
        self,
        input_shape,
        num_connectome_passes,
        decision_making_vector,
        log_transform_weights: bool,
        batch_size,
        dtype,
        num_features=1,
        cell_type_indices=None,
        retina_connection=True,
    ):
        super(FullGraphModel, self).__init__()
        self.retina_connection = retina_connection
        if retina_connection:
            self.retina_connection = RetinaConnectionLayer(
                cell_type_indices, batch_size, num_features=1, dtype=dtype
            )
        self.register_buffer("decision_making_vector", decision_making_vector)

        self.connectome = TrainableEdgeConv(input_shape, num_connectome_passes)

        self.final_fc = nn.Linear(1, 1, dtype=dtype)
        self.log_transform_weights = log_transform_weights
        self.num_features = num_features
        self.batch_size = batch_size

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if self.log_transform_weights:
            # if there are a lot of passes, this prevents the system from exploding
            edge_weight = torch.log1p(edge_weight)

        if self.retina_connection:
            # find the optimal connection between retina model and connectome
            x = self.retina_connection(x)
            x = self.normalize_non_zero(x, batch)

        # pass through the connectome
        x = self.connectome(x, edge_index, edge_weight, batch)
        # get final decision
        x, batch = self.decision_making_mask(x, batch)
        x = self.normalize_non_zero(x, batch)
        x = x.view(self.batch_size, -1, self.num_features)
        x = torch.mean(x, dim=1, keepdim=True)

        # final layer to get the correct magnitude
        x = self.final_fc(x)
        return F.relu(x).squeeze()
    
    def decision_making_mask(self, x, batch):
        x = x.view(self.batch_size, -1, self.num_features)
        x = x[:, self.decision_making_vector == 1, :]

        batch = batch.view(self.batch_size, -1)
        batch = batch[:, self.decision_making_vector == 1]
        return x.view(-1, self.num_features), batch.view(-1)

    @staticmethod
    def normalize_non_zero(x, batch, epsilon=1e-5):
        # x has shape [batch_size * num_neurons, num_features]
        # batch has shape [batch_size * num_neurons]
        batch_size = batch.max().item() + 1
        non_zero_mask = x != 0

        non_zero_entries = x[non_zero_mask]
        non_zero_batch = batch[non_zero_mask.squeeze()]

        mean_per_batch = scatter_mean(non_zero_entries, non_zero_batch, dim=0, dim_size=batch_size)
        std_per_batch = (
            scatter_std(non_zero_entries, non_zero_batch, dim=0, dim_size=batch_size)
            + epsilon
        )

        x[non_zero_mask] = (
            non_zero_entries - mean_per_batch[non_zero_batch]
        ) / std_per_batch[non_zero_batch]

        return x

    @staticmethod
    def min_max_norm(x):
        return (x - x.min()) / (x.max() - x.min())


class GNNModel(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        decision_making_vector,
        num_passes,
        cell_type_indices,
        batch_size,
        num_heads=1,
        visual_input_persistence_rate=1,
    ):

        super(GNNModel, self).__init__()
        self.attention_conv = GATConv(
            num_node_features, out_channels=1, heads=num_heads, concat=False
        )
        self.register_buffer("decision_making_vector", decision_making_vector)
        self.num_passes = num_passes
        self.batch_size = batch_size
        self.visual_input_persistence_rate = visual_input_persistence_rate
        # swish function to attempt to mimic inhibitory behaviours too
        # TODO: check whether this is a good idea
        self.activation = torch.nn.SiLU()
        self.final_activation = torch.nn.ReLU()
        self.norm = torch.nn.BatchNorm1d(num_node_features)

        self.permutation_layer = RetinaConnectionLayer(
            cell_type_indices, batch_size, num_node_features
        )
        self.final_decision_layer = torch.nn.Linear(in_features=1, out_features=1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Match visual stimulus to correct connectome neuron
        x = self.permutation_layer(x)

        # Make a copy to be used later
        x_res = x

        # Noise layer for the non-initialized neurons
        x = self.__add_noise(x)

        # Multiple passes through the connectome
        for i in range(self.num_passes):
            x = self.attention_conv(x, edge_index)
            x = self.norm(x)
            x = self.activation(x)
            x = F.dropout(x, training=self.training, p=DROPOUT)

            # Add the initial input (because we might still be visualizing the input)
            # but decay its influence over time
            x = x + x_res * self.visual_input_persistence_rate**i

        # Apply the decision-making mask and pool the results of decision-making neurons
        decision_mask = (
            self.decision_making_vector.unsqueeze(0)
            .repeat(self.batch_size, 1)
            .view(-1, 1)
        )
        x = x * decision_mask
        pooled_x = global_max_pool(x, batch)

        # Create a final layer of just one neuron to "normalize" outputs
        # This is equivalent to the soul of the fruit fly
        return self.final_activation(self.final_decision_layer(pooled_x))

    @staticmethod
    def __add_noise(x):
        mask = x == 0
        std_dev = x[x != 0].std().item()
        noise = (std_dev / 100) * torch.randn_like(x)
        return torch.where(mask, noise, x)


class RetinaConnectionLayer(Module):
    def __init__(self, cell_type_indices, batch_size, num_features=1, dtype=torch.float):
        super().__init__()
        self.cell_type_indices = cell_type_indices
        # Dictionary: cell type -> count of neurons
        self.neuron_counts = self.get_neuron_counts(cell_type_indices)
        self.num_features = num_features
        self.batch_size = batch_size
        self.dtype = dtype

        # Initialize weight matrices for each cell type based on neuron_counts
        self.weights = ParameterList(
            [
                Parameter(torch.randn(batch_size, count, count, dtype=dtype))
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
                dtype=torch.int32,
                device=x.device,
            )

            if len(mask_indices) > 0:

                # To simulate a permutation pairing, apply gumbel softmax in the row dimension
                soft_weight = F.gumbel_softmax(self.weights[type_index], dim=1)

                # FIXME: two visual neurons can map to the same connectome neuron

                # Apply weights to the nodes of this cell type
                # [num_neurons_for_selected_type, num_neurons_for_selected_type] *
                # [batch_size, num_neurons_for_selected_type, num_features] =
                # [batch_size, num_neurons_for_selected_type, num_features]
                output[:, mask_indices] = torch.matmul(
                    soft_weight, torch.index_select(x, 1, mask_indices)
                )

        # We need to return output in the same shape as the input x
        # [num_neurons * batch_size, num_features]
        return output.view(-1, self.num_features)

    @staticmethod
    def get_neuron_counts(cell_type_indices):
        return dict(sorted(cell_type_indices.value_counts().to_dict().items()))
