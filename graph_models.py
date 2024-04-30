import torch
from torch.nn import functional as F
from torch.nn import Parameter, ParameterList, Module
from torch_geometric.nn import GATConv, global_max_pool, MessagePassing
from torch import nn

DROPOUT = 0.0


class TrainableEdgeConv(MessagePassing):
    def __init__(self, input_shape, num_connectome_passes=1):
        super(TrainableEdgeConv, self).__init__(aggr="add")

        self.num_passes = num_connectome_passes
        self.norm = nn.LayerNorm(normalized_shape=input_shape)

    def forward(self, x, edge_index, edge_weight):
        # Start propagating messages.
        size = (x.size(0), x.size(0))

        for _ in range(self.num_passes):
            x = self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)
            # otherwise the thing will explode
            # todo: think about only doing this after all passes
            x = self.norm(x.t()).t()

        return x

    def message(self, x_j, edge_weight):
        # Message: edge_weight (learnable) multiplied by node feature of the neighbor
        return x_j * edge_weight.unsqueeze(1)

    def update(self, aggr_out):
        # Each node gets its updated feature as the sum of its neighbor contributions.
        return aggr_out


class EdgeWeightedGNNModel(torch.nn.Module):
    def __init__(self, input_shape, log_transform_weights, num_connectome_passes):
        super(EdgeWeightedGNNModel, self).__init__()

        self.conv = TrainableEdgeConv(input_shape, num_connectome_passes)
        self.log_transform_weights = log_transform_weights

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        if self.log_transform_weights:
            edge_weight = torch.log1p(edge_weight)
        x = self.conv(x, edge_index, edge_weight)

        return x


class FullGraphModel(nn.Module):

    def __init__(
        self,
        input_shape,
        num_connectome_passes,
        cell_type_indices,
        decision_making_vector,
        log_transform_weights: bool,
        batch_size,
        dtype,
        num_features=1,
    ):
        super(FullGraphModel, self).__init__()

        self.retina_connection = RetinaConnectionLayer(
            cell_type_indices, 1, 1, dtype=dtype
        )
        self.register_buffer("decision_making_vector", decision_making_vector)

        self.connectome = TrainableEdgeConv(input_shape, num_connectome_passes)

        self.final_fc = nn.Linear(1, 1, dtype=dtype)
        self.log_transform_weights = log_transform_weights
        self.num_features = num_features
        self.batch_size = batch_size

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        if self.log_transform_weights:
            # if there are a lot of passes, this prevents the system from exploding
            edge_weight = torch.log1p(edge_weight)

        # find the optimal connection between retina model and connectome
        x = self.retina_connection(x)
        x = self.normalize_non_zero(x)
        # pass through the connectome
        x = self.connectome(x, edge_index, edge_weight)
        # get final decision
        x = x[self.decision_making_vector == 1]
        x = self.min_max_norm(x)
        # fixme: this only works for batch size 1
        x = torch.mean(x, dim=0, keepdim=True)

        # final layer to get the correct magnitude
        x = self.final_fc(x)
        return F.relu(x).squeeze(0)

    @staticmethod
    def normalize_non_zero(x):
        # normalize only non-zero values
        mask = x != 0
        x[mask] = (x[mask] - x[mask].mean()) / x[mask].std()

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
