import torch
from torch.nn import Parameter
from graph_models_helpers import min_max_norm
from torch_geometric.nn import MessagePassing
from torch import nn


class Connectome(MessagePassing):
    def __init__(
        self,
        input_shape,
        edge_weights,
        num_connectome_passes,
        batch_size,
        dtype,
        device,
        train_edges,
        train_neurons,
        lambda_func=None,
    ):
        super(Connectome, self).__init__(aggr="add")

        self.num_passes = num_connectome_passes
        self.num_nodes_per_graph = input_shape
        self.batch_size = batch_size
        self.train_edges = train_edges
        self.train_neurons = train_neurons
        self.lambda_func = lambda_func

        self.register_buffer(
            "edge_weight", torch.tensor(edge_weights, dtype=dtype, device=device)
        )

        if train_edges:
            # This allows us to have negative weights, as well as synapses comprised
            #  of many weights, where some are positive and some negative, and the result
            #  is edge_weight * edge_weight_multiplier
            self.edge_weight_multiplier = Parameter(
                torch.Tensor(edge_weights.shape[0]).to(device)
            )
            nn.init.uniform_(self.edge_weight_multiplier, a=-0.1, b=0.1)
        if train_neurons:
            self.neuron_activation_threshold = Parameter(
                torch.Tensor(input_shape).to(device)
            )
            nn.init.uniform_(self.neuron_activation_threshold, a=0, b=0.1)

    def forward(self, x, edge_index):
        # Start propagating messages.
        size = (x.size(0), x.size(0))

        for _ in range(self.num_passes):
            x = self.propagate(edge_index, size=size, x=x, edge_weight=self.edge_weight)

        return x

    def message(self, x_j, edge_weight):
        # Message: edge_weight (learnable) multiplied by node feature of the neighbor
        # manual reshape to make sure that the multiplication is done correctly
        x_j = x_j.view(self.batch_size, -1)
        if self.train_edges:
            # multiply by the modulated edge weight
            x_j = x_j * edge_weight * self.edge_weight_multiplier
        else:
            x_j = x_j * edge_weight

        return x_j.view(-1, 1)

    def update(self, aggr_out):

        if self.train_neurons:
            # Each node gets its updated feature as the sum of its neighbor contributions.
            # Then, we apply the lambda function with a threshold, to emulate the biological
            temp = aggr_out.view(self.batch_size, -1)
            # Min-max normalization per graph so that the thresholds are not too high
            temp = min_max_norm(temp)
            # Apply the threshold. Note the "abs" to make sure that the threshold is not
            #  helping the neuron to activate
            sig_out = self.lambda_func(temp - abs(self.neuron_activation_threshold))
            return sig_out.view(-1, 1)

        return aggr_out


class FullGraphModel(nn.Module):

    def __init__(
        self,
        input_shape,
        num_connectome_passes,
        decision_making_vector,
        batch_size,
        dtype,
        edge_weights,
        device,
        train_edges,
        train_neurons,
        lambda_func,
        final_layer="mean",
        num_classes=2,
        num_features=1,
    ):
        super(FullGraphModel, self).__init__()
        self.register_buffer("decision_making_vector", decision_making_vector)

        self.connectome = Connectome(
            input_shape=input_shape,
            edge_weights=edge_weights,
            num_connectome_passes=num_connectome_passes,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            train_edges=train_edges,
            train_neurons=train_neurons,
            lambda_func=lambda_func,
        )
        final_layer_input_size = int(decision_making_vector.sum()) if final_layer == "nn" else 1
        self.final_fc = nn.Linear(final_layer_input_size, num_classes, dtype=dtype)
        self.num_features = num_features
        self.batch_size = batch_size
        self.final_layer = final_layer
        self.train_neurons = train_neurons

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # pass through the connectome
        x = self.connectome(x, edge_index)
        # get final decision
        x = x.view(self.batch_size, -1, self.num_features)
        x, batch = self.decision_making_mask(x, batch)
        if self.final_layer == "mean":
            # get the mean for each batch
            x = torch.mean(x, dim=1, keepdim=True)
        
        if not self.train_neurons:
            # When we are training edges or only the final layer, the output
            #  explodes a bit and we need to normalize it
            x = x / x.norm()

        # final layer to get the correct magnitude
        # Squeeze the num_features. If at some point it is not 1, then we have to change this
        return self.final_fc(x.squeeze(2)).squeeze()

    def decision_making_mask(self, x, batch):
        x = x[:, self.decision_making_vector == 1, :]

        batch = batch.view(self.batch_size, -1)
        batch = batch[:, self.decision_making_vector == 1]
        return x, batch

    @staticmethod
    def min_max_norm(x):
        return (x - x.min()) / (x.max() - x.min())
