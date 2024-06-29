import torch
from torch.nn import Parameter
from graph_models_helpers import min_max_norm
from torch_geometric.nn import MessagePassing
from torch import nn


class Connectome(MessagePassing):
    def __init__(self, data_processor, config):
        super(Connectome, self).__init__(aggr="add")

        self.num_passes = config.NUM_CONNECTOME_PASSES
        num_nodes = data_processor.number_of_synapses
        edge_weight = data_processor.synaptic_matrix.data
        num_synapses = edge_weight.shape[0]
        self.batch_size = config.batch_size
        self.train_edges = config.train_edges
        self.train_neurons = config.train_neurons
        self.lambda_func = config.lambda_func
        self.refined_synaptic_data = config.refined_synaptic_data
        dtype = config.dtype
        device = config.DEVICE

        self.register_buffer(
            "edge_weight",
            torch.tensor(edge_weight, dtype=dtype, device=device),
        )

        if config.train_edges:
            # This allows us to have negative weights, as well as synapses comprised
            #  of many weights, where some are positive and some negative, and the result
            #  is edge_weight * edge_weight_multiplier
            self.edge_weight_multiplier = Parameter(
                torch.Tensor(num_synapses).to(device)
                )
            nn.init.uniform_(self.edge_weight_multiplier, a=-1, b=1)
        if config.train_neurons:
            self.neuron_activation_threshold = Parameter(
                torch.Tensor(num_nodes).to(device)
            )
            nn.init.uniform_(self.neuron_activation_threshold, a=0, b=0.1)

        self.neuron_dropout = nn.Dropout(config.neuron_dropout)

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
            # Here we are multiplying the edge weight by the edge_weight_multiplier. For biological 
            # plausibility, we want to make sure that the edge_weight_multiplier is not bigger than 1
            # Now, if we have the raw synaptic data, all weights are positive, so we need to allow for
            # negative weights, so edge_weight_multiplier will be \in [-1, 1]
            # If we have refined synaptic data, which can have negative weights, the edge_weight_multiplier
            # will be \in [0, 1]
            if self.refined_synaptic_data:
                edge_weight_multiplier = torch.sigmoid(self.edge_weight_multiplier)
            else:
                edge_weight_multiplier = torch.tanh(self.edge_weight_multiplier)
            
            x_j = x_j * edge_weight * edge_weight_multiplier
        else:
            x_j = x_j * edge_weight

        # Apply the neuron dropout
        x_j = self.neuron_dropout(x_j)

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

    def __init__(self, data_processor, config_):
        super(FullGraphModel, self).__init__()

        self.connectome = Connectome(data_processor, config_)
        self.register_buffer("decision_making_vector", data_processor.decision_making_vector)
        final_layer = config_.final_layer
        final_layer_input_size = int(data_processor.decision_making_vector.sum()) if final_layer == "nn" else 1
        self.final_fc = nn.Linear(final_layer_input_size, len(config_.CLASSES), dtype=config_.dtype)
        self.decision_making_dropout = nn.Dropout(config_.decision_dropout)
        self.num_features = 1 # only works with 1 for now
        self.batch_size = config_.batch_size
        self.final_layer = final_layer
        self.train_neurons = config_.train_neurons

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # pass through the connectome
        x = self.connectome(x, edge_index)
        # get final decision
        x = x.view(self.batch_size, -1, self.num_features)
        x, batch = self.decision_making_mask(x, batch)
        # Save the intermediate output for analysis
        if not self.training:
            self.intermediate_output = x.view(self.batch_size, -1).clone().detach()

        if self.final_layer == "mean":
            # get the mean for each batch
            x = torch.mean(x, dim=1, keepdim=True)

        if not self.train_neurons:
            # When we are training edges or only the final layer, the output
            #  explodes a bit and we need to normalize it
            x = x / x.norm()

        x = self.decision_making_dropout(x)

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
