# Probably deprecated
import torch
from torch.nn import Module, Parameter, ParameterList, functional as F
from scipy.sparse import csr_matrix

from connectome import DROPOUT


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


# deprecated
class IntegratedModel(nn.Module):
    def __init__(self, adult_connectome_net, synapse_df, neuron_type):
        super(IntegratedModel, self).__init__()
        self.adult_connectome_net = adult_connectome_net

        # Filter the DataFrame for the specific neuron type and create a synaptic count matrix
        filtered_df = synapse_df[synapse_df["cell_type"] == neuron_type]
        self.synaptic_matrix = self.create_synaptic_matrix(filtered_df)

    @staticmethod
    def create_synaptic_matrix(df):
        # Assuming 'pre_root_id' and 'post_root_id' are integer indices.
        max_index = max(df["pre_root_id"].max(), df["post_root_id"].max()) + 1
        synaptic_matrix_csr = csr_matrix(
            (df["syn_count"], (df["pre_root_id"], df["post_root_id"])),
            shape=(max_index, max_index),
        )
        return torch.LongTensor(synaptic_matrix_csr.toarray(), dtype=torch.float32)

    def forward(self, x):
        # Pass the input through the TemporalConvNet
        x = self.temporal_conv_net(x)

        # Reshape and multiply with the synaptic count matrix
        x = x.view(x.size(0), -1)  # Flatten if necessary
        x = torch.mm(x, self.synaptic_matrix)

        # Pass through the AdultConnectomeNetwork
        return self.adult_connectome_net(x)


class DEPRECATED_RetinaConnectionLayer(Module):
    # DEPRECATED
    def __init__(
        self, cell_type_indices, batch_size, num_features=1, dtype=torch.float
    ):
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


class GNNModel(torch.nn.Module):

    # DEPRECATED

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

        self.permutation_layer = DEPRECATED_RetinaConnectionLayer(
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
