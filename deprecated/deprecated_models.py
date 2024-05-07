# Probably deprecated
import torch
from torch.nn import Parameter
from scipy.sparse import csr_matrix


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
