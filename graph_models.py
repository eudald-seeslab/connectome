import torch
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool, GATConv

DROPOUT = 0.0


class GNNModel(torch.nn.Module):
    def __init__(
        self, num_node_features, decision_making_vector, num_passes, num_heads=1
    ):
        super(GNNModel, self).__init__()
        self.attention_conv = GATConv(
            num_node_features, out_channels=1, heads=num_heads, concat=False
        )
        self.register_buffer("decision_making_vector", decision_making_vector)
        self.num_passes = num_passes
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.norm = torch.nn.BatchNorm1d(num_node_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_res = x
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

        # Global pooling and final classifier without additional scaling
        x_pooled = global_mean_pool(x, batch)

        # Scale the activations
        return (x_pooled - x_pooled.min()) / (x_pooled.max() - x_pooled.min() + 1e-6)
