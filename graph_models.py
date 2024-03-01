import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


DROPOUT = 0.0


class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, decision_making_vector):
        super(GNNModel, self).__init__()
        self.conv = GCNConv(num_node_features, 1)
        self.register_buffer("decision_making_vector", decision_making_vector)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=DROPOUT)

        # Apply the decision-making mask
        batch_size = batch.max().item() + 1
        decision_mask = (
            self.decision_making_vector.unsqueeze(0).repeat(batch_size, 1).view(-1, 1)
        )
        x = x * decision_mask

        # Global pooling and final classifier
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)
