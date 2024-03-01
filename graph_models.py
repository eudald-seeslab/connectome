import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(
            16, 32
        )  # Use an intermediary feature size greater than num_classes
        self.fc = torch.nn.Linear(
            32, num_classes
        )  # Fully connected layer to get to num_classes

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # Global pooling
        x = global_mean_pool(x, batch)  # Pool node features to the graph level

        # Apply a final classifier
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
