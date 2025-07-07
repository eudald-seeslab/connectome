import torch
from torch_geometric.data import Data


class GraphBuilder:
    """
    Build a batched PyG graph from per-neuron activation tensors.

    This is the `connectome` part of the code. It builds a Pytorch Geometric graph from the 
    connections between neurons, and initializes it with the per-neuron activations coming from
    neuron_mapper (which acts as the retina).
    """

    def __init__(self, edges: torch.Tensor, *, device: torch.device):
        self.edges = edges
        self.device = device


    def build_batch(self, activation_tensor: torch.Tensor, labels) -> tuple[Data, torch.Tensor]:
        """Create a batched graph for *labels* from *activation_tensor*.

        Parameters
        ----------
        activation_tensor : torch.Tensor
            Shape ``(N_nodes, B)`` where ``B`` is the batch size.
        labels : Sequence[int]
            Integer class labels (length ``B``).
        """
        batch_size = len(labels)
        num_nodes = activation_tensor.shape[0]
        num_edges = self.edges.shape[1]

        x = activation_tensor.t().contiguous().view(-1, 1)

        edge_index_rep = self.edges.to(self.device).repeat(1, batch_size)
        node_offsets = (
            torch.arange(batch_size, device=self.device, dtype=torch.int32) * num_nodes
        ).repeat_interleave(num_edges)
        edge_index_rep = edge_index_rep + node_offsets.unsqueeze(0)

        batch_vec = torch.arange(batch_size, device=self.device).repeat_interleave(num_nodes)

        inputs = Data(x=x, edge_index=edge_index_rep, batch=batch_vec)
        labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)

        return inputs, labels_t
