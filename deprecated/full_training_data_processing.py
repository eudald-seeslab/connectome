import pathlib
from random import sample
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from torch_geometric.data import Data, Batch

from configs import config
import flyvision
from flyvision.utils.activity_utils import LayerActivity
from deprecated.from_image_to_video import image_paths_to_sequences
from deprecated.from_retina_to_connectome_funcs import (
    compute_voronoi_averages,
    from_retina_to_connectome,
)
from deprecated.from_retina_to_connectome_utils import (
    vector_to_one_hot,
)
from connectome import get_image_paths, paths_to_labels


class FullModelsDataProcessor:
    extent = 15
    kernel_size = 13
    dt = config.dt
    last_good_frame = config.last_good_frame
    final_retina_cells = config.final_retina_cells
    dtype = config.dtype
    DEVICE = config.DEVICE
    sparse_layout = config.sparse_layout
    normalize_voronoi_cells = config.normalize_voronoi_cells

    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
        self.receptors = flyvision.rendering.BoxEye(
            extent=self.extent, kernel_size=self.kernel_size
        )
        self.cwd = pathlib.Path().resolve()
        self.data_dir = self.cwd / "adult_data"
        network_view = flyvision.NetworkView(
            flyvision.results_dir / "opticflow/000/0000"
        )
        self.network = network_view.init_network(chkpt="best_chkpt")
        self.root_id_to_index = pd.read_csv(self.data_dir / "root_id_to_index.csv")
        self.classification = pd.read_csv(self.data_dir / "classification_clean.csv")
        self.synaptic_matrix = load_npz(self.data_dir / "good_synaptic_matrix.npz")

    def get_images(self, images_dir, small, small_length):
        return get_image_paths(self.cwd / images_dir, small, small_length)

    def cell_type_indices(self):

        merged_df = self.root_id_to_index.merge(
            self.classification, on="root_id", how="left"
        )

        # Filter only for cells in decoding_cells
        merged_df = merged_df[merged_df["cell_type"].isin(self.final_retina_cells)]

        # Generate a mapping from cell types to unique integer indices
        cell_type_to_index = {
            cell_type: i for i, cell_type in enumerate(merged_df["cell_type"].unique())
        }

        # Apply the mapping to get cell_type indices
        merged_df["cell_type_index"] = merged_df["cell_type"].map(cell_type_to_index)

        # Return a series with one column indicating the cell type index for each node
        return merged_df["cell_type_index"]

    def decision_making_neurons(self):
        rational_neurons = pd.read_csv(
            self.data_dir / "rational_neurons.csv", index_col=0
        )
        decision_making_vector = torch.tensor(
            rational_neurons.values.squeeze(), dtype=self.dtype
        ).detach()
        return vector_to_one_hot(
            decision_making_vector, self.dtype, self.sparse_layout
        ).to(self.DEVICE)

    def process_full_models_layers_data(self, i, batch_files):
        labels = paths_to_labels(batch_files)
        batch_sequences = image_paths_to_sequences(batch_files)
        rendered_sequences = self.receptors(batch_sequences)

        layer_activations = []
        for rendered_sequence in rendered_sequences:
            # rendered sequences are in RGB; move it to 0-1 for better training
            rendered_sequence = torch.div(rendered_sequence, 255)
            simulation = self.network.simulate(rendered_sequence[None], self.dt)
            layer_activations.append(
                LayerActivity(simulation, self.network.connectome, keepref=True)
            )

        self.wandb_logger.log_images(
            i, layer_activations, batch_sequences, rendered_sequences, batch_files
        )

        voronoi_averages_df = compute_voronoi_averages(
            layer_activations,
            self.classification,
            self.final_retina_cells,
            last_good_frame=self.last_good_frame,
            normalize=self.normalize_voronoi_cells,
        )
        activation_df = from_retina_to_connectome(
            voronoi_averages_df, self.classification, self.root_id_to_index
        )
        del layer_activations, rendered_sequences, rendered_sequence, simulation
        torch.cuda.empty_cache()

        inputs = torch.tensor(
            activation_df.values, dtype=self.dtype, device=self.DEVICE
        )
        labels = torch.tensor(labels, dtype=self.dtype, device=self.DEVICE)
        return labels, inputs

    def process_full_models_graph_data(self, i, batch_files):
        labels = paths_to_labels(batch_files)
        batch_sequences = image_paths_to_sequences(batch_files)
        rendered_sequences = self.receptors(batch_sequences)

        layer_activations = []
        for rendered_sequence in rendered_sequences:
            # rendered sequences are in RGB; move it to 0-1 for better training
            rendered_sequence = torch.div(rendered_sequence, 255)
            simulation = self.network.simulate(rendered_sequence[None], self.dt)
            layer_activations.append(
                LayerActivity(simulation, self.network.connectome, keepref=True)
            )

        self.wandb_logger.log_images(
            i, layer_activations, batch_sequences, rendered_sequences, batch_files
        )
        inputs, labels = self.from_retina_to_model(layer_activations, labels)

        del layer_activations, rendered_sequences, rendered_sequence, simulation
        torch.cuda.empty_cache()

        return inputs, labels

    def from_retina_to_model(self, activations, labels):

        voronoi_averages_df = compute_voronoi_averages(
            activations,
            self.classification,
            self.final_retina_cells,
            last_good_frame=self.last_good_frame,
        )
        activation_df = from_retina_to_connectome(
            voronoi_averages_df, self.classification, self.root_id_to_index
        )
        graph_list = self.from_connectome_to_model(activation_df, labels)
        return (
            Batch.from_data_list(graph_list),
            torch.tensor(labels, dtype=self.dtype, device=self.DEVICE),
        )

    def from_connectome_to_model(self, activation_df_, labels_):

        edges = torch.tensor(
            np.array([self.synaptic_matrix.row, self.synaptic_matrix.col]),
            # Note: the edges need to be specificaly int64
            dtype=torch.int64,
        )
        weights = torch.tensor(self.synaptic_matrix.data, dtype=self.dtype)
        activation_tensor = torch.tensor(activation_df_.values, dtype=self.dtype)
        graph_list_ = []
        for i in range(activation_tensor.shape[1]):
            # Shape [num_nodes, 1], one feature per node
            node_features = activation_tensor[:, i].unsqueeze(1)
            graph = Data(
                x=node_features,
                edge_index=edges,
                edge_attr=weights,
                y=labels_[i],
                device=self.DEVICE,
            )
            graph_list_.append(graph)

        return graph_list_
