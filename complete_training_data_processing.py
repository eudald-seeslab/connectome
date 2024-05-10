import numpy as np
import pandas as pd
import torch
from complete_training_funcs import (
    VoronoiCells,
    get_neuron_activations,
    get_side_decision_making_vector,
    get_voronoi_averages,
    import_images,
    process_images,
)
from utils import paths_to_labels
import config
from scipy.sparse import load_npz
from torch_geometric.data import Data, Batch


class CompleteModelsDataProcessor:

    tesselated_df = None

    def __init__(self, log_transform_weights=False):
        # get data
        self.right_root_ids = pd.read_csv("adult_data/root_id_to_index.csv")
        self.synaptic_matrix = load_npz("adult_data/good_synaptic_matrix.npz")
        if log_transform_weights:
            self.synaptic_matrix.data = np.log1p(self.synaptic_matrix.data)
        self.decision_making_vector = get_side_decision_making_vector(
            self.right_root_ids, "right"
        )

    @property
    def number_of_synapses(self):
        return self.synaptic_matrix.shape[0]

    def create_voronoi_cells(self):
        self.voronoi_cells = VoronoiCells(voronoi_criteria=config.voronoi_criteria)
        self.tesselated_df = self.voronoi_cells.get_tesselated_neurons()
        self.voronoi_indices = self.voronoi_cells.get_image_indices()

    @staticmethod
    def get_data_from_paths(paths):
        imgs = import_images(paths)
        labels = paths_to_labels(paths)
        return imgs, labels

    def process_batch(self, imgs, labels):
        # FIXME: this needs a lot of cleaning

        processed_imgs = process_images(imgs, self.voronoi_indices)
        voronoi_averages = get_voronoi_averages(processed_imgs)
        neuron_activations = pd.concat(
            [get_neuron_activations(self.tesselated_df, a) for a in voronoi_averages],
            axis=1,
        )
        activation_df = (
            self.right_root_ids.merge(
                neuron_activations, left_on="root_id", right_index=True, how="left"
            )
            .fillna(0)
            .set_index("index_id")
            .drop(columns=["root_id"])
        )
        # Check that all neurons have been activated
        assert activation_df[activation_df.iloc[:, 0] > 0].shape[0] == self.tesselated_df.shape[0]
        edges = torch.tensor(
            np.array([self.synaptic_matrix.row, self.synaptic_matrix.col]),
            dtype=torch.int64,  # do not touch
        )
        weights = torch.tensor(self.synaptic_matrix.data, dtype=config.dtype)
        activation_tensor = torch.tensor(activation_df.values, dtype=config.dtype)
        graph_list_ = []
        for j in range(activation_tensor.shape[1]):
            # Shape [num_nodes, 1], one feature per node
            node_features = activation_tensor[:, j].unsqueeze(1)
            graph = Data(
                x=node_features,
                edge_index=edges,
                edge_attr=weights,
                y=labels[j],
                device=config.DEVICE,
            )
            graph_list_.append(graph)

        inputs = Batch.from_data_list(graph_list_).to(config.DEVICE)
        labels = torch.tensor(labels, dtype=config.dtype).to(config.DEVICE)

        return inputs, labels

    def plot_voronoi_cells_with_neurons(self):
        return self.voronoi_cells.plot_voronoi_cells_with_neurons(self.tesselated_df)

    def plot_voronoi_cells_with_image(self, img):
        return self.voronoi_cells.plot_voronoi_cells_with_image(img)
