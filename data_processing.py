import os
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, Batch

from train_funcs import (
    apply_inhibitory_r7_r8,
    construct_synaptic_matrix,
    get_activation_from_cell_type,
    get_neuron_activations,
    get_side_decision_making_vector,
    get_voronoi_averages,
    import_images,
    preprocess_images,
    process_images,
)
from voronoi_cells import VoronoiCells
from utils import paths_to_labels


class CompleteModelsDataProcessor:

    tesselated_neurons = None
    retinal_cells = ["R1-6", "R7", "R8"]

    def __init__(self, config_, data_dir=None):
        rational_cell_types = self.get_rational_cell_types()
        self.protected_cell_types = self.retinal_cells + rational_cell_types
        self._check_filtered_neurons(config_.filtered_celltypes)
        neuron_classification = self._get_neurons(
            config_.filtered_celltypes, config_.filtered_fraction, side=None
            )
        connections = self._get_connections(config_.refined_synaptic_data)
        self.root_ids = self._get_root_ids(neuron_classification, connections)
        self.synaptic_matrix = construct_synaptic_matrix(
            neuron_classification, connections, self.root_ids
        )

        if config_.log_transform_weights:
            self.synaptic_matrix.data = np.log1p(self.synaptic_matrix.data)
        if config_.random_synapses:
            self.synaptic_matrix = self.shuffle_synaptic_matrix(self.synaptic_matrix)

        self.decision_making_vector = get_side_decision_making_vector(
            self.root_ids, rational_cell_types, neuron_classification
        )
        self.neurons = config_.neurons
        self.voronoi_cells = VoronoiCells(
            eye=config_.eye,
            neurons=self.neurons,
            voronoi_criteria=config_.voronoi_criteria,
        )
        if config_.voronoi_criteria == "R7":
            self.tesselated_neurons = self.voronoi_cells.get_tesselated_neurons()
            self.voronoi_indices = self.voronoi_cells.get_image_indices()

        self.filtered_celltypes = config_.filtered_celltypes
        self.dtype = config_.dtype
        self.device = config_.DEVICE
        # This is somewhat convoluted to be compatible with the multitask training
        self.classes = sorted(os.listdir(data_dir)) if data_dir is not None else config_.CLASSES

        self.edges = torch.tensor(
            np.array([self.synaptic_matrix.row, self.synaptic_matrix.col]),
            dtype=torch.int64,  # do not touch
        )
        self.weights = torch.tensor(self.synaptic_matrix.data, dtype=self.dtype)
        self.inhibitory_r7_r8 = config_.inhibitory_r7_r8

    @property
    def num_classes(self):
        return len(self.classes)
    
    def process_batch(self, imgs, labels):
        """
        Preprocesses a batch of images and labels. This includes reshaping and colouring the images if necessary, 
        tesselating it according to the voronoi cells from the connectome, and getting the neuron activations for each
        cell. Finally, it constructs the graphs of this batch with the appropriate activations.

        Args:
            imgs (list): A list of images to be processed.
            labels (list): A list of corresponding labels.

        Returns:
            inputs (Batch): A Batch object containing processed graph data.
            labels (torch.Tensor): A tensor containing the labels.

        Raises:
            None

        """

        # Reshape and colour if needed
        imgs = preprocess_images(imgs)
        # Compute mean of three colours and add the voronoi indices
        processed_imgs = process_images(imgs, self.voronoi_indices)
        # Get the average activation for each voronoi cell
        voronoi_averages = get_voronoi_averages(processed_imgs)
        # Map these activations to the appropriate retinal neurons
        activation_tensor = self.calculate_neuron_activations(voronoi_averages)
        # Create the connectome graph with the appropirate activations
        graph_list_ = [
            Data(
                x=activation_tensor[:, j].unsqueeze(1),
                edge_index=self.edges,
                edge_attr=self.weights,
                y=labels[j],
            )
            for j in range(activation_tensor.shape[1])
        ]

        inputs = Batch.from_data_list(graph_list_).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        return inputs, labels
    
    @property
    def number_of_synapses(self):
        return self.synaptic_matrix.shape[0]

    def recreate_voronoi_cells(self):
        self.voronoi_cells.regenerate_random_centers()
        self.tesselated_neurons = self.voronoi_cells.get_tesselated_neurons()
        self.voronoi_indices = self.voronoi_cells.get_image_indices()

    def get_data_from_paths(self, paths):
        imgs = import_images(paths)
        labels = paths_to_labels(paths, self.classes)
        return imgs, labels

    def calculate_neuron_activations(self, voronoi_averages):
        neuron_activations = pd.concat(
            [get_neuron_activations(self.tesselated_neurons, a, self.inhibitory_r7_r8) for a in voronoi_averages],
            axis=1,
        )
        neuron_activations.index = neuron_activations.index.astype("string")
        activation_df = (
            self.root_ids.merge(
                neuron_activations, left_on="root_id", right_index=True, how="left"
            )
            .fillna(0)
            .set_index("index_id")
            .drop(columns=["root_id"])
        )

        return torch.tensor(activation_df.values, dtype=self.dtype)

    def plot_input_images(self, img):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self.voronoi_cells.plot_voronoi_cells_with_neurons(self.tesselated_neurons, axes[0])
        self.voronoi_cells.plot_voronoi_cells_with_image(img, axes[1])
        self.plot_neuron_activations(img, axes[2])
        plt.tight_layout()
        plt.close("all")

        return fig, "Voronoi - Original - Activations"

    def plot_neuron_activations(self, img, ax):
        # This is repeated in process_batch, but it's the cleanest way to get the plots
        img = preprocess_images(np.expand_dims(img, 0))
        processed_img = process_images(img, self.voronoi_indices)
        voronoi_average = get_voronoi_averages(processed_img)[0]
        # we need to put everything in terms of the voronoi point regions
        voronoi_average.index = [
            self.voronoi_cells.voronoi.point_region[int(i)]
            for i in voronoi_average.index
        ]
        corr_tess = self.tesselated_neurons.copy()
        corr_tess["voronoi_indices"] = [
            self.voronoi_cells.voronoi.point_region[int(i)]
            for i in corr_tess["voronoi_indices"]
        ]
        neuron_activations = corr_tess.merge(
            voronoi_average, left_on="voronoi_indices", right_index=True
        )
        if self.inhibitory_r7_r8:
            neuron_activations = apply_inhibitory_r7_r8(neuron_activations)
            
        neuron_activations["activation"] = neuron_activations.apply(
            get_activation_from_cell_type, axis=1
        )
        return self.voronoi_cells.plot_neuron_activations(neuron_activations, ax)

    @staticmethod
    def shuffle_synaptic_matrix(synaptic_matrix):
        shuffled_col = np.random.permutation(synaptic_matrix.col)
        synaptic_matrix = coo_matrix(
            (synaptic_matrix.data, (synaptic_matrix.row, shuffled_col)),
            shape=synaptic_matrix.shape,
        )
        synaptic_matrix.sum_duplicates()
        return synaptic_matrix

    @staticmethod
    def get_rational_cell_types():
        return pd.read_csv(
            "adult_data/rational_cell_types.csv", index_col=0
        ).index.tolist()

    @staticmethod
    def _get_root_ids(classification, connections):

        # get neuron root_ids that appear in both classification and in either
        #  connections pre_root_id or post_root_id
        neurons = classification[
            classification["root_id"].isin(connections["pre_root_id"])
            | classification["root_id"].isin(connections["post_root_id"])
        ]
        # pandas is really bad:
        return neurons.reset_index(drop=True).reset_index()[["root_id", "index"]].rename(columns={"index": "index_id"})

    def _get_neurons(self, filtered_celltpyes=None, filtered_fraction=None, side=None):
        all_neurons = pd.read_csv(
            "adult_data/classification.csv",
            usecols=["root_id", "cell_type", "side"],
            dtype={"root_id": "string"},
        ).fillna("Unknown")

        if filtered_celltpyes is not None and len(filtered_celltpyes) > 0:
            # If it's not a list, make it so
            if not isinstance(filtered_celltpyes, list):
                filtered_celltpyes = [filtered_celltpyes]
            all_neurons = all_neurons[
                ~all_neurons["cell_type"].isin(filtered_celltpyes)
            ]

        if filtered_fraction is not None:
            # We can't filter neurons in the retina or decision making neurons
            # so we separate these first
            protected_neurons = all_neurons[
                all_neurons["cell_type"].isin(self.protected_cell_types)
            ]
            non_protected_neurons = all_neurons[
                ~all_neurons["cell_type"].isin(self.protected_cell_types)
            ]
            # We filter the non-protected neurons
            non_protected_neurons = non_protected_neurons.sample(
                frac=filtered_fraction, random_state=1714
            )
            # And put everything back together
            all_neurons = pd.concat([protected_neurons, non_protected_neurons])

        if side is not None:
            all_neurons = all_neurons[all_neurons["side"] == side]

        # check that we have neurons
        if all_neurons.empty:
            raise ValueError("No neurons found with the given criteria.")

        return all_neurons

    @staticmethod
    def _get_connections(refined_synaptic_data=False):
        file_char = "_refined" if refined_synaptic_data else ""
        return (
            pd.read_csv(
                f"adult_data/connections{file_char}.csv",
                dtype={
                    "pre_root_id": "string",
                    "post_root_id": "string",
                    "syn_count": np.int32,
                },
                index_col=0
            )
            .groupby(["pre_root_id", "post_root_id"])
            .sum("syn_count")
            .reset_index()
        )

    def _check_filtered_neurons(self, filtered_cell_types):
        if not set(filtered_cell_types).isdisjoint(self.protected_cell_types):
            raise ValueError(
                f"You can't fitler out any of the following cell types: {', '.join(self.forbidden_cell_types)}"
            )
