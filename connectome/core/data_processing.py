import os
import random

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix

from paths import PROJECT_ROOT

from connectome.core.train_funcs import (
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
from connectome.core.voronoi_cells import VoronoiCells
from connectome.core.neuron_mapper import NeuronMapper
from connectome.core.utils import paths_to_labels
from connectome.core.image_processor import ImageProcessor
from connectome.core.graph_builder import GraphBuilder


class DataProcessor:

    tesselated_neurons = None
    retinal_cells = ["R1-6", "R7", "R8"]

    def __init__(self, config_, input_images_dir=None):
        # TODO: this init does too many things, and they depend on each other. It should be refactored
        
        np.random.seed(config_.random_seed)
        torch.manual_seed(config_.random_seed)
        random.seed(config_.random_seed)

        data_dir_ = "new_data" if config_.new_connectome else "adult_data"
        self.data_dir = os.path.join(PROJECT_ROOT, data_dir_)
        rational_cell_types = self.get_rational_cell_types(config_.rational_cell_types)
        self.protected_cell_types = self.retinal_cells + rational_cell_types
        self._check_filtered_neurons(config_.filtered_celltypes)
        neuron_classification = self._get_neurons(
            config_.filtered_celltypes,
            config_.filtered_fraction,
            side=None,
        )
        connections = self._get_connections(config_.refined_synaptic_data, config_.randomization_strategy)
        self.root_ids = self._get_root_ids(neuron_classification, connections)
        self.synaptic_matrix = construct_synaptic_matrix(
            neuron_classification, connections, self.root_ids
        )

        if config_.log_transform_weights:
            self.synaptic_matrix.data = np.log1p(self.synaptic_matrix.data)

        self.decision_making_vector = get_side_decision_making_vector(
            self.root_ids, rational_cell_types, neuron_classification
        )
        self.neurons = config_.neurons
        self.voronoi_cells = VoronoiCells(
            data_dir=self.data_dir,
            eye=config_.eye,
            neurons=self.neurons,
            voronoi_criteria=config_.voronoi_criteria,
            )
        if config_.voronoi_criteria == "R7":
            self.tesselated_neurons = self.voronoi_cells.get_tesselated_neurons()
            self.voronoi_indices = self.voronoi_cells.get_image_indices()
            # Torch version for fast GPU operations
            self.voronoi_indices_torch = torch.tensor(self.voronoi_indices, device=config_.DEVICE, dtype=torch.long)

        self.filtered_celltypes = config_.filtered_celltypes
        self.dtype = config_.dtype
        self.device = config_.DEVICE
        # Image preprocessing helper (keeps heavy resize logic out of DataProcessor)
        self._image_processor = ImageProcessor(self.device)
        # This is somewhat convoluted to be compatible with the multitask training
        self.classes = (
            sorted(os.listdir(input_images_dir))
            if input_images_dir is not None
            else config_.CLASSES
        )

        # Store edges as int32 to halve memory; they will be cast to int64 lazily in the model.
        self.edges = torch.tensor(
            np.array([self.synaptic_matrix.row, self.synaptic_matrix.col]),
            dtype=torch.int32,
        )
        self.weights = torch.tensor(self.synaptic_matrix.data, dtype=self.dtype)
        self.inhibitory_r7_r8 = config_.inhibitory_r7_r8

        # Pre-compute mapping neuron → activations helper
        self.neuron_mapper = NeuronMapper(
            self.root_ids,
            self.tesselated_neurons,
            device=self.device,
            dtype=self.dtype,
            inhibitory_r7_r8=self.inhibitory_r7_r8,
        )

        # Graph builder utility
        self.graph_builder = GraphBuilder(self.edges, device=self.device)

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

        # Reshape and colour if needed (delegate to helper)
        imgs_t = self._image_processor.preprocess(imgs)
        processed_imgs = self._image_processor.process(imgs_t, self.voronoi_indices_torch)
        voronoi_means = VoronoiCells.compute_voronoi_means(processed_imgs, self.device)
        activation_tensor = self.neuron_mapper.activations_from_voronoi_means(voronoi_means)

        # Delete bulky intermediate tensors to reclaim GPU memory before constructing
        # the (potentially huge) batched edge index. This prevents peak-memory spikes
        # that previously caused CUDA OOMs.
        del imgs_t, processed_imgs, voronoi_means
        torch.cuda.empty_cache()

        # Convert activations into a batched PyG graph
        inputs, labels_t = self.graph_builder.build_batch(activation_tensor, labels)

        return inputs, labels_t

    @property
    def number_of_synapses(self):
        return self.synaptic_matrix.shape[0]

    def recreate_voronoi_cells(self):
        self.voronoi_cells.regenerate_random_centers()
        self.tesselated_neurons = self.voronoi_cells.get_tesselated_neurons()
        self.voronoi_indices = self.voronoi_cells.get_image_indices()
        self.voronoi_indices_torch = torch.tensor(self.voronoi_indices, device=self.device, dtype=torch.long)
        # Update fast-mapping tables because Voronoi indices changed
        self.neuron_mapper = NeuronMapper(
            self.root_ids,
            self.tesselated_neurons,
            device=self.device,
            dtype=self.dtype,
            inhibitory_r7_r8=self.inhibitory_r7_r8,
        )

    def get_data_from_paths(self, paths, get_labels=True):
        imgs = import_images(paths)
        if get_labels:
            labels = paths_to_labels(paths, self.classes)
        else:
            labels = None
        return imgs, labels

    def calculate_neuron_activations(self, voronoi_averages):
        neuron_activations = pd.concat(
            [
                get_neuron_activations(
                    self.tesselated_neurons, a, self.inhibitory_r7_r8
                )
                for a in voronoi_averages
            ],
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

    def plot_input_images(self, img, voronoi_colour="orange", voronoi_width=1):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self.voronoi_cells.plot_voronoi_cells_with_neurons(
            self.tesselated_neurons, axes[0], voronoi_colour, voronoi_width
        )
        self.plot_neuron_activations(img, axes[1], voronoi_colour, voronoi_width)
        self.voronoi_cells.plot_input_image(img, axes[2])
        plt.tight_layout()
        plt.close("all")

        return fig, "Activations -> Voronoi <- Input image"

    def plot_neuron_activations(self, img, ax, voronoi_colour, voronoi_width):
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
        return self.voronoi_cells.plot_neuron_activations(
            neuron_activations, ax, voronoi_colour, voronoi_width
        )

    @staticmethod
    def shuffle_synaptic_matrix(synaptic_matrix):
        shuffled_col = np.random.permutation(synaptic_matrix.col)
        synaptic_matrix = coo_matrix(
            (synaptic_matrix.data, (synaptic_matrix.row, shuffled_col)),
            shape=synaptic_matrix.shape,
        )
        synaptic_matrix.sum_duplicates()
        return synaptic_matrix

    def get_rational_cell_types_from_file(self):
        path = os.path.join(self.data_dir, "rational_cell_types.csv")
        df = self._read_csv_cached(path, index_col=0)
        return df.index.tolist()

    def get_rational_cell_types(self, rational_cell_types):
        if rational_cell_types is None:
            return self.get_rational_cell_types_from_file()
        return rational_cell_types

    @staticmethod
    def _get_root_ids(classification, connections):

        # get neuron root_ids that appear in both classification and in either
        #  connections pre_root_id or post_root_id
        neurons = classification[
            classification["root_id"].isin(connections["pre_root_id"])
            | classification["root_id"].isin(connections["post_root_id"])
        ]
        # pandas is terrible:
        return (
            neurons.reset_index(drop=True)
            .reset_index()[["root_id", "index"]]
            .rename(columns={"index": "index_id"})
        )

    def _get_neurons(
        self,
        filtered_celltpyes=None,
        filtered_fraction=None,
        side=None,
    ):

        all_neurons = self._read_csv_cached(
            os.path.join(self.data_dir, "classification.csv"),
            usecols=["root_id", "cell_type", "side"],
            dtype={"root_id": "string"},
        ).fillna("Unknown")

        if filtered_celltpyes is not None and len(filtered_celltpyes) > 0:
            all_neurons = all_neurons[~all_neurons["cell_type"].isin(filtered_celltpyes)]

        if filtered_fraction is not None:
            protected_neurons = all_neurons[
                all_neurons["cell_type"].isin(self.protected_cell_types)
            ]
            non_protected_neurons = all_neurons[
                ~all_neurons["cell_type"].isin(self.protected_cell_types)
            ]

            non_protected_neurons = non_protected_neurons.sample(
                frac=filtered_fraction, random_state=1714
            )
            all_neurons = pd.concat([protected_neurons, non_protected_neurons])

        if side is not None:
            all_neurons = all_neurons[all_neurons["side"] == side]

        return all_neurons

    def _get_connections(self, refined_synaptic_data=False, randomization_strategy=None):
        file_char = "_refined" if refined_synaptic_data else ""
        if randomization_strategy is not None:
            file_char += f"_random_{randomization_strategy}"

        filename = os.path.join(self.data_dir, f"connections{file_char}.csv")
        connections = self._read_csv_cached(
            filename,
            dtype={
                "pre_root_id": "string",
                "post_root_id": "string",
                "syn_count": np.int32,
            },
            index_col=0,
        )

        grouped = (
            connections.groupby(["pre_root_id", "post_root_id"])
            .sum("syn_count")
            .reset_index()
        )

        sorted_conns = grouped.sort_values(["pre_root_id", "post_root_id"])

        return sorted_conns

    def _check_filtered_neurons(self, filtered_cell_types):
        if not set(filtered_cell_types).isdisjoint(self.protected_cell_types):
            raise ValueError(
                f"You can't filter out any of the following cell types: {', '.join(self.protected_cell_types)}"
            )

    # ------------------------------------------------------------------
    # CSV cache helper
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv_cached(csv_path, **read_csv_kwargs):
        """Read a CSV file, caching a pickle the first time for faster reuse.

        A companion file with the same name and extension ``.pkl`` is created
        next to the original CSV. If the pickle is newer than the CSV, it is
        loaded directly, giving 10-30× faster load times in subsequent runs.
        """

        pkl_path = csv_path.replace(".csv", ".pkl")

        if os.path.exists(pkl_path) and os.path.getmtime(pkl_path) >= os.path.getmtime(csv_path):
            return pd.read_pickle(pkl_path)

        df = pd.read_csv(csv_path, **read_csv_kwargs)

        # Store pickle for next time; ignore failures (e.g., network fs read-only)
        try:
            df.to_pickle(pkl_path)
        except Exception:
            pass

        return df
