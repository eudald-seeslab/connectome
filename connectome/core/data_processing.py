import os
import random

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch_scatter import scatter_add
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
import torch.nn.functional as F

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
from connectome.core.utils import paths_to_labels


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

        # Pre-compute mapping neuron -> (cell_idx, channel_idx) for fast activation gathering
        self._build_neuron_mappings()

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
        imgs_t = self._preprocess_images_torch(imgs)
        processed_imgs = self._process_images_torch(imgs_t)
        voronoi_means = self._get_voronoi_means_torch(processed_imgs)
        activation_tensor = self._calculate_neuron_activations_torch(voronoi_means)

        # Delete bulky intermediate tensors to reclaim GPU memory before constructing
        # the (potentially huge) batched edge index. This prevents peak-memory spikes
        # that previously caused CUDA OOMs.
        del imgs_t, processed_imgs, voronoi_means
        torch.cuda.empty_cache()

        # Build a single batched graph (avoids Python-level loops)
        batch_size = len(labels)
        num_nodes = activation_tensor.shape[0]
        num_edges = self.edges.shape[1]

        # Node features: flatten batch dimension
        x = activation_tensor.t().contiguous().view(-1, 1)

        # Edge index replication with node offsets per graph
        edge_index_rep = self.edges.to(self.device).repeat(1, batch_size)
        node_offsets = (
            torch.arange(batch_size, device=self.device, dtype=torch.int32) * num_nodes
        ).repeat_interleave(num_edges)
        edge_index_rep = edge_index_rep + node_offsets.unsqueeze(0)

        # Batch vector indicating graph id per node (needed for pooling)
        batch_vec = torch.arange(batch_size, device=self.device).repeat_interleave(num_nodes)

        inputs = Data(
            x=x,
            edge_index=edge_index_rep,
            batch=batch_vec,
        )

        labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        return inputs, labels

    @property
    def number_of_synapses(self):
        return self.synaptic_matrix.shape[0]

    def recreate_voronoi_cells(self):
        self.voronoi_cells.regenerate_random_centers()
        self.tesselated_neurons = self.voronoi_cells.get_tesselated_neurons()
        self.voronoi_indices = self.voronoi_cells.get_image_indices()
        self.voronoi_indices_torch = torch.tensor(self.voronoi_indices, device=self.device, dtype=torch.long)
        # Update fast-mapping tables because Voronoi indices changed
        self._build_neuron_mappings()

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

    # ---------------------------------------------------------------------
    # Fast, NumPy-based pipeline (no pandas) used during training
    # ---------------------------------------------------------------------

    def _build_neuron_mappings(self):
        """Create vectorised look-up tables mapping neuron index → (cell, channel).

        The neuron index (*index_id*) order is determined by ``self.root_ids``.
        For neurons that are **not** in ``self.tesselated_neurons`` (i.e. have no
        Voronoi cell), ``cell_idx`` is set to ``-1`` so that their activation is
        forced to zero.
        """

        import numpy as _np

        num_neurons = len(self.root_ids)
        self._neuron_cell_idx = np.full(num_neurons, -1, dtype=np.int32)
        self._neuron_channel_idx = np.zeros(num_neurons, dtype=np.int8)

        # Map cell_type to channel: r=0, g=1, b=2, mean=3
        channel_map = {"R1-6": 3, "R7": 2, "R8p": 1, "R8y": 0}

        tesselated = self.tesselated_neurons.copy()
        tesselated["root_id"] = tesselated["root_id"].astype("string")

        mapping_df = (
            self.root_ids.merge(
                tesselated[["root_id", "voronoi_indices", "cell_type"]],
                on="root_id",
                how="left",
            )
            .sort_values("index_id")
        )

        valid_mask = ~mapping_df["voronoi_indices"].isna()
        valid_indices = mapping_df[valid_mask]["index_id"].values.astype(int)
        self._neuron_cell_idx[valid_indices] = mapping_df[valid_mask]["voronoi_indices"].astype(int).values

        self._neuron_channel_idx[valid_indices] = mapping_df[valid_mask]["cell_type"].map(channel_map).astype(int).values

        # Torch versions on device for quick gather
        device = self.device
        self._neuron_cell_idx_torch = torch.tensor(self._neuron_cell_idx, device=device, dtype=torch.long)
        self._neuron_channel_idx_torch = torch.tensor(self._neuron_channel_idx, device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # Vectorised Voronoi means
    # ------------------------------------------------------------------

    def _get_voronoi_means_torch(self, processed_imgs):
        """Return ndarray (B, num_cells, 4) with mean r, g, b, mean-channel.

        The implementation loops over the batch dimension but stays within NumPy
        and avoids pandas/DataFrame overhead.
        """

        # Keep the original dtype of *processed_imgs* to avoid an up-cast that doubles
        # the temporary memory footprint. When running on CUDA we typically receive
        # *float16* tensors from *_process_images_torch*; staying in half precision
        # during the scatter operations cuts the required memory in half and is
        # fully supported by ``torch_scatter``.
        processed_t = processed_imgs if torch.is_tensor(processed_imgs) else torch.from_numpy(processed_imgs).to(self.device)
        
        B, P, _ = processed_t.shape
        cell_idx = processed_t[0, :, 4].long()
        num_cells = int(cell_idx.max().item()) + 1
        
        channels = processed_t[:, :, :4]  # B, P, 4
        cell_indices = processed_t[:, :, 4].long()  # B,P
        
        means = torch.zeros((B, num_cells, 4), device=self.device, dtype=channels.dtype)
        
        for i in range(B):
            idx = cell_indices[i]
            sums = scatter_add(channels[i], idx.unsqueeze(-1).expand(-1, 4), dim=0, dim_size=num_cells)
            counts = scatter_add(torch.ones_like(idx, dtype=channels.dtype), idx, dim=0, dim_size=num_cells).clamp(min=1.0)
            means[i] = sums / counts.unsqueeze(-1)
        
        return means  # stay on device

    # ------------------------------------------------------------------
    # Vectorised neuron activation computation
    # ------------------------------------------------------------------

    def _calculate_neuron_activations_torch(self, voronoi_means):
        """Map per-cell colour averages to neuron activations (NumPy).

        Parameters
        ----------
        voronoi_means : np.ndarray
            Shape ``(B, num_cells, 4)`` where channels follow the order
            ``r, g, b, mean``.
        """

        # Convert to torch on correct device
        vm = voronoi_means.to(self.device)

        r, g, b, m = vm[..., 0], vm[..., 1], vm[..., 2], vm[..., 3]

        if self.inhibitory_r7_r8:
            mask = b > torch.maximum(r, g)
            r = torch.where(mask, 0, r)
            g = torch.where(mask, 0, g)
            b = torch.where(mask, b, 0)

        channels = torch.stack([r, g, b, m], dim=-1)  # B, num_cells, 4

        valid_mask = self._neuron_cell_idx_torch != -1
        valid_neuron_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze()
        cell_indices = self._neuron_cell_idx_torch[valid_mask]
        ch_indices = self._neuron_channel_idx_torch[valid_mask]

        gathered = channels[:, cell_indices, ch_indices]  # B, N_valid

        # Work in the model dtype (typically *float32*) to avoid downstream dtype
        # mismatches while still benefiting from the reduced precision earlier.
        activation = torch.zeros(len(self._neuron_cell_idx_torch), vm.shape[0], device=self.device, dtype=self.dtype)
        activation[valid_neuron_idx, :] = gathered.transpose(0, 1).to(self.dtype)

        return activation

    # ------------------------------------------------------------------
    # Fast torch-based image processing
    # ------------------------------------------------------------------

    def _process_images_torch(self, imgs_input):
        """GPU version of process_images (resize + colour repeat).

        Accepts numpy uint8 images of shape (B,H,W) or (B,H,W,3).
        Returns torch tensor float32 (B, H, W, 3) on device; no /255 scaling.
        """
        # Use half precision on GPU to slash memory usage; fall back to full
        # precision on CPU where memory is less constrained.
        target_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if isinstance(imgs_input, np.ndarray):
            imgs_t = torch.from_numpy(imgs_input).to(self.device, dtype=target_dtype)
        else:
            imgs_t = imgs_input.to(self.device, dtype=target_dtype)

        if imgs_t.ndim == 3:
            imgs_t = imgs_t.unsqueeze(-1)  # grayscale

        B, H, W, C = imgs_t.shape
        assert C == 3, "Images must have 3 channels"

        if H != 512 or W != 512:
            imgs_t = imgs_t.permute(0,3,1,2)
            # Interpolation can be performed in half precision as well.
            imgs_t = F.interpolate(imgs_t, size=(512,512), mode="bilinear", align_corners=False)
            imgs_t = imgs_t.permute(0,2,3,1)

        imgs_t = imgs_t / 255.0

        # Flatten spatial to B,P,3
        imgs_t = imgs_t.reshape(B, -1, 3)

        mean_channel = imgs_t.mean(dim=2, keepdim=True)  # B,P,1
        imgs_t = torch.cat([imgs_t, mean_channel], dim=2)  # B,P,4

        vor_idx = self.voronoi_indices_torch.view(1, -1, 1).expand(B, -1, 1).to(target_dtype)
        imgs_t = torch.cat([imgs_t, vor_idx], dim=2)  # B,P,5

        return imgs_t

    def _preprocess_images_torch(self, imgs_np):
        """GPU version of preprocess_images (resize + colour repeat).

        Accepts numpy uint8 images of shape (B,H,W) or (B,H,W,3).
        Returns torch tensor float32 (B, H, W, 3) on device; no /255 scaling.
        """
        imgs_t = torch.from_numpy(imgs_np).to(self.device)

        if imgs_t.ndim == 3:  # grayscale
            imgs_t = imgs_t.unsqueeze(-1).repeat(1,1,1,3)

        B,H,W,C = imgs_t.shape
        if H != 512 or W != 512:
            imgs_t = imgs_t.permute(0,3,1,2).float()
            imgs_t = F.interpolate(imgs_t, size=(512,512), mode="bilinear", align_corners=False)
            imgs_t = imgs_t.permute(0,2,3,1)

        return imgs_t.float()

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
