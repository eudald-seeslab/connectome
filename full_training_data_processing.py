import pathlib
from random import sample
import pandas as pd
import torch
from scipy.sparse import load_npz

import flyvision
from flyvision.utils.activity_utils import LayerActivity
from flyvision_ans import FINAL_CELLS
from from_image_to_video import image_paths_to_sequences
from from_retina_to_connectome_funcs import compute_voronoi_averages, from_retina_to_connectome
from from_retina_to_connectome_utils import get_files_from_directory, paths_to_labels, vector_to_one_hot


DT = 1 / 100
LAST_GOOD_FRAME = 2
FINAL_RETINA_CELLS = FINAL_CELLS


class FullModelsDataProcessor:
    extent = 15
    kernel_size = 13
    dt = DT
    last_good_frame = LAST_GOOD_FRAME
    final_retina_cells = FINAL_RETINA_CELLS

    def __init__(
        self,
        wandb_logger,
        normalize_voronoi_cells,
        dtype,
        DEVICE,
        sparse_layout,
    ):
        self.wandb_logger = wandb_logger
        self.receptors = flyvision.rendering.BoxEye(
            extent=self.extent, kernel_size=self.kernel_size
        )
        self.cwd = pathlib.Path().resolve()
        self.data_dir = self.cwd / "adult_data"
        network_view = flyvision.NetworkView(flyvision.results_dir / "opticflow/000/0000")
        self.network = network_view.init_network(chkpt="best_chkpt")
        self.root_id_to_index = pd.read_csv(self.data_dir / "root_id_to_index.csv")
        self.classification = pd.read_csv(
            self.data_dir / "classification_clean.csv"
        )
        self.normalize_voronoi_cells = normalize_voronoi_cells
        self.dtype = dtype
        self.DEVICE = DEVICE
        self.sparse_layout = sparse_layout

    def get_videos(self, data_dir, small, small_length):
        videos = get_files_from_directory(self.cwd / data_dir)
        assert len(videos) > 0, f"No videos found in {data_dir}."

        if small:
            videos = sample(videos, small_length)

        return videos

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
        decision_making_vector = torch.tensor(rational_neurons.values.squeeze(), dtype=self.dtype).detach()
        return vector_to_one_hot(decision_making_vector, self.dtype, self.sparse_layout).to(
            self.DEVICE
        )

    def synaptic_matrix(self):
        return load_npz(self.data_dir / "good_synaptic_matrix.npz")

    def process_full_models_data(self, i, batch_files):
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

        inputs = torch.tensor(activation_df.values, dtype=self.dtype, device=self.DEVICE)
        labels = torch.tensor(labels, dtype=self.dtype, device=self.DEVICE)
        return labels, inputs
