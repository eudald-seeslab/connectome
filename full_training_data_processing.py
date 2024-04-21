import torch

from flyvision.utils.activity_utils import LayerActivity
from from_image_to_video import image_paths_to_sequences
from from_retina_to_connectome_funcs import compute_voronoi_averages, from_retina_to_connectome
from from_video_to_training_batched_funcs import paths_to_labels


DT = 1 / 100
LAST_GOOD_FRAME = 2


class FullModelsDataProcessor:
    def __init__(self, wandb_logger, receptors, network, classification, final_retina_cells, normalize_voronoi_cells, root_id_to_index, dtype, DEVICE):
        self.wandb_logger = wandb_logger
        self.receptors = receptors
        self.network = network
        self.dt = DT
        self.classification = classification
        self.final_retina_cells = final_retina_cells
        self.last_good_frame = LAST_GOOD_FRAME
        self.normalize_voronoi_cells = normalize_voronoi_cells
        self.root_id_to_index = root_id_to_index
        self.dtype = dtype
        self.DEVICE = DEVICE

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
    