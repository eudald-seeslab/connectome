import torch
from torch.utils.data import DataLoader, TensorDataset
from flyvision.utils.activity_utils import LayerActivity
from flyvision_ans import DECODING_CELLS
from from_image_to_video import image_paths_to_sequences
from from_retina_to_connectome_utils import layer_activations_to_decoding_images
from from_video_to_training_batched_funcs import paths_to_labels, select_random_videos
from logs_to_wandb import log_images_to_wandb


LAST_GOOD_FRAME = 2
DT = 1 / 100
CELL_TYPE_PLOT = "TmY18"


class DataBatchProcessor:
    def __init__(self, batch_size, receptors, network, store_to_wandb=False):
        self.batch_size = batch_size
        self.receptors = receptors
        self.network = network
        self.last_good_frame = LAST_GOOD_FRAME
        self.DECODING_CELLS = DECODING_CELLS
        self.cell_type_plot = CELL_TYPE_PLOT
        self.store_to_wandb = store_to_wandb
        self.dt = DT
        self.already_selected = []
        self.batch_files = []


    def preprocess_batch(self, batch_files):

        labels = paths_to_labels(batch_files)
        batch_sequences = image_paths_to_sequences(batch_files)
        rendered_sequences = self.receptors(batch_sequences)
        
        layer_activations = []
        for rendered_sequence in rendered_sequences:
            # rendered sequences are in RGB; move it to 0-1 for better training
            rendered_sequence = torch.div(rendered_sequence, 255)
            # TODO: try to run this on cpu to multithread it
            simulation = self.network.simulate(rendered_sequence[None], self.dt)
            layer_activations.append(
                LayerActivity(simulation, self.network.connectome, keepref=True)
            )
        
        decoding_images = layer_activations_to_decoding_images(layer_activations, self.last_good_frame, DECODING_CELLS)

        if self.store_to_wandb:
            da = decoding_images[0][self.DECODING_CELLS.index(self.cell_type_plot)]
            log_images_to_wandb(batch_sequences[0], rendered_sequences[0], da, batch_files[0], frame=self.last_good_frame, cell_type=self.cell_type_plot)
        
        # clean up the memory
        del rendered_sequences, layer_activations, batch_sequences, simulation, rendered_sequence
        torch.cuda.empty_cache()

        # update the internal state
        self.batch_files = batch_files

        return decoding_images, labels
    

def prepare_dataloader(images, labels_, dtype=torch.float):
    dataset = TensorDataset(torch.tensor(images, dtype=dtype), torch.tensor(labels_, dtype=dtype))
    # i'm dealing with the batches by hand, so batch size is all of them
    return DataLoader(dataset, batch_size=len(labels_), shuffle=False)
