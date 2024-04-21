import wandb

from from_retina_to_connectome_utils import hex_to_square_grid
from logs_to_wandb import log_images_to_wandb, log_running_stats_to_wandb


class WandBLogger:
    def __init__(
        self,
        project_name,
        config,
        enabled=True,
        log_images_every=100,
        cell_type_plot="TmY18",
        last_good_frame=2,
    ):
        self.project_name = project_name
        self.config = config
        self.enabled = enabled
        self.log_images_every = log_images_every
        self.cell_type_plot = cell_type_plot
        self.last_good_frame = last_good_frame
        self.initialized = False

    def initialize(self):
        if self.enabled and not self.initialized:
            wandb.init(project=self.project_name, config=self.config)
            self.initialized = True

    def log_images(
        self,
        iteration,
        layer_activations,
        batch_sequences,
        rendered_sequences,
        batch_files,
    ):
        if self.enabled and iteration % self.log_images_every == 0:
            try:
                # Process layer activations to prepare for logging
                transformed_activation = hex_to_square_grid(
                    layer_activations[0][self.cell_type_plot]
                    .squeeze()[-self.last_good_frame]
                    .cpu()
                    .numpy()
                )
                # Log the images to wandb
                log_images_to_wandb(
                    batch_sequences[0],
                    rendered_sequences[0],
                    (transformed_activation,),
                    batch_files[0],
                    frame=self.last_good_frame,
                    cell_type=self.cell_type_plot,
                )
            except Exception as e:
                print(f"Error logging images to wandb: {e}. Continuing...")

    def log_metrics(self, iteration, running_loss, total_correct, total, results):
        if self.enabled:
            try:
                log_running_stats_to_wandb(
                    0, iteration, running_loss, total_correct, total, results
                )
            except Exception as e:
                print(f"Error logging running stats to wandb: {e}. Continuing...")
