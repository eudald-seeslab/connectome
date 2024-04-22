import os
import numpy as np
import wandb
from wandb import AlertLevel
from from_retina_to_connectome_utils import hex_to_square_grid


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
                self.log_images(
                    batch_sequences[0],
                    rendered_sequences[0],
                    (transformed_activation,),
                    batch_files[0],
                )
            except Exception as e:
                print(f"Error logging images to wandb: {e}. Continuing...")

    def log_metrics(self, iteration, running_loss, total_correct, total, results):
        if self.enabled:
            try:
                self.log_running_stats(
                    0, iteration, running_loss, total_correct, total, results
                )
            except Exception as e:
                print(f"Error logging running stats to wandb: {e}. Continuing...")

    
    def log_images(self, bs, rs, la, img_path,):
        frame = self.last_good_frame
        wandb.log(
            {
                "Original image": wandb.Image(
                    bs[-frame], caption=f"Original image {os.path.basename(img_path)}"
                ),
                "Rendered sequence": wandb.Image(
                    hex_to_square_grid(rs[-frame].squeeze()),
                    caption=f"Rendered image",
                ),
                "Layer activation": wandb.Image(
                    la / np.nanmax(la) * 255,
                    caption=f"{self.cell_type_plot} activation",
                ),
            }
        )

    def log_original(self, vals, img_path):
        wandb.log(
            {
                "Original image": wandb.Image(
                    vals, caption=f"Original image {os.path.basename(img_path)}"
                ),
            }
        )

    def log_running_stats(
        self, epoch_, iteration, running_loss_, total_correct_, total_, results_
    ):
        wandb.log(
            {
                "epoch": epoch_,
                "iteration": iteration,
                "loss": running_loss_ / total_,
                "accuracy": total_correct_ / total_,
                "results": wandb.Table(dataframe=results_),
            }
        )

    def log_validation_stats(
        self, running_loss_, total_correct_, total_, results_, weber_plot_
    ):
        wandb.log(
            {
                "Validation loss": running_loss_ / total_,
                "Validation accuracy": total_correct_ / total_,
                "Validation results": wandb.Table(dataframe=results_),
                "Weber Fraction Plot": wandb.Image(weber_plot_),
            }
        )
    
    def send_crash(self, message):
        if self.enabled:
            wandb.alert(
                title=f"Error in run at {self.project_name}", 
                text=message,
                level=AlertLevel.ERROR
                )

    def finish(self):
        if self.enabled:
            wandb.finish()
