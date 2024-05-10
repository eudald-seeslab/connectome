import os
import numpy as np
import wandb
from wandb import AlertLevel
import config
from deprecated.from_retina_to_connectome_utils import hex_to_square_grid


MODEL_CONFIG = {
    "debugging": config.debugging,
    "num_epochs": config.num_epochs,
    "batch_size": config.batch_size,
    "dropout": config.dropout,
    "base_lr": config.base_lr,
    "weight_decay": config.weight_decay,
    "num_connectome_passes": config.NUM_CONNECTOME_PASSES,
}


class WandBLogger:
    def __init__(self, project_name):
        self.project_name = project_name
        self.model_config = MODEL_CONFIG
        self.enabled = config.wandb_
        self.log_images_every = config.wandb_images_every
        self.cell_type_plot = config.cell_type_plot
        self.last_good_frame = config.last_good_frame
        self.initialized = False

    def initialize(self):
        if self.enabled and not self.initialized:
            wandb.init(project=self.project_name, config=self.model_config)
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
                # TODO: this should probably be elsewhere
                transformed_activation = hex_to_square_grid(
                    layer_activations[0][self.cell_type_plot]
                    .squeeze()[-self.last_good_frame]
                    .cpu()
                    .numpy()
                )
                # Log the images to wandb
                self.log_images_func(
                    batch_sequences[0],
                    rendered_sequences[0],
                    (transformed_activation,),
                    batch_files[0],
                )
            except Exception as e:
                print(f"Error logging images to wandb: {e}. Continuing...")

    def log_metrics(self, epoch, iteration, running_loss, total_correct, total, results):
        if self.enabled:
            try:
                self.log_running_stats(
                    epoch, iteration, running_loss, total_correct, total, results
                )
            except Exception as e:
                print(f"Error logging running stats to wandb: {e}. Continuing...")

    def log_images_func(self, bs, rs, la, img_path,):
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

    def log_original(self, vals, img_path, iteration):
        if self.enabled and iteration % self.log_images_every == 0:
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
        if self.enabled:
            wandb.log(
                {
                    "epoch": epoch_,
                    "iteration": iteration + 1,
                    "loss": running_loss_ / total_,
                    "accuracy": total_correct_ / total_,
                    # "results": wandb.Table(dataframe=results_),
                }
            )

    def log_validation_stats(
        self, running_loss_, total_correct_, total_, results_, plots
    ):
        if len(plots) > 0:
            plot_dict = {f"Plot {i}": wandb.Image(plot) for i, plot in enumerate(plots)}
        else:
            plot_dict = {}
        if self.enabled:
            wandb.log(
                {
                    "Validation loss": running_loss_ / total_,
                    "Validation accuracy": total_correct_ / total_,
                    "Validation results": wandb.Table(dataframe=results_),
                }
                | plot_dict
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
