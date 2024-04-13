import os

import torch

import wandb

from from_retina_to_connectome_utils import hex_to_square_grid


def log_images_to_wandb(bs, rs, la, img_path, frame, cell_type):
    la_vals = la[cell_type][:, -frame, :].squeeze()
    max_la = la_vals.max()
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
                torch.mul(torch.div(la_vals, max_la), 255),
                caption=f"{cell_type} activation",
            ),
        }
    )


def log_original_to_wandb(vals, img_path):
    wandb.log(
        {
            "Original image": wandb.Image(
                vals, caption=f"Original image {os.path.basename(img_path)}"
            ),
        }
    )


def log_running_stats_to_wandb(
    epoch_, iteration, running_loss_, total_correct_, total_, results_
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


def log_validation_stats_to_wandb(
    running_loss_, total_correct_, total_, results_, weber_plot_
):
    wandb.log(
        {
            "Validation loss": running_loss_ / total_,
            "Validation accuracy": total_correct_ / total_,
            "Validation results": wandb.Table(dataframe=results_),
            "Weber Fraction Plot": wandb.Image(weber_plot_),
        }
    )
