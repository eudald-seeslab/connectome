import os

import numpy as np
import torch

import wandb

from flyvision import utils


def hex_to_square_grid(color, hex_size=15):
    # Coordinate mapping
    u, v = utils.hex_utils.get_hex_coords(hex_size)

    grid_size = hex_size * 2 + 1
    square_x = u + hex_size
    square_y = hex_size - v
    grid = np.full((grid_size, grid_size), np.nan)

    for hex_u, hex_v, hex_color in zip(square_x, square_y, color):
        grid[hex_u, hex_v] = hex_color
    return grid


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
                hex_to_square_grid(torch.mul(torch.div(la_vals, max_la), 256)),
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
