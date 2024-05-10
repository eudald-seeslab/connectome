from concurrent.futures import ThreadPoolExecutor
import os
from imageio.v3 import imread
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree, Voronoi, voronoi_plot_2d
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch


num_workers = os.cpu_count() - 2


class VoronoiCells:
    pixel_num = 512
    ommatidia_size = 8
    tree = None
    data_cols = ["x_axis", "y_axis"]
    centers = None

    def __init__(self, voronoi_criteria="all"):
        self.voronoi_criteria = voronoi_criteria
        self.img_coords = self.create_image_coords(self.pixel_num)

    def get_tesselated_neurons(self):
        neuron_data = self._get_neuronal_data()
        self.generate_voronoi_centers(neuron_data)
        self.tree = cKDTree(self.centers)
        _, neuron_indices = self.tree.query(neuron_data[self.data_cols].values)

        neuron_data["voronoi_indices"] = neuron_indices
        neuron_data["cell_type"] = neuron_data.apply(assign_cell_type, axis=1)

        return neuron_data

    def get_image_indices(self):
        assert self.tree is not None, "You need to call get_neuron_indices first."

        _, img_indices = self.tree.query(self.img_coords)
        return img_indices

    def _get_neuronal_data(self):
        if self.voronoi_criteria == "all":
            neuron_file = "right_visual_positions_all_neurons.csv"
        elif self.voronoi_criteria == "selected":
            neuron_file = "right_visual_positions_selected_neurons.csv"
        data_path = os.path.join("adult_data", neuron_file)

        return pd.read_csv(data_path).drop(columns=["x", "y", "z", "PC1", "PC2"])

    def generate_voronoi_centers(self, neuron_data):
        if self.voronoi_criteria == "all":
            n_centers = neuron_data.shape[0] // self.ommatidia_size
            self.centers = neuron_data[self.data_cols].sample(n_centers).values
        elif self.voronoi_criteria == "R7":
            self.centers = neuron_data[neuron_data["cell_type"] == "R7"][
                self.data_cols
            ].values
        else:
            raise ValueError(
                f"Voronoi criteria {self.voronoi_criteria} not implemented."
            )

    @staticmethod
    def create_image_coords(pixel_num):
        """Create a grid of pixel_num x pixel_num coordinates."""
        return (
            np.array(
                np.meshgrid(np.arange(pixel_num), np.arange(pixel_num), indexing="ij")
            )
            .reshape(2, -1)
            .T
        )

    def plot_voronoi_cells_with_neurons(self, neuron_data):

        color_map = {"R1-6": "gray", "R7": "red", "R8p": "green", "R8y": "blue"}
        neuron_data["color"] = neuron_data["cell_type"].apply(lambda x: color_map[x])

        fig, ax = plt.subplots(figsize=(8, 8))
        # Voronoi mesh
        voronoi_plot_2d(
            Voronoi(self.centers),
            ax=ax,
            show_vertices=False,
            line_colors="orange",
            line_width=2,
            line_alpha=0.6,
            point_size=2,
        )

        # Visual neurons (complicated because we want the legend and matplotlib is stupid)
        for cell_type, color in color_map.items():
            points = neuron_data[neuron_data["cell_type"] == cell_type]
            ax.scatter(points["x_axis"], points["y_axis"], color=color, s=5, label=cell_type)
        ax.legend(title = "", loc="lower left")

        return fig

    def plot_voronoi_cells_with_image(self, image):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Display the image
        ax.imshow(image, extent=[0, 512, 0, 512])

        # Plot Voronoi diagram
        voronoi_plot_2d(
            Voronoi(self.centers),
            ax=ax,
            show_vertices=False,
            show_points=False,
            line_colors="orange",
            line_width=2,
            line_alpha=0.6,
        )

        return fig


def import_images(img_paths):
    # Read images using a thread pool to speed up disk I/O operations
    with ThreadPoolExecutor() as executor:
        imgs = list(executor.map(imread, img_paths))

    # Stack images into a single NumPy array
    imgs = np.stack(imgs, axis=0)
    return imgs


def process_images(imgs, voronoi_indices):
    # Reshape images: [n_images, n_pixels*n_pixels, n_channels]
    imgs = imgs.reshape(imgs.shape[0], -1, imgs.shape[-1])

    # Convert to 0-1 scale
    imgs = imgs / 255.0

    # Calculate mean of channels and stack it
    mean_channel = np.mean(imgs, axis=2, keepdims=True)
    imgs = np.concatenate([imgs, mean_channel], axis=2)

    # Prepare Voronoi indices
    # Shape it to (1, n_pixels*n_pixels, 1)
    voronoi_indices = voronoi_indices.reshape(1, -1, 1)
    # Repeat for all images
    voronoi_indices = np.repeat(voronoi_indices, imgs.shape[0], axis=0)

    # Append Voronoi indices
    imgs = np.concatenate([imgs, voronoi_indices], axis=2)

    return imgs


def get_voronoi_averages(processed_imgs):
    dfs = []
    for img in processed_imgs:
        img = pd.DataFrame(img)
        img.columns = ["r", "g", "b", "mean", "cell"]
        dfs.append(img.groupby("cell").mean())
    return dfs


def assign_cell_type(row):
    # In right_visual, when cell_type is R8, we can have two types of cells
    # R8p (30%) and R8y (70%). Let's create a new column and randomly assign
    # the cell type to each cell.
    if row["cell_type"] == "R8":
        return "R8p" if np.random.rand(1) < 0.3 else "R8y"
    return row["cell_type"]


def get_activation_from_cell_type(row):
    match row["cell_type"]:
        case "R1-6":
            return row["mean"]
        case "R7":
            return row["b"]
        case "R8p":
            return row["g"]
        case "R8y":
            return row["r"]
        case _:
            raise ValueError("Invalid cell type")


def get_neuron_activations(right_visual, voronoi_average):
    neuron_activations = right_visual.merge(
        voronoi_average, left_on="voronoi_indices", right_index=True
    )
    neuron_activations["activation"] = neuron_activations.apply(
        get_activation_from_cell_type, axis=1
    )
    return neuron_activations.set_index("root_id")[["activation"]]


def get_side_decision_making_vector(right_root_ids, side):
    cell_type_rational = pd.read_csv("data/cell_type_rational_short.csv")
    # get the cell types with rational = 1
    rational_cell_types = cell_type_rational[cell_type_rational["rational"] == 1][
        "cell_type"
    ]
    right_neurons = pd.read_csv("adult_data/classification.csv")
    right_neurons = right_neurons[right_neurons["side"] == side]
    right_neurons = right_neurons[
        right_neurons["root_id"].isin(right_root_ids["root_id"])
    ]
    rational_neurons = right_neurons[
        right_neurons["cell_type"].isin(rational_cell_types)
    ]
    temp = right_root_ids.merge(rational_neurons, on="root_id", how="left")
    return torch.tensor(
        temp.assign(rational=np.where(temp["side"].isna(), 0, 1))["rational"].values
    )
