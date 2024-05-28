import os
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, cKDTree, voronoi_plot_2d

from complete_training_funcs import assign_cell_type

class VoronoiCells:
    pixel_num = 512
    ommatidia_size = 8
    data_cols = ["x_axis", "y_axis"]
    centers = None
    voronoi = None
    tree = None
    index_map = None
    color_map = {"R1-6": "gray", "R7": "red", "R8p": "green", "R8y": "blue"}

    def __init__(self, eye="right", neurons="all", voronoi_criteria="all"):
        self.img_coords = self.get_image_coords(self.pixel_num)
        self.neuron_data = self._get_visual_neurons_data(neurons, eye)
        # if we use the R7 neurons as centers, we can create them already,
        # since they are fixed
        if voronoi_criteria == "R7":
            self.centers = self.get_R7_neurons(self.neuron_data)
            self.voronoi = Voronoi(self.centers)
            self.tree = cKDTree(self.centers)

    def query_points(self, points):
        _, indices = self.tree.query(points)

        return indices

    def get_tesselated_neurons(self):
        neuron_data = self.neuron_data.copy()

        neuron_indices = self.query_points(neuron_data[self.data_cols].values)

        neuron_data["voronoi_indices"] = neuron_indices
        neuron_data["cell_type"] = neuron_data.apply(assign_cell_type, axis=1)

        return neuron_data

    def get_image_indices(self):
        return self.query_points(self.img_coords)

    @staticmethod
    def _get_visual_neurons_data(neurons, side="right"):
        file = f"{side}_visual_positions_{neurons}_neurons.csv"
        data_path = os.path.join("adult_data", file)

        return pd.read_csv(data_path).drop(columns=["x", "y", "z", "PC1", "PC2"])

    def regenerate_random_centers(self):

        n_centers = self.neuron_data.shape[0] // self.ommatidia_size
        self.centers = self.neuron_data[self.data_cols].sample(n_centers).values
        self.voronoi = Voronoi(self.centers)
        self.tree = cKDTree(self.centers)

    def get_R7_neurons(self, neuron_data):
        return neuron_data[neuron_data["cell_type"] == "R7"][self.data_cols].values

    @ staticmethod
    def get_image_coords(pixel_num):
        coords = np.array(
            np.meshgrid(
                np.arange(pixel_num), np.arange(pixel_num), 
                indexing="xy")
                ).reshape(2, -1).T

        # Invert "y" to start from the bottom, like with the neurons
        coords[:, 1] = pixel_num - 1 - coords[:, 1]
        return coords

    def _plot_voronoi_cells(self, ax):
        voronoi_plot_2d(
            self.voronoi,
            ax=ax,
            show_vertices=False,
            line_colors="orange",
            line_width=1,
            line_alpha=0.6,
            point_size=2,
        )

    def plot_voronoi_cells_with_neurons(self, neuron_data, ax):

        neuron_data["color"] = neuron_data["cell_type"].apply(lambda x: self.color_map[x])

        self._plot_voronoi_cells(ax)

        # reverse the y-axis
        neuron_data["y_axis"] = self.pixel_num - 1 - neuron_data["y_axis"]

        # Visual neurons (complicated because we want the legend and matplotlib is stupid)
        for cell_type, color in self.color_map.items():
            points = neuron_data[neuron_data["cell_type"] == cell_type]
            ax.scatter(
                points["x_axis"], points["y_axis"], color=color, s=5, label=cell_type
            )
        ax.legend(title="", loc="lower left")

    def plot_voronoi_cells_with_image(self, image, ax):

        # Display the image
        ax.imshow(image, extent=[0, 512, 0, 512])

        # Plot Voronoi diagram
        self._plot_voronoi_cells(ax)

    def plot_neuron_activations(self, n_acts, ax):
        self._plot_voronoi_cells(ax)

        # n_acts["voronoi_indices"] = n_acts["voronoi_indices"].map(self.index_map).astype(int)

        rgb_values = (
            n_acts.loc[:, ["voronoi_indices", "cell_type", "activation"]]
            .groupby(["voronoi_indices", "cell_type"])
            .mean()
            .pivot_table(
                index="voronoi_indices", columns="cell_type", values="activation"
            )
            .fillna(0)
            .rename(columns={"R1-6": "mean", "R7": "b", "R8p": "g", "R8y": "r"})
        )

        rgb_values["b"] = self.get_colour_average(rgb_values, "b")
        rgb_values["g"] = self.get_colour_average(rgb_values, "g")
        rgb_values["r"] = self.get_colour_average(rgb_values, "r")

        # Fill Voronoi regions with colors based on aggregated RGB values
        for region_index in self.voronoi.point_region:
            region = self.voronoi.regions[region_index]
            if not -1 in region:
                polygon = [self.voronoi.vertices[i] for i in region]
                color = rgb_values.loc[region_index, ["r", "g", "b"]]
                ax.fill(*zip(*polygon), color=color)

    @staticmethod
    def get_colour_average(values, colour):
        return (values[colour] + values["mean"]) / 2
