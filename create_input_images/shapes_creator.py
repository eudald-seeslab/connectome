import os
import sys
import math
import random
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import config

random.seed(1714)


class ShapesGenerator:
    """
    A class for generating images with geometric shapes for machine learning tasks.

    Attributes:
        shape (str): The shape of the geometric objects.
        train_num (int): The number of training images to generate.
        test_num (int): The number of test images to generate.
        min_radius (int): The minimum radius of the shapes.
        max_radius (int): The maximum radius of the shapes.
        jitter (bool): Whether to add jitter to the shapes.
    """

    colour_1: str = "yellow"
    colour_2: str = "blue"
    background_colour = "#808080"
    boundary_width = 5
    colours = {colour_1: "#fffe04", colour_2: "#0003f9"}
    img_paths = {}

    def __init__(self, shape, train_num, test_num, min_radius, max_radius, jitter):
        self.img_dir = (
            f"images/{shape}_{min_radius}_{max_radius}{'_jitter' if jitter else ''}"
        )

        self.shape = shape
        self.train_num = train_num
        self.test_num = test_num
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.jitter = jitter

    def create_dirs(self):
        """Creates necessary directories to store generated images."""
        for t in ["train", "test"]:
            self.img_paths[f"{t}_1"] = os.path.join(self.img_dir, t, self.colour_1)
            self.img_paths[f"{t}_2"] = os.path.join(self.img_dir, t, self.colour_2)
        [os.makedirs(p, exist_ok=True) for p in self.img_paths.values()]

    def get_vertices(self, shape: str, center: tuple, radius: int) -> list:
        """Calculate vertices of the shapes based on the given parameters."""
        if shape == "circle":
            return [
                (center[0] - radius, center[1] - radius),
                (center[0] + radius, center[1] + radius),
            ]
        elif shape == "triangle":
            return [
                (center[0], center[1] - radius),
                (center[0] - radius, center[1] + radius),
                (center[0] + radius, center[1] + radius),
            ]
        elif shape == "square":
            return [
                (center[0] - radius, center[1] - radius),
                (center[0] + radius, center[1] - radius),
                (center[0] + radius, center[1] + radius),
                (center[0] - radius, center[1] + radius),
            ]
        elif shape == "star":
            vertices = []
            for i in range(5):
                angle_deg = -90 + (i * 72)  # 72 degrees between each point
                angle_rad = math.radians(angle_deg)
                vertices.append(
                    (
                        center[0] + radius * math.cos(angle_rad),
                        center[1] + radius * math.sin(angle_rad),
                    )
                )
                angle_deg += 36  # Repeat for inner vertices
                angle_rad = math.radians(angle_deg)
                vertices.append(
                    (
                        center[0] + (radius / 2) * math.cos(angle_rad),
                        center[1] + (radius / 2) * math.sin(angle_rad),
                    )
                )
            return vertices
        else:
            raise ValueError(f"Shape {shape} not implemented.")

    def draw_shape(self, shape: str, radius: int, colour: str, jitter: bool = False):
        """Draws a single shape on an image and saves it to the appropriate directory."""
        pixels_x, pixels_y = 512, 512
        image = Image.new("RGB", (pixels_x, pixels_y), color=self.background_colour)
        draw = ImageDraw.Draw(image)

        x_space = int(pixels_x / 2 - radius) if jitter else 0
        y_space = int(pixels_y / 2 - radius) if jitter else 0
        dist_x = random.randint(-x_space, x_space)
        dist_y = random.randint(-y_space, y_space)
        center = (int(pixels_x / 2) + dist_x, int(pixels_y / 2) + dist_y)

        vertices = self.get_vertices(shape, center, radius)
        if shape == "circle":
            draw.ellipse(vertices, fill=self.colours[colour])
        else:
            draw.polygon(vertices, fill=self.colours[colour])

        return image, int(np.sqrt(dist_x**2 + dist_y**2))

    def save_image(self, image, radius, dist_from_center, it, path):
        file_path = os.path.join(
            path,
            f"{self.shape}_{radius}_{dist_from_center}_{it}.png",
        )
        image.save(file_path)

    def generate_images(self):
        """Generate sets of training and test images."""
        min_radius, max_radius = self.min_radius, self.max_radius
        shape = self.shape
        jitter = self.jitter

        self.create_dirs()
        radius_range = max_radius - min_radius
        print(
            f"Creating images for {shape} with {radius_range * self.train_num} "
            f"training images and {radius_range * self.test_num} test images."
        )

        for i in tqdm(range(self.train_num)):
            for r in range(min_radius, max_radius):
                image, dist = self.draw_shape(shape, r, self.colour_1, jitter)
                self.save_image(image, r, dist, i, self.img_paths["train_1"])
                image, dist = self.draw_shape(shape, r, self.colour_2, jitter)
                self.save_image(image, r, dist, i, self.img_paths["train_2"])

        for i in tqdm(range(self.test_num)):
            for r in range(min_radius, max_radius):
                image, dist = self.draw_shape(shape, r, self.colour_1, jitter)
                self.save_image(image, r, dist, i, self.img_paths["test_1"])
                image, dist = self.draw_shape(shape, r, self.colour_2, jitter)
                self.save_image(image, r, dist, i, self.img_paths["test_2"])


if __name__ == "__main__":
    img_gen = ShapesGenerator(
        shape=config.SHAPE,
        train_num=config.TRAIN_NUM,
        test_num=config.TEST_NUM,
        min_radius=config.MIN_RADIUS,
        max_radius=config.MAX_RADIUS,
        jitter=config.JITTER,
    )
    img_gen.generate_images()
