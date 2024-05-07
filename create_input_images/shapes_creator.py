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
        val_num (int): The number of validation images to generate.
        min_radius (int): The minimum radius of the shapes.
        max_radius (int): The maximum radius of the shapes.
        jitter (bool): Whether to add jitter to the shapes.
    """

    colour_1: str = "yellow"
    colour_2: str = "blue"
    background_colour = "#808080"
    boundary_width = 5
    colours = {colour_1: "#fffe04", colour_2: "#0003f9"}

    def __init__(self, shape, train_num, val_num, min_radius, max_radius, jitter):
        img_dir = f"images/{shape}_{min_radius}_{max_radius}_{'jitter' if jitter else ''}"
        self.train_dir = os.path.join(img_dir, "train")
        self.val_dir = os.path.join(img_dir, "val")
        self.shape = shape
        self.train_num = train_num
        self.val_num = val_num
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.jitter = jitter

    def create_dirs(self):
        """Creates necessary directories to store generated images."""
        for col in [self.colour_1, self.colour_2]:
            os.makedirs(os.path.join(self.train_dir, col), exist_ok=True)
            os.makedirs(os.path.join(self.val_dir, col), exist_ok=True)

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

    def draw_shape(
        self, shape: str, radius: int, colour: str, iteration: int, jitter: bool = False, path=None
    ):
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
        draw.polygon(vertices, fill=self.colours[colour])

        file_path = os.path.join(
            path,
            f"{shape}_{radius}_{int(np.sqrt(dist_x ** 2 + dist_y ** 2))}_{iteration}.png",
        )
        image.save(file_path)

    def generate_images(self):
        """Generate sets of training and validation images."""
        min_radius, max_radius = self.min_radius, self.max_radius
        shape = self.shape
        jitter = self.jitter

        self.create_dirs()
        train_1 = os.path.join(self.train_dir, self.colour_1)
        train_2 = os.path.join(self.train_dir, self.colour_2)
        val_1 = os.path.join(self.val_dir, self.colour_1)
        val_2 = os.path.join(self.val_dir, self.colour_2)
        print(
            f"Creating images for {shape} with {self.train_num} training images and {self.val_num} validation images"
        )

        for i in tqdm(range(self.train_num)):
            for r in range(min_radius, max_radius):
                self.draw_shape(shape, r, self.colour_1, i, jitter, train_1)
                self.draw_shape(shape, r, self.colour_2, i, jitter, train_2)

        for i in tqdm(range(self.val_num)):
            for r in range(min_radius, max_radius):
                self.draw_shape(shape, r, self.colour_1, i, jitter, val_1)
                self.draw_shape(shape, r, self.colour_2, i, jitter, val_2)

        return train_1, train_2, val_1, val_2


if __name__ == "__main__":
    img_gen = ShapesGenerator(
        shape=config.SHAPE,
        train_num=config.TRAIN_NUM,
        val_num=config.VAL_NUM,
        min_radius=config.MIN_RADIUS,
        max_radius=config.MAX_RADIUS,
        jitter=config.JITTER,
    )
    train_dir_1, train_dir_2, test_dir_1, test_dir_2 = img_gen.generate_images()
