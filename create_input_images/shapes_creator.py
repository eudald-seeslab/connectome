import argparse
import os
import math
import random
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import data_config as config

random.seed(1714)


COLOUR_MAP = {
    "black": "#000000",
    "blue": "#0003f9",
    "yellow": "#fffe04",
    "red": "#ff0000",
    "green": "#00ff00",
}


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
        colours (list): The colours of the shapes.
    """

    background_colour = "#000000"  # #808080 for gray
    boundary_width = 5
    img_paths = {}

    def __init__(self, shape, train_num, test_num, min_radius, max_radius, jitter, colours=["blue", "yellow"]):
        self.img_dir = (
            f"images/{shape}_{min_radius}_{max_radius}{'_jitter' if jitter else ''}"
        )
        try:
            self.colours = {col: COLOUR_MAP[col] for col in colours}
        except KeyError:
            raise ValueError(f"One or more of your colours are not implemented. Choices are {', '.join(COLOUR_MAP.keys())}.")
        
        self.shape = shape
        self.train_num = train_num
        self.test_num = test_num
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.jitter = jitter

    def create_dirs(self):
        """Creates necessary directories to store generated images."""
        for t in ["train", "test"]:
            for i, col in enumerate(self.colours.keys()):
                self.img_paths[f"{t}_{i + 1}"] = os.path.join(self.img_dir, t, col)

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
            draw.ellipse(vertices, fill=colour)
        else:
            draw.polygon(vertices, fill=colour)

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
                for i, col in enumerate(self.colours.values()):
                    image, dist = self.draw_shape(shape, r, col, jitter)
                    self.save_image(image, r, dist, i, self.img_paths[f"train_{i + 1}"])

        for i in tqdm(range(self.test_num)):
            for r in range(min_radius, max_radius):
                for i, col in enumerate(self.colours.values()):
                    image, dist = self.draw_shape(shape, r, col, jitter)
                    self.save_image(image, r, dist, i, self.img_paths[f"test_{i + 1}"])

    def create_dirs_all_classes(self):
        """Create directories for all classes of shapes."""
        for t in ["train", "test"]:
            for i, cl in enumerate(config.CLASSES):
                self.img_paths[f"{t}_{i}"] = os.path.join(self.img_dir, t, cl)
        [os.makedirs(p, exist_ok=True) for p in self.img_paths.values()]

    def generate_all_classes(self):
        """Generate all classes of shapes."""
        jitter = self.jitter
        min_radius, max_radius = self.min_radius, self.max_radius
        shapes = config.CLASSES
        colour = self.colours[list(self.colours.keys())[0]]

        self.create_dirs_all_classes()

        radius_range = max_radius - min_radius
        print(
            f"Creating images for {', '.join(shapes)} with {len(shapes) * radius_range * self.train_num} "
            f"training images and {len(shapes) * radius_range * self.test_num} test images."
        )
        for i in tqdm(range(self.train_num)):
            for j, shape in enumerate(shapes):
                for r in range(min_radius, max_radius):
                    image, dist = self.draw_shape(shape, r, colour, jitter)
                    self.save_image(image, r, dist, i, self.img_paths[f"train_{j}"])

        for i in tqdm(range(self.test_num)):
            for j, shape in enumerate(shapes):
                for r in range(min_radius, max_radius):
                    image, dist = self.draw_shape(shape, r, colour, jitter)
                    self.save_image(image, r, dist, i, self.img_paths[f"test_{j}"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate shapes."
    )
    parser.add_argument(
        "--all_shapes",
        action='store_true',
        help="Are we trying to differentiate among several shapes?",
    )
    args = parser.parse_args()

    img_gen = ShapesGenerator(
        shape=config.SHAPE,
        train_num=config.TRAIN_NUM,
        test_num=config.TEST_NUM,
        min_radius=config.MIN_RADIUS,
        max_radius=config.MAX_RADIUS,
        jitter=config.JITTER,
        colours=config.COLOURS,
    )
    if args.all_shapes:
        img_gen.generate_all_classes()
    else:
        img_gen.generate_images()
