import argparse
import os
import math
import random
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

random.seed(1714)


COLOUR_MAP = {
    "black": "#000000",
    "blue": "#0003f9",
    "yellow": "#fffe04",
    "red": "#ff0000",
    "green": "#00ff00",
}

TRAIN_NUM = 200
TEST_NUM = 200
MIN_RADIUS = 80
MAX_RADIUS = 110
JITTER = True
COLOR_TASK_BASE_SHAPE = "circle"


class ShapesGenerator:
    """
    A class for generating images with geometric shapes for machine learning tasks.

    Attributes:
        shape (str): The shape of the geometric objects.
        colours (list): The colours of the shapes.
        task_type (str): The type of task to generate images for.
    """

    background_colour = "#000000"
    boundary_width = 5
    img_paths = {}

    def __init__(
        self,
        shapes,
        colors,
        task_type=None,
    ):
        self.shapes = shapes if isinstance(shapes, list) else [shapes]
        self.colors = {col: COLOUR_MAP[col] for col in colors}
        self.task_type = task_type

        # Set directory based on task
        if task_type == "two_shapes":
            self.img_dir = "images/two_shapes"
        elif task_type == "two_colors":
            self.img_dir = "images/two_colors"
        else:
            self.img_dir = f"images/{'_'.join(shapes)}_{'_'.join(colors)}"

        self.train_num = TRAIN_NUM
        self.test_num = TEST_NUM
        self.min_radius = MIN_RADIUS
        self.max_radius = MAX_RADIUS
        self.jitter = JITTER
        self.img_paths = {}

    def create_dirs(self):
        """Creates class directories for train and test sets."""
        for t in ["train", "test"]:
            if self.task_type == "two_shapes":
                classes = self.shapes  # classes are shapes
            elif self.task_type == "two_colors":
                classes = self.colors.keys()  # classes are colors
            else:
                # For custom, each shape-color combination is a class
                classes = [
                    f"{shape}_{color}" for shape in self.shapes for color in self.colors
                ]

            for class_name in classes:
                path_key = f"{t}_{class_name}"
                self.img_paths[path_key] = os.path.join(self.img_dir, t, class_name)
                os.makedirs(self.img_paths[path_key], exist_ok=True)

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

        # for bookkeeping
        distance = int(np.sqrt(dist_x**2 + dist_y**2))
        angle = int((np.arctan2(dist_y, dist_x) / np.pi + 1) * 180)

        vertices = self.get_vertices(shape, center, radius)
        if shape == "circle":
            draw.ellipse(vertices, fill=colour)
        else:
            draw.polygon(vertices, fill=colour)

        return image, distance, angle

    def save_image(self, image, shape, radius, dist_from_center, angle, it, path):
        file_path = os.path.join(
            path,
            f"{shape}_{radius}_{dist_from_center}_{angle}_{it}_v2.png",
        )
        image.save(file_path)

    def generate_images(self):
        """Generate all images for training and testing."""
        self.create_dirs()

        for phase, num_images in [("train", self.train_num), ("test", self.test_num)]:
            for i in tqdm(range(num_images)):
                for radius in range(self.min_radius, self.max_radius):
                    if self.task_type == "two_shapes":
                        # For shape recognition: each shape in yellow is a class
                        for shape in self.shapes:
                            image, dist, angle = self.draw_shape(shape, radius, self.colors['yellow'], self.jitter)
                            self.save_image(image, shape, radius, dist, angle, i, self.img_paths[f"{phase}_{shape}"])
                    elif self.task_type == "two_colors":
                        # For color recognition: circle in each color is a class
                        for color_name, color_code in self.colors.items():
                            image, dist, angle = self.draw_shape(
                                COLOR_TASK_BASE_SHAPE, radius, color_code, self.jitter
                            )
                            self.save_image(
                                image,
                                COLOR_TASK_BASE_SHAPE,
                                radius,
                                dist,
                                angle,
                                i,
                                self.img_paths[f"{phase}_{color_name}"],
                            )
                    else:
                        # For custom: each shape-color combination is a class
                        for shape in self.shapes:
                            for color_name, color_code in self.colors.items():
                                image, dist, angle = self.draw_shape(shape, radius, color_code, self.jitter)
                                class_name = f"{shape}_{color_name}"
                                self.save_image(
                                    image,
                                    shape,
                                    radius,
                                    dist,
                                    angle,
                                    i,
                                    self.img_paths[f"{phase}_{class_name}"],
                                )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate shapes for training visual models."
    )

    # Create mutually exclusive group for the two main use cases
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--shape_recognition",
        action="store_true",
        help="Generate dataset for shape recognition (circles and stars in yellow)",
    )
    task_group.add_argument(
        "--color_recognition",
        action="store_true",
        help="Generate dataset for color recognition (circles in yellow and blue)",
    )
    task_group.add_argument(
        "--custom",
        action="store_true",
        help="Generate custom dataset with specified shapes and colors",
    )

    # Custom options for advanced users
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=["circle", "star", "triangle", "square"],
        help="Shapes to generate (only used with --custom)",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        choices=["yellow", "blue", "red", "green"],
        help="Colors to use (only used with --custom)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate shapes for training visual models."
    )

    # Main task selection
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--shapes", action="store_true", help="Generate circles and stars in yellow"
    )
    task_group.add_argument(
        "--colors", action="store_true", help="Generate circles in yellow and blue"
    )
    task_group.add_argument(
        "--custom",
        action="store_true",
        help="Generate custom combination of shapes and colors",
    )

    # Custom options
    parser.add_argument(
        "--custom_shapes",
        nargs="+",
        choices=["circle", "star", "triangle", "square"],
        help="Shapes to generate (only used with --custom)",
    )
    parser.add_argument(
        "--custom_colors",
        nargs="+",
        choices=["yellow", "blue", "red", "green"],
        help="Colors to use (only used with --custom)",
    )

    args = parser.parse_args()

    if args.shapes:
        img_gen = ShapesGenerator(
            shapes=["circle", "star"], colors=["yellow"], task_type="two_shapes"
        )
    elif args.colors:
        img_gen = ShapesGenerator(
            shapes=["circle"], colors=["yellow", "blue"], task_type="two_colors"
        )
    else:  # custom
        if not args.custom_shapes or not args.custom_colors:
            raise ValueError(
                "Must specify both --custom_shapes and --custom_colors with --custom"
            )
        img_gen = ShapesGenerator(
            shapes=args.custom_shapes, colors=args.custom_colors, task_type="custom"
        )

    img_gen.generate_images()
