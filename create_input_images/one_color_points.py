#!/usr/bin/env python3

import os
import argparse
import random
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from points import NumberPoints, PointLayoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TerminalPointLayoutError(ValueError):
    """Raised when point layout attempts are exhausted."""

    pass


class OneColourImageGenerator:
    """Generates images with configurable colored points."""

    def __init__(self, config):
        self.config = config
        self.nmin = self.config["min_point_num"]
        self.nmax = self.config["max_point_num"]
        self.total_area = self.config["total_area"]
        self._check_areas_make_sense()
        self.setup_directories()

        if self.nmin == 0:
            raise ValueError("min_point_num must be at least 1")

    def _check_areas_make_sense(self):
        """Validate that the requested total area is feasible."""
        if self.total_area is not None:
            min_area_max_num = np.pi * self.config["min_point_radius"] ** 2 * self.nmax
            max_area_max_num = np.pi * self.config["max_point_radius"] ** 2 * self.nmax

            if self.total_area < min_area_max_num:
                raise ValueError(
                    f"total_area is too small. It must be at least {min_area_max_num}"
                )
            if self.total_area > max_area_max_num:
                raise ValueError(
                    f"Total_area is very large, please make total area smaller than {max_area_max_num}"
                )

    def setup_directories(self):
        """Create necessary directories for saving images."""
        os.makedirs(self.config["IMG_DIR"], exist_ok=True)
        for c in range(self.nmin, self.nmax + 1):
            os.makedirs(os.path.join(self.config["IMG_DIR"], str(c)), exist_ok=True)

    def create_image(self, n):
        """Create a single image with n points."""
        img = Image.new(
            self.config["mode"],
            (self.config["init_size"], self.config["init_size"]),
            color=self.config["background_colour"],
        )

        number_points = NumberPoints(
            img,
            self.config["init_size"],
            yellow=self.config["yellow"],
            blue=None,
            min_point_radius=self.config["min_point_radius"],
            max_point_radius=self.config["max_point_radius"],
            attempts_limit=self.config["attempts_limit"],
        )

        point_array = number_points.design_n_points(n, self.config["colour"])

        if self.total_area is not None:
            point_array = number_points.fix_total_area(point_array, self.total_area)

        return number_points.draw_points(point_array)

    def create_and_save(self, n, tag=""):
        """Create and save an image, with retry logic."""
        v_tag = f"_{self.config['version_tag']}" if self.config["version_tag"] else ""
        ac_tag = "_ac" if self.total_area is not None else ""
        name = f"img_{n}_{tag}{ac_tag}{v_tag}.png"

        attempts = 0
        while attempts < self.config["attempts_limit"]:
            try:
                self.create_and_save_once(name, n)
                break
            except PointLayoutError as e:
                logging.debug(f"Failed to create image {name} because '{e}' Retrying.")
                attempts += 1

                if attempts == self.config["attempts_limit"]:
                    raise TerminalPointLayoutError(
                        f"Failed to create image {name} after {attempts} attempts. "
                        "Your points are probably too big, or there are too many. Stopping."
                    )

    def create_and_save_once(self, name, n):
        """Create and save a single image without retry logic."""
        img = self.create_image(n)
        img.save(os.path.join(self.config["IMG_DIR"], str(n), name))

    def generate_images(self):
        """Generate the full set of images based on configuration."""
        total_images = self.config["IMAGE_SET_NUM"] * (self.nmax - self.nmin + 1)
        logging.info(f"Generating {total_images} images...")

        for i in tqdm(range(self.config["IMAGE_SET_NUM"])):
            for n in range(self.nmin, self.nmax + 1):
                self.create_and_save(n, tag=i)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images with configurable colored points."
    )
    parser.add_argument(
        "--img_set_num", type=int, default=100, help="Number of image sets to generate"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="images/extremely_easy",
        help="Directory to save images",
    )
    parser.add_argument(
        "--total_area",
        type=int,
        default=None,
        help="Total area of the points in the image (optional)",
    )
    parser.add_argument(
        "--seed", type=int, default=1714, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--version_tag",
        type=str,
        default="",
        help="Version tag to append to image names",
    )
    parser.add_argument(
        "--min_points", type=int, default=1, help="Minimum number of points per image"
    )
    parser.add_argument(
        "--max_points", type=int, default=5, help="Maximum number of points per image"
    )

    return parser.parse_args()


def main():
    """Main entry point of the script."""
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Base configuration
    config = {
        "version_tag": args.version_tag,
        "colour": "yellow",
        "boundary_width": 5,
        "background_colour": "#000000",
        "yellow": "#fffe04",
        "min_point_radius": 20,
        "max_point_radius": 30,
        "init_size": 512,
        "mode": "RGB",
        "min_point_num": args.min_points,
        "max_point_num": args.max_points,
        "attempts_limit": 5000,
        "IMAGE_SET_NUM": args.img_set_num,
        "IMG_DIR": args.img_dir,
        "total_area": args.total_area,
    }

    try:
        image_generator = OneColourImageGenerator(config)
        image_generator.generate_images()
        logging.info("Image generation completed successfully!")
    except Exception as e:
        logging.error(f"Error during image generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
