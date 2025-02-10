#!/usr/bin/env python3

import os
import argparse
import logging
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class StripePatternGenerator:
    """Generates images with rotated stripe patterns."""

    def __init__(self, config):
        self.config = config
        self.min_thickness = config["min_thickness"]
        self.max_thickness = config["max_thickness"]
        self.min_spacing = config["min_spacing"]
        self.min_stripe_num = config["min_stripe_num"]
        self.max_stripe_num = config["max_stripe_num"]
        self.size = config["img_size"]
        self.dir_path = config["output_dir"]
        self.angles = config["angles"]
        self.max_attempts = config["max_attempts"]
        self.img_sets = config["img_sets"]
        self.tag = config["tag"]

        # Calculate circumscribed size for rotation
        self.c_size = int(self.size / 2 * np.sqrt(2)) * 2

    def create_images(self):
        """Generate the complete set of images with different angles and stripe counts."""
        self._create_directories()

        total_images = (
            self.img_sets
            * len(self.angles)
            * (self.max_stripe_num - self.min_stripe_num + 1)
        )
        logging.info(f"Generating {total_images} images...")

        for i in tqdm(range(self.img_sets)):
            for angle in self.angles:
                for num_stripes in range(self.min_stripe_num, self.max_stripe_num + 1):
                    try:
                        img = self.create_rotated_stripes(num_stripes, angle)
                        tag_suffix = f"_{self.tag}" if self.tag else ""
                        filename = f"img_{num_stripes}_{i}{tag_suffix}.png"
                        img.save(os.path.join(self.dir_path, str(angle), filename))
                    except Exception as e:
                        logging.error(
                            f"Failed to generate image: angle={angle}, stripes={num_stripes}, set={i}"
                        )
                        logging.error(str(e))
                        raise

    def create_rotated_stripes(self, num_stripes, angle):
        """Create an image with the specified number of stripes at the given angle."""
        img = Image.new(
            "RGB", (self.c_size, self.c_size), color=self.config["background_color"]
        )
        draw = ImageDraw.Draw(img)

        # Generate random stripe thicknesses
        stripe_thickness = np.random.randint(
            self.min_thickness, self.max_thickness, num_stripes
        )

        # Calculate valid range for stripe positions
        min_start_point = (self.c_size - self.size) // 2 * np.cos(angle * np.pi / 180)
        max_start_point = (
            self.c_size - min_start_point - self.min_thickness - self.min_spacing
        )

        # Generate non-overlapping stripe positions
        starting_positions = self._generate_valid_positions(
            num_stripes, min_start_point, max_start_point, stripe_thickness
        )

        # Draw the stripes
        for i in range(num_stripes):
            upper_left = (starting_positions[i], 0)
            lower_right = (
                starting_positions[i] + stripe_thickness[i],
                self.c_size,
            )
            draw.rectangle([upper_left, lower_right], fill="white")

        # Rotate and crop
        rotated_img = img.rotate(angle)
        crop_box = (
            (self.c_size - self.size) // 2,
            (self.c_size - self.size) // 2,
            (self.c_size + self.size) // 2,
            (self.c_size + self.size) // 2,
        )
        return rotated_img.crop(crop_box)

    def _generate_valid_positions(self, num_stripes, min_start, max_start, thicknesses):
        """Generate non-overlapping positions for stripes."""
        attempts = 0
        while attempts < self.max_attempts:
            positions = np.random.randint(min_start, max_start, num_stripes)
            if not self._check_overlaps(positions, thicknesses):
                return positions
            attempts += 1

        raise ValueError(
            f"Failed to generate non-overlapping positions after {self.max_attempts} attempts"
        )

    def _check_overlaps(self, starting_positions, stripe_thickness):
        """Check if any stripes overlap."""
        for i in range(len(starting_positions)):
            for j in range(i + 1, len(starting_positions)):
                if (
                    starting_positions[i]
                    < starting_positions[j] + stripe_thickness[j] + self.min_spacing
                    and starting_positions[i] + stripe_thickness[i] + self.min_spacing
                    > starting_positions[j]
                ):
                    return True
        return False

    def _create_directories(self):
        """Create output directories for each angle."""
        os.makedirs(self.dir_path, exist_ok=True)
        for angle in self.angles:
            os.makedirs(os.path.join(self.dir_path, str(angle)), exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images with rotated stripe patterns."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../images/head_rotation_one_stripe",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--img-sets", type=int, default=50, help="Number of image sets to generate"
    )
    parser.add_argument(
        "--angles",
        type=int,
        nargs="+",
        default=[0, 45, 90, 135],
        help="List of rotation angles",
    )
    parser.add_argument(
        "--min-stripes", type=int, default=2, help="Minimum number of stripes per image"
    )
    parser.add_argument(
        "--max-stripes",
        type=int,
        default=10,
        help="Maximum number of stripes per image",
    )
    parser.add_argument(
        "--img-size", type=int, default=512, help="Size of the output images"
    )
    parser.add_argument(
        "--tag", type=str, default="", help="Optional tag to add to image filenames"
    )
    parser.add_argument(
        "--min-thickness", type=int, default=10, help="Minimum stripe thickness"
    )
    parser.add_argument(
        "--max-thickness", type=int, default=30, help="Maximum stripe thickness"
    )
    parser.add_argument(
        "--min-spacing", type=int, default=5, help="Minimum spacing between stripes"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10000,
        help="Maximum attempts to generate non-overlapping stripes",
    )

    return parser.parse_args()


def main():
    """Main entry point of the script."""
    args = parse_args()

    config = {
        "output_dir": args.output_dir,
        "img_sets": args.img_sets,
        "angles": args.angles,
        "min_stripe_num": args.min_stripes,
        "max_stripe_num": args.max_stripes,
        "img_size": args.img_size,
        "tag": args.tag,
        "min_thickness": args.min_thickness,
        "max_thickness": args.max_thickness,
        "min_spacing": args.min_spacing,
        "max_attempts": args.max_attempts,
        "background_color": "#000000",
    }

    try:
        generator = StripePatternGenerator(config)
        generator.create_images()
        logging.info("Image generation completed successfully!")
    except Exception as e:
        logging.error(f"Error during image generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
