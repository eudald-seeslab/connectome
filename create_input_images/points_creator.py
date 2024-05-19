import os
import argparse
from PIL import Image
from points import NumberPoints, PointLayoutError
import random
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


GENERAL_CONFIG = {
    # so that we can create more images without the names clashing with the previous
    "version_tag": "",
    "colour_1": "yellow",
    "colour_2": "blue",
    "boundary_width": 5,
    "background_colour": "#000000", # for gray: #808080
    "yellow": "#fffe04",
    "blue": "#0003f9",
    "point_sep": 20,
    "min_point_radius": 40,
    "max_point_radius": 80,
    "init_size": 512,
    "mode": "RGB",
    # these are per colour
    "min_point_num": 5,
    "max_point_num": 15,
    "attempts_limit": 200,
}


EASY_RATIOS = [1 / 3, 2 / 5, 1 / 2, 2 / 3, 3 / 4]
HARD_RATIOS = [
    4 / 5,
    5 / 6,
    7 / 8,
    8 / 9,
    9 / 10,
    10 / 11,
    11 / 12,
]


class TerminalPointLayoutError(ValueError):
    pass


class ImageGenerator:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.ratios = EASY_RATIOS if self.config["EASY"] else EASY_RATIOS + HARD_RATIOS

    def setup_directories(self):
        os.makedirs(self.config["IMG_DIR"], exist_ok=True)
        os.makedirs(
            os.path.join(self.config["IMG_DIR"], self.config["colour_1"]), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.config["IMG_DIR"], self.config["colour_2"]), exist_ok=True
        )

    def create_image(self, n1, n2, equalized):
        img = Image.new(
            self.config["mode"],
            (self.config["init_size"], self.config["init_size"]),
            color=self.config["background_colour"],
        )
        number_points = NumberPoints(
            img,
            self.config["init_size"],
            yellow=self.config["yellow"],
            blue=self.config["blue"],
            min_point_radius=self.config["min_point_radius"],
            max_point_radius=self.config["max_point_radius"],
            attempts_limit=self.config["attempts_limit"],
        )
        point_array = number_points.design_n_points(n1, self.config["colour_1"])
        point_array = number_points.design_n_points(
            n2, self.config["colour_2"], point_array=point_array
        )
        if equalized:
            point_array = number_points.equalize_areas(point_array)
        return number_points.draw_points(point_array)

    def create_and_save(self, n1, n2, equalized, tag=""):
        eq = "_equalized" if equalized else ""
        v_tag = f"_{config['version_tag']}" if config["version_tag"] is not None else ""
        name = f"img_{n1}_{n2}_{tag}{eq}{v_tag}.png"

        attempts = 0
        while attempts < self.config["attempts_limit"]:
            try:
                self.create_and_save_once(name, n1, n2, equalized)
                break
            except PointLayoutError as e:
                logging.debug(f"Failed to create image {name} because '{e}' Retrying.")
                attempts += 1

                if attempts == self.config["attempts_limit"]:
                    raise TerminalPointLayoutError(
                        f"""Failed to create image {name} after {attempts} attempts. 
                        Your points are probably too big, or there are too many. 
                        Stopping."""
                    )

    def create_and_save_once(self, name, n1, n2, equalized):
        img = self.create_image(n1, n2, equalized)
        img.save(
        os.path.join(
            self.config["IMG_DIR"],
            self.config["colour_1"] if n1 > n2 else self.config["colour_2"],
            name,
            )
        )

    def get_positions(self):
        min_p = self.config["min_point_num"]
        max_p = self.config["max_point_num"]

        positions = []
        # Note that we don't need the last value of 'a', since 'b' will always be greater.
        for a in range(min_p, max_p):
            # Given 'a', we need to find 'b' in the tuple (a, b) such that b/a is in the ratios list.
            for ratio in self.ratios:
                b = a / ratio

                # We keep this tuple if b is an integer and within the allowed range.
                if b == round(b) and b <= max_p:
                    positions.append((a, int(b)))

        return positions

    def generate_images(self):
        positions = self.get_positions()
        logging.info(
            f"This will make {self.config['IMAGE_SET_NUM'] * len(positions) * 4} images."
        )
        for i in tqdm(range(self.config["IMAGE_SET_NUM"])):
            for pair in positions:
                self.create_and_save(pair[0], pair[1], equalized=True, tag=i)
                self.create_and_save(pair[1], pair[0], equalized=True, tag=i)
                self.create_and_save(pair[0], pair[1], equalized=False, tag=i)
                self.create_and_save(pair[1], pair[0], equalized=False, tag=i)


def get_config():
    parser = argparse.ArgumentParser(
        description="Generate images based on number point configurations."
    )
    parser.add_argument(
        "--img_set_num",
        type=int,
        default=100,
        help="Number of image sets to generate.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="images/extremely_easy",
        help="Directory to save images.",
    )
    parser.add_argument(
        "--easy", action='store_true', help="Use easier ratios between colours."
    )
    args = parser.parse_args()

    config = GENERAL_CONFIG | {
        "EASY": args.easy,
        "IMAGE_SET_NUM": args.img_set_num,
        "IMG_DIR": args.img_dir,
    }
    return config


if __name__ == "__main__":
    config = get_config()
    random.seed(1714)
    image_generator = ImageGenerator(config)
    image_generator.generate_images()
