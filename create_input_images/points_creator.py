import os
import argparse
from PIL import Image
from points import NumberPoints
import random
from tqdm import tqdm


EASY_RATIOS = [2 / 5, 1 / 2, 2 / 3, 3 / 4]
HARD_RATIOS = [
    4 / 5,
    5 / 6,
    7 / 8,
    8 / 9,
    9 / 10,
    10 / 11,
    11 / 12,
]


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
        )
        point_array = number_points.design_n_points(n1, self.config["colour_1"])
        point_array = number_points.design_n_points(
            n2, self.config["colour_2"], point_array=point_array
        )
        if equalized:
            point_array = number_points.equalize_areas(point_array)
        return number_points.draw_points(point_array)

    def create_and_save(self, n1, n2, equalized, tag=""):
        name = f"img_{n1}_{n2}_{tag}{'_equalized' if equalized else ''}v3.png"
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
        p = [
            [(b, int(b / a)) for a in self.ratios if max_p >= b / a == round(b / a)]
            for b in range(min_p, max_p + 1)
        ]
        return [item for sublist in p for item in sublist]

    def generate_images(self):
        positions = self.get_positions()
        print(
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
        "--image_set_num",
        type=int,
        default=10,
        help="Number of image sets to generate.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="images/easy_v2",
        help="Directory to save images.",
    )
    parser.add_argument(
        "--easy", type=bool, default=True, help="Use easy mode for image generation."
    )
    args = parser.parse_args()

    config = {
        "EASY": args.easy,
        "IMAGE_SET_NUM": args.image_set_num,
        "IMG_DIR": args.img_dir,
        "colour_1": "yellow",
        "colour_2": "blue",
        "boundary_width": 5,
        "background_colour": "#808080",
        "yellow": "#fffe04",
        "blue": "#0003f9",
        "point_sep": 20,
        "min_point_radius": 8,
        "max_point_radius": 16,
        "init_size": 512,
        "mode": "RGB",
        "min_point_num": 4,
        "max_point_num": 16,
    }
    return config


if __name__ == "__main__":
    config = get_config()
    random.seed(1714)
    image_generator = ImageGenerator(config)
    image_generator.generate_images()
