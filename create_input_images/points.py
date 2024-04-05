from random import randint
from PIL import ImageDraw
import numpy as np


class NumberPoints:

    boundary_width = 5
    background_colour = "#ebecf0"
    point_sep = 10
    min_point_radius = 8
    max_point_radius = 16
    # We consider equal areas if they differ by less than this number:
    area_tolerance = 0.1

    def __init__(self, img, init_size, yellow, blue):
        self.img = img
        self.init_size = init_size
        self.draw = ImageDraw.Draw(img)
        self.yellow = yellow
        self.blue = blue
        # TODO: this is deprecated
        np.random.seed(1234)

    def _create_random_point(self):
        radius = randint(self.min_point_radius, self.max_point_radius + 1)
        limit = self.boundary_width + radius + self.point_sep

        # Change mental coordinate system to the center of the square
        rad = int(self.init_size / 2 - limit)
        rx = randint(-rad, rad)
        # Make sure we are always inside the circle
        max_ = int(np.sqrt((self.init_size / 2 - limit) ** 2 - rx ** 2))
        ry = randint(-max_, max_)

        # Transform to coordinates with origin on the upper left quadrant
        x = rx + self.init_size / 2
        y = ry + self.init_size / 2

        return x, y, radius

    def _check_no_overlaps(self, point_array, new_point):
        return all([self._check_overlapping_points(a[0], new_point) for a in point_array])

    def _check_overlapping_points(self, point, new_point):
        dist = np.sqrt((point[0] - new_point[0]) ** 2 + (point[1] - new_point[1]) ** 2)

        return dist > point[2] + new_point[2] + self.point_sep

    def design_n_points(self, n, colour, point_array=None):

        if point_array is None:
            point_array = []

        for i in range(n):
            new_point = self._create_random_point()

            while not self._check_no_overlaps(point_array, new_point):
                new_point = self._create_random_point()
            point_array.append((new_point, colour))

        return point_array

    def _draw_point(self, point, colour):
        # pos: position of the center: a tuple of x and y in pixels
        # size: radius of the circle (in pixels)
        x1 = point[0] - point[2]
        x2 = point[0] + point[2]
        y1 = point[1] - point[2]
        y2 = point[1] + point[2]
        self.draw.ellipse((x1, y1, x2, y2), fill= self.yellow if colour == "yellow" else self.blue)

    def draw_points(self, point_array):
        [self._draw_point(a[0], a[1]) for a in point_array]
        return self.img

    @staticmethod
    def compute_area(point_array, colour):
        return sum([np.pi * a[0][2] ** 2 for a in point_array if a[1] == colour])

    def _check_areas_equal(self, big, small):
        return (big - small) / big < self.area_tolerance

    def _get_areas(self, point_array):
        yellow_area = self.compute_area(point_array, "yellow")
        blue_area = self.compute_area(point_array, "blue")

        # Who is big and who is small
        small = "blue" if yellow_area > blue_area else "yellow"
        big_area, small_area = (yellow_area, blue_area) if small == "blue" else (blue_area, yellow_area)

        return small, big_area, small_area

    @staticmethod
    def _increase_radius(point):
        return (point[0][0], point[0][1], point[0][2] + 1), point[1]

    def equalize_areas(self, point_array):

        # Who is big and who is small
        small, big_area, small_area = self._get_areas(point_array)

        # Make all points in small area bigger to match bigger area
        # This brings us to this problem: solve a = sum_i^n (x_i^2 + 2r_i*x_i),
        # which is not solvable analytically. Therefore, what we'll do is add
        # pixel after pixel to all points until we are close to the target value
        while not self._check_areas_equal(big_area, small_area):
            point_array = [self._increase_radius(a) if a[1] == small else a for a in point_array]
            # Recompute
            small, big_area, small_area = self._get_areas(point_array)

        return point_array
