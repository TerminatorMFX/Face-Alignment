from unittest import TestCase

import numpy as np

from EyeMarking import angle_of_points, calculate_scaling_factor, find_left_eye


class AngleConversionTestCase(TestCase):

    def test_zero_degree_inclination(self):
        angle = angle_of_points((0, 0), (1, 0))
        self.assertEqual(0, angle)

    def test_45_degree_inclination(self):
        angle = angle_of_points((0, 0), (1, 1))
        self.assertEqual(np.pi / 4, angle)

    def test_45_degree_negative_inclination(self):
        # TODO: Could be a potential bug, we have a negative inclination but a positive return.
        angle = angle_of_points((0, 0), (-1, -1))
        self.assertEqual(np.pi / 4, angle)

    def test_scaling(self):
        mock_img = np.zeros((3088, 2320, 3))
        x_scale = 308.8
        y_scale = 232
        x_scale_factor, y_scale_factor = calculate_scaling_factor(mock_img, x_scale, y_scale)

        self.assertEqual(x_scale_factor, 10)
        self.assertEqual(y_scale_factor, 10)

    def test_find_left_eye(self):
        eye_coordinates = [(1, 100), (0, 0)]
        left_eye_coordinate = find_left_eye(eye_coordinates)
        self.assertEqual(left_eye_coordinate, (0, 0))
