import unittest

import numpy as np

from surface_potential_analysis._legacy.brillouin_zone import (
    get_coordinate_fractions,
    get_point_fractions,
    get_points_in_brillouin_zone,
    grid_space,
)


class TestBrillouinZone(unittest.TestCase):
    def test_get_points_in_brillouin_zone(self) -> None:
        actual = get_points_in_brillouin_zone(
            (1, 0), (0, 1), size=(1, 1), include_zero=False
        )
        expected = [[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]]
        np.testing.assert_array_equal(actual, expected)

        actual = get_points_in_brillouin_zone(
            (1, 0), (0, 1), size=(1, 1), include_zero=True
        )
        expected = [[-0.5, -0.5], [-0.5, 0.0], [0.0, -0.5], [0.0, 0.0]]
        np.testing.assert_array_equal(actual, expected)

        actual = get_points_in_brillouin_zone(
            (1, 0), (1, 1), size=(1, 1), include_zero=True
        )
        expected = [[-1.0, -0.5], [-0.5, 0.0], [-0.5, -0.5], [0.0, 0.0]]
        np.testing.assert_array_equal(actual, expected)

    def test_get_coordinate_fractions(self) -> None:
        fractions = get_point_fractions(shape=(8, 8), endpoint=False)
        delta_x0: tuple[float, float] = (np.random.rand(), np.random.rand())
        delta_x1: tuple[float, float] = (np.random.rand(), np.random.rand())
        coordinates = grid_space(delta_x0, delta_x1, shape=(8, 8), endpoint=False)

        calculated_fractions = get_coordinate_fractions(delta_x0, delta_x1, coordinates)
        np.testing.assert_allclose(fractions, calculated_fractions, atol=5e-16)
