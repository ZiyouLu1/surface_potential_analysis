import unittest

import numpy as np

from surface_potential_analysis.brillouin_zone import get_points_in_brillouin_zone


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
