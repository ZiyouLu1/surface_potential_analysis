import unittest

import numpy as np


class InterpolationTest(unittest.TestCase):
    def test_interpolate_periodic_on_diagonal_grid(self) -> None:
        """Test that interpolating a periodic potential on a diagonal grid is the same as that on a square grid"""

        points_grid_unit_cell = np.random.random((2, 2, 3))
