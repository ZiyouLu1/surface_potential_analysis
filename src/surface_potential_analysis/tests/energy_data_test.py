import random
import unittest

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.energy_data import (
    EnergyGrid,
    extend_z_data,
    get_energy_grid_coordinates,
    repeat_original_data,
)


class TestEnergyData(unittest.TestCase):
    def test_repeat_original_data_shape(self) -> None:

        data: EnergyGrid = {
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "delta_x0": (0, 2 * np.pi * hbar),
            "delta_x1": (2 * np.pi * hbar, 0),
            "z_points": [0, 1],
        }

        extended = repeat_original_data(data)
        self.assertTrue(np.array_equal(np.shape(extended["points"]), (6, 6, 2)))

    def test_repeat_original_data_spacing(self) -> None:
        delta_xy = 2 * np.pi * hbar
        n_xy = random.randrange(1, 10) * 2 + 1

        data: EnergyGrid = {
            "points": np.zeros(shape=(n_xy - 1, n_xy - 1, 2)).tolist(),
            "delta_x0": (0, delta_xy),
            "delta_x1": (delta_xy, 0),
            "z_points": [0, 1],
        }

        extended = repeat_original_data(data)

        self.assertEqual(0, extended["delta_x0"][0])
        self.assertEqual(3 * delta_xy, extended["delta_x0"][1])
        self.assertEqual(3 * delta_xy, extended["delta_x1"][0])
        self.assertEqual(0, extended["delta_x1"][1])

    def test_extend_z_data(self) -> None:
        points = [[[1.0, 2.0, 3.0]]]

        data: EnergyGrid = {
            "points": points,
            "delta_x0": (1, 0),
            "delta_x1": (0, 1),
            "z_points": [0, 1, 4],
        }

        extended = extend_z_data(data)
        np.testing.assert_array_equal(extended["points"], [[[1, 1, 1, 2, 3, 3, 3]]])
        np.testing.assert_array_equal(
            extended["z_points"][0:5], [-2.0, -1.0, 0.0, 1.0, 4.0]
        )

    def test_get_energy_grid_coordinates(self) -> None:
        data: EnergyGrid = {
            "points": np.zeros((2, 2, 4)).tolist(),
            "delta_x0": (1, 0),
            "delta_x1": (0, 1),
            "z_points": [0, 1, 2, 3],
        }

        actual = get_energy_grid_coordinates(data)
        expected = [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
                [[0.0, 0.5, 0.0], [0.0, 0.5, 1.0], [0.0, 0.5, 2.0], [0.0, 0.5, 3.0]],
            ],
            [
                [[0.5, 0.0, 0.0], [0.5, 0.0, 1.0], [0.5, 0.0, 2.0], [0.5, 0.0, 3.0]],
                [[0.5, 0.5, 0.0], [0.5, 0.5, 1.0], [0.5, 0.5, 2.0], [0.5, 0.5, 3.0]],
            ],
        ]
        np.testing.assert_array_equal(expected, actual)
