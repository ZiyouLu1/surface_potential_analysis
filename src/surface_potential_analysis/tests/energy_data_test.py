import random
import unittest

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.energy_data import (
    EnergyGrid,
    extend_z_data,
    get_energy_grid_coordinates,
    interpolate_points_fourier,
    repeat_original_data,
)


class TestEnergyData(unittest.TestCase):
    def test_repeat_original_data_shape(self) -> None:

        data: EnergyGrid = {
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "delta_x1": (0, 2 * np.pi * hbar),
            "delta_x2": (2 * np.pi * hbar, 0),
            "z_points": [0, 1],
        }

        extended = repeat_original_data(data)
        self.assertTrue(np.array_equal(np.array(extended["points"]).shape, (6, 6, 2)))

    def test_repeat_original_data_spacing(self) -> None:
        delta_xy = 2 * np.pi * hbar
        n_xy = random.randrange(1, 10) * 2 + 1

        data: EnergyGrid = {
            "points": np.zeros(shape=(n_xy - 1, n_xy - 1, 2)).tolist(),
            "delta_x1": (0, delta_xy),
            "delta_x2": (delta_xy, 0),
            "z_points": [0, 1],
        }

        extended = repeat_original_data(data)

        self.assertEqual(0, extended["delta_x1"][0])
        self.assertEqual(3 * delta_xy, extended["delta_x1"][1])
        self.assertEqual(3 * delta_xy, extended["delta_x2"][0])
        self.assertEqual(0, extended["delta_x2"][1])

    def test_extend_z_data(self) -> None:
        points = [[[1.0, 2.0, 3.0]]]

        data: EnergyGrid = {
            "points": points,
            "delta_x1": (1, 0),
            "delta_x2": (0, 1),
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
            "delta_x1": (1, 0),
            "delta_x2": (0, 1),
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

    def test_interpolate_points_fourier_double(self) -> None:

        original_shape = tuple(np.random.randint(1, 10, size=2))
        points = np.random.random(size=original_shape).tolist()
        interpolated_shape = (original_shape[0] * 2, original_shape[1] * 2)
        expected = points
        actual = np.array(interpolate_points_fourier(points, interpolated_shape))
        np.testing.assert_array_almost_equal(expected, actual[::2, ::2])

    def test_interpolate_points_fourier_flat(self) -> None:

        value = np.random.rand()
        shape_in = tuple(np.random.randint(1, 10, size=2))
        points = (value * np.ones(shape_in)).tolist()
        shape_out = tuple(np.random.randint(1, 10, size=2))

        expected = value * np.ones(shape_out)
        actual = interpolate_points_fourier(points, shape_out)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_interpolate_points_fourier_original(self) -> None:

        shape = tuple(np.random.randint(1, 10, size=2))
        points = np.random.random(size=shape).tolist()

        expected = points
        actual = interpolate_points_fourier(points, shape)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_fourier_transform_of_interpolation(self) -> None:

        int_shape = tuple(2 * np.random.randint(2, 10, size=2))
        points = np.random.random(size=(2, 2)).tolist()

        expected = np.zeros(int_shape, dtype=complex)
        expected[0:2, 0:2] = np.real_if_close(np.fft.ifft2(points, axes=(0, 1)))
        interpolation = interpolate_points_fourier(points, int_shape)
        actual = np.real_if_close(np.fft.ifft2(interpolation, axes=(0, 1)))

        np.testing.assert_array_almost_equal(expected, actual)
