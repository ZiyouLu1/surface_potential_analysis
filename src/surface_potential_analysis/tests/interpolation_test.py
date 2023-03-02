import unittest
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from surface_potential_analysis.interpolation import (
    interpolate_points_fourier_complex,
    interpolate_real_points_along_axis_fourier,
)


def interpolate_points_fourier(points: List[List[float]], shape: Tuple[int, int]):
    x_interp = interpolate_real_points_along_axis_fourier(points, shape[0], axis=0)
    y_interp = interpolate_real_points_along_axis_fourier(x_interp, shape[1], axis=1)
    return y_interp


class InterpolationTest(unittest.TestCase):
    def test_interpolate_periodic_on_diagonal_grid(self) -> None:
        """Test that interpolating a periodic potential on a diagonal grid is the same as that on a square grid"""

        points_grid_unit_cell = np.random.random((2, 2, 3))

    def test_interpolation_cosine(self) -> None:
        def test_fn(x: NDArray):
            return x**2 * (x - 1) ** 2

        points_4 = np.linspace(0, 1, num=4, endpoint=False)
        points_5 = np.linspace(0, 1, num=5, endpoint=False)

        print(points_4, test_fn(points_4))
        print(np.fft.rfft(test_fn(points_4), norm="backward"))
        print(np.fft.rfft(test_fn(points_4)).shape)
        print(points_5, test_fn(points_5))
        print(np.fft.rfft(test_fn(points_5), norm="backward"))
        print(np.fft.rfft(test_fn(points_5)).shape)

        print(interpolate_real_points_along_axis_fourier(points_4, 5))

    def test_interpolate_points_fourier_double(self) -> None:

        original_shape = tuple(np.random.randint(1, 2, size=2))
        points = np.random.random(size=original_shape).tolist()
        interpolated_shape = (original_shape[0] * 2, original_shape[1] * 2)
        expected = points

        actual = np.array(interpolate_points_fourier(points, interpolated_shape))
        np.testing.assert_array_almost_equal(expected, actual[::2, ::2])

        actual = np.array(
            interpolate_points_fourier_complex(points, interpolated_shape)
        )
        np.testing.assert_array_almost_equal(expected, actual[::2, ::2])

    def test_interpolate_points_fourier_flat(self) -> None:

        value = np.random.rand()
        shape_in = tuple(np.random.randint(1, 10, size=2))
        points = (value * np.ones(shape_in)).tolist()
        shape_out = tuple(np.random.randint(1, 10, size=2))

        expected = value * np.ones(shape_out)
        actual = interpolate_points_fourier(points, (shape_out[0], shape_out[1]))
        np.testing.assert_array_almost_equal(expected, actual)

        actual = interpolate_points_fourier_complex(
            points, (shape_out[0], shape_out[1])
        )
        np.testing.assert_array_almost_equal(expected, actual)

    def test_interpolate_points_fourier_original(self) -> None:

        shape = tuple(np.random.randint(1, 10, size=2))
        points = np.random.random(size=shape).tolist()

        expected = points
        actual = interpolate_points_fourier(points, shape=(shape[0], shape[1]))
        np.testing.assert_array_almost_equal(expected, actual)

        actual = interpolate_points_fourier_complex(points, (shape[0], shape[1]))
        np.testing.assert_array_almost_equal(expected, actual)

    def test_fourier_transform_of_interpolation(self) -> None:

        int_shape = (np.random.randint(2, 10), np.random.randint(2, 10))
        points = np.random.random(size=(2, 2)).tolist()

        expected = np.zeros(int_shape, dtype=complex)
        original_ft = np.fft.ifft2(points, axes=(0, 1))

        # Note: 0:2 range, as we make use of the hermitian property
        # of the fourier transform of a real array
        expected[0:2, 0:2] = original_ft[0:2, 0:2]
        expected[0:2, -1:] = original_ft[0:2, -1:]
        expected[-1:, 0:2] = original_ft[-1:, 0:2]
        expected[-1:, -1:] = original_ft[-1:, -1:]

        interpolation = interpolate_points_fourier(points, int_shape)
        actual = np.fft.ifft2(interpolation, axes=(0, 1))

        np.testing.assert_array_almost_equal(expected, actual)
