from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from surface_potential_analysis.util.interpolation import (
    interpolate_points_fftn,
    interpolate_points_rfft,
    interpolate_points_rfftn,
    pad_ft_points,
    pad_ft_points_real_axis,
)


def _interpolate_real_points_fourier(
    points: np.ndarray[tuple[int, int], Any], shape: tuple[int, int]
) -> np.ndarray[tuple[int, int], Any]:
    x_interpolation = interpolate_points_rfft(points, shape[0], axis=0)
    return interpolate_points_rfft(  # type: ignore[return-value]
        x_interpolation, shape[1], axis=1
    )


def _interpolate_complex_points_fourier(
    points: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    shape: tuple[int, int],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    return interpolate_points_fftn(points, shape, axes=(0, 1))


rng = np.random.default_rng()


class InterpolationTest(unittest.TestCase):
    def test_pad_ft_points_real_axis(self) -> None:
        initial_n = rng.integers(1, 100)  # type: ignore bad libary types
        initial = rng.random((initial_n,))

        n = rng.integers(2 * (initial_n - 1), 2 * (initial_n - 1) + 2)  # type: ignore bad libary types
        final = pad_ft_points_real_axis(initial, n)
        np.testing.assert_array_equal(initial, final)

        n = rng.integers(2 * (initial_n - 1) + 2, 2 * (initial_n - 1) + 20)  # type: ignore bad libary types
        final = pad_ft_points_real_axis(initial, n)
        np.testing.assert_array_equal(initial, final[: initial.shape[0]])

        n = rng.integers(1, 2 * (initial_n - 1))  # type: ignore bad libary types
        final = pad_ft_points_real_axis(initial, n)
        np.testing.assert_array_equal(initial[: final.shape[0]], final)

    def test_interpolate_points_fourier_double(self) -> None:
        # Note the interpolation that assumes the potential is real will not
        # return the same points if the original data has an even number of points
        original_shape = tuple(2 * rng.integers(1, 5, size=2) - 1)  # type: ignore bad libary types
        points = rng.random(size=original_shape).tolist()
        interpolated_shape = (original_shape[0] * 2, original_shape[1] * 2)
        expected = points
        actual = np.array(_interpolate_real_points_fourier(points, interpolated_shape))
        np.testing.assert_array_almost_equal(expected, actual[::2, ::2])

        actual = np.array(
            _interpolate_complex_points_fourier(points, interpolated_shape)
        )
        np.testing.assert_array_almost_equal(expected, actual[::2, ::2])

    def test_interpolate_points_fourier_flat(self) -> None:
        value = rng.random()
        shape_in = tuple(rng.integers(1, 10, size=2))  # type: ignore bad libary types
        points = (value * np.ones(shape_in)).tolist()
        shape_out = tuple(rng.integers(1, 10, size=2))  # type: ignore bad libary types

        expected = value * np.ones(shape_out)
        actual = _interpolate_real_points_fourier(points, (shape_out[0], shape_out[1]))
        np.testing.assert_array_almost_equal(expected, actual)

        actual = _interpolate_complex_points_fourier(
            points, (shape_out[0], shape_out[1])
        )
        np.testing.assert_array_almost_equal(expected, actual)

    def test_interpolate_points_fourier_original(self) -> None:
        shape = tuple(rng.integers(1, 10, size=2))  # type: ignore bad libary types
        points = rng.random(size=shape).tolist()

        expected = points
        actual = _interpolate_real_points_fourier(points, shape=(shape[0], shape[1]))
        np.testing.assert_array_almost_equal(expected, actual)

        actual = _interpolate_complex_points_fourier(points, (shape[0], shape[1]))
        np.testing.assert_array_almost_equal(expected, actual)

    def test_fourier_transform_of_interpolation(self) -> None:
        int_shape = (rng.integers(2, 10), rng.integers(2, 10))  # type: ignore bad libary types
        points = rng.random(size=(2, 2)).tolist()

        expected = np.zeros(int_shape, dtype=complex)
        original_ft = np.fft.ifft2(points, axes=(0, 1))

        # Note: 0:2 range, as we make use of the hermitian property
        # of the fourier transform of a real array
        expected[0:2, 0:2] = original_ft[0:2, 0:2]
        expected[0:2, -1:] = original_ft[0:2, -1:]
        expected[-1:, 0:2] = original_ft[-1:, 0:2]
        expected[-1:, -1:] = original_ft[-1:, -1:]

        interpolation = _interpolate_real_points_fourier(points, int_shape)
        actual = np.fft.ifft2(interpolation, axes=(0, 1))

        np.testing.assert_array_almost_equal(expected, actual)

    def test_interpolate_real_is_real(self) -> None:
        in_shape = (rng.integers(2, 10), rng.integers(2, 10))  # type: ignore bad libary types
        points = rng.random(size=in_shape)
        out_shape = (rng.integers(2, 10), in_shape[1])  # type: ignore bad libary types
        interpolated = interpolate_points_rfftn(points, out_shape, axes=(0, 1))
        np.testing.assert_array_equal(interpolated, np.real(interpolated))

    def test_interpolate_symmetric_is_symmetric(self) -> None:
        in_shape = (rng.integers(2, 10), rng.integers(2, 10))  # type: ignore bad libary types

        points = rng.random(size=in_shape)
        points[1:, :] += points[:0:-1, :]
        points[:, 1:] += points[:, :0:-1]
        np.testing.assert_array_equal(points[1:, :], points[:0:-1, :])
        np.testing.assert_array_equal(points[:, 1:], points[:, :0:-1])

        out_shape = (rng.integers(2, 10), rng.integers(2, 10))  # type: ignore bad libary types
        interpolated = _interpolate_real_points_fourier(points, out_shape)
        np.testing.assert_array_equal(interpolated[1:, :], interpolated[:0:-1, :])

    def test_pad_ft_points(self) -> None:
        array = np.array([1])
        actual = pad_ft_points(array, s=(2,), axes=(0,))
        expected = np.array([1, 0])
        np.testing.assert_array_equal(actual, expected)

        array = np.array([1, 2])
        actual = pad_ft_points(array, s=(3,), axes=(0,))
        expected = np.array([1, 0, 2])
        np.testing.assert_array_equal(actual, expected)

        array = np.array([1, 2, 3])
        actual = pad_ft_points(array, s=(4,), axes=(0,))
        expected = np.array([1, 2, 0, 3])
        np.testing.assert_array_equal(actual, expected)

        array = np.array(
            [
                [[1, 1], [2, 1], [3, 1]],
                [[1, 2], [2, 2], [3, 2]],
                [[1, 3], [2, 3], [3, 3]],
            ]
        )
        actual = pad_ft_points(array, s=(4, 4), axes=(0, 1))
        expected = np.array(
            [
                [[1, 1], [2, 1], [0, 0], [3, 1]],
                [[1, 2], [2, 2], [0, 0], [3, 2]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [[1, 3], [2, 3], [0, 0], [3, 3]],
            ]
        )
        np.testing.assert_array_equal(actual, expected, verbose=True)
