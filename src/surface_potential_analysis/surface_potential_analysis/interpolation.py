from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def interpolate_real_points_along_axis_fourier(
    points: ArrayLike, n: int, axis=-1
) -> NDArray:
    """
    Given a uniformly spaced (real) grid of points interpolate along the given
    axis to a new length n.

    This makes use of the fact that the potential is real, and therefore if the
    input is even along the interpolation axis we get an additional ft point for 'free'
    using the hermitian property of the fourier transform

    Parameters
    ----------
    points : ArrayLike
        The initial set of points to interpolate
    n : int
        The number of points in the interpolated grid
    axis : int, optional
        The axis over which to interpolate, by default -1

    Returns
    -------
    NDArray
        The points interpolated along the given axis
    """
    # We use the forward norm here, as otherwise we would also need to
    # scale the ft_potential by a factor of n / shape[axis]
    # when we pad or truncate it
    ft_potential = np.fft.rfft(points, axis=axis, norm="forward")
    # Invert the rfft, padding (or truncating) for the new length n
    interpolated_potential = np.fft.irfft(ft_potential, n, axis=axis, norm="forward")

    return interpolated_potential


def interpolate_points_fourier_complex(
    points: List[List[complex]], shape: Tuple[int, int]
) -> List[List[complex]]:
    """
    Given a uniform grid of points in the unit cell interpolate
    a grid of points with the given shape using the fourier transform

    We don't make use of the fact that the potential is real in this case,
    and as such the output may not be real
    """
    ft_potential = np.fft.fft2(points, norm="forward")
    original_shape = ft_potential.shape
    sample_shape = (
        np.min([original_shape[0], shape[0]]),
        np.min([original_shape[1], shape[1]]),
    )
    new_ft_potential = np.zeros(shape, dtype=complex)

    # See https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html for the choice of frequencies
    # We want to map points to the location with the same frequency in the final ft grid
    # We want points with ftt freq from -(n)//2 to -1 in uhp
    kx_floor = sample_shape[0] // 2
    ky_floor = sample_shape[1] // 2
    # We want points with ftt freq from 0 to (n+1)//2 - 1 in lhp
    kx_ceil = (sample_shape[0] + 1) // 2
    ky_ceil = (sample_shape[1] + 1) // 2

    new_ft_potential[:kx_ceil, :ky_ceil] = ft_potential[:kx_ceil, :ky_ceil]
    if kx_floor != 0:
        new_ft_potential[-kx_floor:, :ky_ceil] = ft_potential[-kx_floor:, :ky_ceil]
    if ky_floor != 0:
        new_ft_potential[:kx_ceil, -ky_floor:] = ft_potential[:kx_ceil, -ky_floor:]
    if ky_floor != 0 and ky_floor != 0:
        new_ft_potential[-kx_floor:, -ky_floor:] = ft_potential[-kx_floor:, -ky_floor:]

    new_points = np.fft.ifft2(new_ft_potential, norm="forward")

    # A 2% difference in the output, since we dont make use of the fact the potential is real
    # therefore if the input has an even number of points the output is no longer
    # hermitian!
    # np.testing.assert_array_equal(np.abs(new_points), np.real_if_close(new_points))

    return new_points.tolist()


# The old method, which would produce a wavefunction
# which was slightly asymmetric if supplied wth
# a symmetric wavefunction
# def interpolate_points_fourier(
#     points: List[List[float]], shape: Tuple[int, int]
# ) -> List[List[float]]:
#     """
#     Given a uniform grid of points in the unit cell interpolate
#     a grid of points with the given shape using the fourier transform
#     """
#     ft_potential = np.fft.ifft2(points)
#     ft_indices = get_ft_indexes((ft_potential.shape[0], ft_potential.shape[1]))

#     # List of [x1_frac, x2_frac] for the interpolated grid
#     fractions = get_point_fractions(shape, endpoint=False)
#     # print(fractions)
#     # print(ft_indices)
#     # List of (List of list of [x1_phase, x2_phase] for the interpolated grid)
#     interpolated_phases = np.multiply(
#         fractions[:, np.newaxis, np.newaxis, :],
#         ft_indices[np.newaxis, :, :, :],
#     )
#     # Sum over phase from x and y, raise to exp(-i * phi)
#     summed_phases = np.exp(-2j * np.pi * np.sum(interpolated_phases, axis=-1))
#     # print(summed_phases)
#     # Multiply the exponential by the prefactor form the fourier transform
#     # Add the contribution from each ikx1, ikx2
#     interpolated_points = np.sum(
#         np.multiply(ft_potential[np.newaxis, :, :], summed_phases), axis=(1, 2)
#     )
#     return np.real_if_close(interpolated_points).reshape(shape).tolist()
