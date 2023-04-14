import itertools
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import scipy.fft
import scipy.interpolate

from surface_potential_analysis.util import slice_along_axis

_DT = TypeVar("_DT", bound=np.dtype[Any])


def pad_ft_points(
    array: np.ndarray[tuple[int, ...], _DT], s: Sequence[int], axes: Sequence[int]
) -> np.ndarray[tuple[int, ...], _DT]:
    """
    Pad the points in the fourier transform with zeros.

    Pad the points in the fourier transform with zeros, keeping the frequencies of
    each point the same in the initial and final grid.

    Parameters
    ----------
    array : NDArray
        The array to pad
    s : Sequence[int]
        The length along each axis to pad or truncate to
    axes : NDArray
        The list of axis to pad

    Returns
    -------
    NDArray
        The padded array
    """
    shape_arr = np.asarray(array.shape)
    axes_arr = np.asarray(axes)

    padded_shape = shape_arr.copy()
    padded_shape[axes_arr] = s
    padded: np.ndarray[tuple[int, ...], _DT] = np.zeros(
        shape=padded_shape, dtype=array.dtype
    )

    slice_start = np.array([slice(None) for _ in array.shape], dtype=slice)
    slice_start[axes_arr] = np.array(
        [
            slice(1 + min((n - 1) // 2, (s - 1) // 2))
            for (n, s) in zip(shape_arr[axes_arr], s, strict=True)
        ],
        dtype=slice,
    )
    slice_end = np.array([slice(None) for _ in array.shape], dtype=slice)
    slice_end[axes_arr] = np.array(
        [
            slice(start, None) if (start := max((-n + 1) // 2, (-s + 1) // 2)) < 0
            # else no negative frequencies
            else slice(0, 0)
            for (n, s) in zip(shape_arr[axes_arr], s, strict=True)
        ],
        dtype=slice,
    )
    # For each combination of start/end region of the array
    # add in the corresponding values to the padded array
    for slices in itertools.product(*np.array([slice_start, slice_end]).T.tolist()):
        padded[tuple(slices)] = array[tuple(slices)]

    return padded


def interpolate_points_fftn(
    points: np.ndarray[Any, Any],
    s: Sequence[int],
    axes: Sequence[int] | None = None,
) -> np.ndarray[Any, np.dtype[np.complex_]]:
    """
    Interpolate a grid of points with the given shape using the fourier transform.

    We don't make use of the fact that the potential is real in this case,
    and as such the output is not guaranteed to be real if the input is real

    Parameters
    ----------
    points : ArrayLike
        The initial set of points to interpolate
    s : Sequence[int]
        Shape (length of each transformed axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.).
    axes : Sequence[int] | None, optional
        Axes over which to compute the FFT. If not given, the last len(s) axes are used, or all axes if s is also not specified.

    Returns
    -------
    NDArray
        The points interpolated along the given axes
    """
    axes_arr = np.arange(-1, -1 - len(s), -1) if axes is None else np.array(axes)
    # We use the forward norm here, as otherwise we would also need to
    # scale the ft_potential by a factor of n / shape[axis]
    # when we pad or truncate it
    ft_points = scipy.fft.fftn(points, axes=axes_arr, norm="forward")
    # pad (or truncate) for the new lengths s
    padded = pad_ft_points(ft_points, s, axes_arr)
    return scipy.fft.ifftn(  # type:ignore[no-any-return]
        padded, s, axes=axes_arr, norm="forward", overwrite_x=True
    )


def pad_ft_points_real_axis(
    array: np.ndarray[tuple[int, ...], _DT], n: int, axis: int = -1
) -> np.ndarray[tuple[int, ...], _DT]:
    """
    Pad the points in the fourier transform with zeros, along the 'real' axis.

    Pad the points in the fourier transform with zeros, keeping the frequencies of
    each point the same in the initial and final grid, using the rfft storage of
    frequencies. Note n is the length of the 'non-ft' points, not the length of
    the resulting array in the 'axis' direction

    Parameters
    ----------
    array : NDArray
        The array to pad
    n : int
        The target length of the non-ft points (the new length along the given axis is n//2 + 1)
    axes : NDArray
        The axis to pad

    Returns
    -------
    NDArray
        The padded array
    """
    padded_shape = np.array(array.shape)
    padded_shape[axis] = n // 2 + 1

    padded: np.ndarray[tuple[int, ...], _DT] = np.zeros(
        shape=padded_shape, dtype=array.dtype
    )
    relevant_slice = slice(min(padded.shape[axis], array.shape[axis]))
    padded[slice_along_axis(relevant_slice, axis)] = array[
        slice_along_axis(relevant_slice, axis)
    ]

    return padded


def interpolate_points_rfftn(
    points: np.ndarray[tuple[int, ...], np.dtype[np.float_]],
    s: Sequence[int],
    axes: Sequence[int] | None = None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float_]]:
    """
    Interpolate points using a real fourier transform.

    This method performs poorly compared to fft2 if there is some large
    maximum in the potential, due to weirdness with the automatic ft
    points padding

    Parameters
    ----------
    points : np.ndarray[tuple, np.dtype[np.float_]]
    s : Sequence[int]
    axes : Sequence[int] | None, optional

    Returns
    -------
    np.ndarray[tuple, np.dtype[np.float_]]
    """
    axes_arr = np.arange(-1, -1 - len(s), -1) if axes is None else np.array(axes)
    ft_points = scipy.fft.rfftn(points, axes=axes_arr, norm="forward")
    # pad (or truncate) for the new lengths s
    # we don't need to pad the last axis here, as it is handled correctly by irfftn
    padded = pad_ft_points(ft_points, s[:-1], axes_arr[:-1])
    return scipy.fft.irfftn(  # type:ignore[no-any-return]
        padded, s, axes=axes_arr, norm="forward", overwrite_x=True
    )


def interpolate_points_rfft(
    points: np.ndarray[tuple[int, ...], np.dtype[np.float_]], n: int, axis: int = -1
) -> np.ndarray[tuple[int, ...], np.dtype[np.float_]]:
    """
    interpolate along the given axis to a new length n.

    Given a uniformly spaced (real) grid of points interpolate along the given
    axis to a new length n.

    This makes use of the fact that the potential is real, and therefore if the
    input is even along the interpolation axis we get an additional ft point for 'free'
    using the hermitian property of the fourier transform

    Parameters
    ----------
    points :  np.ndarray[tuple[int, ...], np.dtype[np.float_]]
        The initial set of points to interpolate
    n : int
        The number of points in the interpolated grid
    axis : int, optional
        The axis over which to interpolate, by default -1

    Returns
    -------
     np.ndarray[tuple[int, ...], np.dtype[np.float_]]
        The points interpolated along the given axis
    """
    # We use the forward norm here, as otherwise we would also need to
    # scale the ft_potential by a factor of n / shape[axis]
    # when we pad or truncate it
    ft_potential = scipy.fft.rfft(points, axis=axis, norm="forward")
    # Invert the rfft, padding (or truncating) for the new length n
    interpolated_potential = scipy.fft.irfft(ft_potential, n, axis=axis, norm="forward")

    if np.all(np.isreal(ft_potential)):
        # Force the symmetric potential to stay symmetric
        # Due to issues with numerical precision it diverges by around 1E-34
        lower_half = slice_along_axis(slice(1, (n + 1) // 2), axis=axis)
        upper_half = slice_along_axis(slice(None, n // 2, -1), axis=axis)
        interpolated_potential[upper_half] = interpolated_potential[lower_half]

    return interpolated_potential  # type:ignore[no-any-return]


# ruff: noqa: ERA001


# The old method, of padding which only worked for a 2D array
# def interpolate_points_fourier_complex(
#     points: list[list[complex]], shape: tuple[int, int]
# ) -> list[list[complex]]:
#     """
#     Given a uniform grid of points in the unit cell interpolate
#     a grid of points with the given shape using the fourier transform

#     We don't make use of the fact that the potential is real in this case,
#     and as such the output may not be real
#     """
#     ft_potential = np.fft.fft2(points, norm="forward")
#     original_shape = ft_potential.shape
#     sample_shape = (
#         np.min([original_shape[0], shape[0]]),
#         np.min([original_shape[1], shape[1]]),
#     )
#     new_ft_potential = np.zeros(shape, dtype=complex)

#     # See https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html for the choice of frequencies
#     # We want to map points to the location with the same frequency in the final ft grid
#     # We want points with ftt freq from -(n)//2 to -1 in uhp
#     kx_floor = sample_shape[0] // 2
#     ky_floor = sample_shape[1] // 2
#     # We want points with ftt freq from 0 to (n+1)//2 - 1 in lhp
#     kx_ceil = (sample_shape[0] + 1) // 2
#     ky_ceil = (sample_shape[1] + 1) // 2

#     new_ft_potential[:kx_ceil, :ky_ceil] = ft_potential[:kx_ceil, :ky_ceil]
#     if kx_floor != 0:
#         new_ft_potential[-kx_floor:, :ky_ceil] = ft_potential[-kx_floor:, :ky_ceil]
#     if ky_floor != 0:
#         new_ft_potential[:kx_ceil, -ky_floor:] = ft_potential[:kx_ceil, -ky_floor:]
#     if ky_floor != 0 and ky_floor != 0:
#         new_ft_potential[-kx_floor:, -ky_floor:] = ft_potential[-kx_floor:, -ky_floor:]

#     new_points = np.fft.ifft2(new_ft_potential, norm="forward")
#     return new_points.tolist()


# The old method, which would produce a wavefunction
# which was slightly asymmetric if supplied wth
# a symmetric wavefunction
# def interpolate_points_fourier(
#     points: list[list[float]], shape: tuple[int, int]
# ) -> list[list[float]]:
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


def interpolate_points_along_axis_spline(
    data: np.ndarray[tuple[int, ...], np.dtype[np.float_]],
    old_coords: np.ndarray[tuple[int], np.dtype[np.float_]],
    n: int,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float_]]:
    """Use a spline interpolation to increase the Z resolution, spacing z linearly."""
    new_coords = list(np.linspace(old_coords[0], old_coords[-1], n))

    new_shape = list(data.shape)
    new_shape[axis] = n
    points = np.empty(new_shape)

    swapped_points = points.swapaxes(axis, -1).reshape(-1, n)

    flat_data = data.swapaxes(axis, -1).reshape(-1, data.shape[axis])

    for i in range(flat_data.shape[0]):
        old_energies = flat_data[i]
        tck = scipy.interpolate.splrep(old_coords, old_energies, s=0)
        new_energy = scipy.interpolate.splev(new_coords, tck, der=0)
        swapped_points[i] = new_energy

    return points  # type: ignore[no-any-return]
