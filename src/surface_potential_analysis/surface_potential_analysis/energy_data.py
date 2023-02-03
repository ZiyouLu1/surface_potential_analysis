import json
from pathlib import Path
from typing import List, Set, Tuple, TypedDict

import numpy as np
import scipy.interpolate
from numpy.typing import NDArray

from surface_potential_analysis.brillouin_zone import get_point_fractions, grid_space


class EnergyPoints(TypedDict):
    x_points: List[float]
    y_points: List[float]
    z_points: List[float]
    points: List[float]


def load_energy_points(path: Path) -> EnergyPoints:
    with path.open("r") as f:
        return json.load(f)


def save_energy_points(data: EnergyPoints, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f)


def get_energy_points_xy_locations(data: EnergyPoints) -> List[Tuple[float, float]]:
    x_points = np.array(data["x_points"])
    return [
        (x, y)
        for x in np.unique(x_points)
        for y in np.unique(np.array(data["y_points"])[x_points == x])
    ]


class EnergyGrid(TypedDict):
    """
    A grid of energy points uniformly spaced in the x1,x2 direction
    And possibly unevenly spaced in the z direction
    """

    delta_x1: Tuple[float, float]
    delta_x2: Tuple[float, float]
    z_points: List[float]
    points: List[List[List[float]]]


def load_energy_grid(path: Path) -> EnergyGrid:
    with path.open("r") as f:
        return json.load(f)


def save_energy_grid(data: EnergyGrid, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f)


def get_energy_grid_xy_points(grid: EnergyGrid) -> NDArray:
    points = np.array(grid["points"])
    return grid_space(
        grid["delta_x1"],
        grid["delta_x2"],
        shape=(points.shape[0], points.shape[1]),
        endpoint=False,
    )


def get_energy_grid_coordinates(grid: EnergyGrid) -> NDArray:
    points = np.array(grid["points"])
    xy_points = get_energy_grid_xy_points(grid).reshape(
        points.shape[0], points.shape[1], 2
    )
    z_points = np.array(grid["z_points"])

    tiled_x = (
        np.tile(xy_points[:, :, 0], (z_points.shape[0], 1, 1))
        .swapaxes(0, 1)
        .swapaxes(1, 2)
    )
    tiled_y = (
        np.tile(xy_points[:, :, 1], (z_points.shape[0], 1, 1))
        .swapaxes(0, 1)
        .swapaxes(1, 2)
    )
    tiled_z = np.tile(z_points, (xy_points.shape[0], xy_points.shape[1], 1))

    return (
        np.array([tiled_x, tiled_y, tiled_z])
        .swapaxes(0, 1)
        .swapaxes(1, 2)
        .swapaxes(2, 3)
    )


class EnergyGridLegacy(TypedDict):
    x_points: List[float]
    y_points: List[float]
    z_points: List[float]
    points: List[List[List[float]]]


def load_energy_grid_legacy_as_legacy(path: Path) -> EnergyGridLegacy:
    with path.open("r") as f:
        return json.load(f)


def save_energy_grid_legacy(data: EnergyGridLegacy, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f)


def energy_grid_legacy_as_energy_grid(data: EnergyGridLegacy) -> EnergyGrid:
    x_delta = get_xy_points_delta(data["x_points"])
    y_delta = get_xy_points_delta(data["y_points"])
    return {
        "delta_x1": (x_delta, 0),
        "delta_x2": (0, y_delta),
        "z_points": data["z_points"],
        "points": data["points"],
    }


class EnergyInterpolation(TypedDict):
    dz: float
    # Note - points should exclude the 'nth' point
    points: List[List[List[float]]]


def get_xy_points_delta(points: List[float]):
    # Note additional factor to account for 'missing' point
    return (len(points)) * (points[-1] - points[0]) / (len(points) - 1)


def as_interpolation(data: EnergyGrid) -> EnergyInterpolation:
    """
    Converts between energy data and energy interpolation,
    assuming the x,y,z points are evenly spaced
    """

    dz = data["z_points"][1] - data["z_points"][0]
    nz = len(data["z_points"])
    z_points = np.linspace(0, dz * (nz - 1), nz) + data["z_points"][0]
    if not np.allclose(z_points, data["z_points"]):
        raise AssertionError("z Points Not evenly spaced")

    return {"dz": dz, "points": data["points"]}


def normalize_energy(data: EnergyGrid) -> EnergyGrid:
    points = np.array(data["points"], dtype=float)
    normalized_points = points - points.min()
    return {
        "points": normalized_points.tolist(),
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "z_points": data["z_points"],
    }


# Attempt to fill from one corner.
# We don't have enough points (or the Hydrogen is not fixed enough)
# to be able to 'fill' the whole region we want
def fill_subsurface_from_corner(data: EnergyGrid) -> EnergyGrid:
    points = np.array(data["points"], dtype=float)
    points_to_fill: Set[Tuple[int, int, int]] = set([(0, 0, 0)])
    fill_level = 1.6

    while len(points_to_fill) > 0:
        current_point = points_to_fill.pop()

        if points[current_point] >= fill_level:
            continue

        points[current_point] = 1000
        for (dx, dy, dz) in [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
        ]:

            next_point = (
                (current_point[0] + dx) % points.shape[0],
                (current_point[1] + dy) % points.shape[1],
                current_point[2] + dz,
            )
            points_to_fill.add(next_point)

    return {
        "points": points.tolist(),
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "z_points": data["z_points"],
    }


def fill_subsurface_from_hollow_sample(data: EnergyGrid) -> EnergyGrid:
    points = np.array(data["points"], dtype=float)

    # Number of points to fill
    fill_height = 5
    hollow_sample = points[5, 5, :fill_height]

    points[:, :, :fill_height] = 0.5 * points[:, :, :fill_height] + 0.5 * hollow_sample

    return {
        "points": points.tolist(),
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "z_points": data["z_points"],
    }


# Fills the surface below the maximum pof the potential
# Since at the bridge point the maximum is at r-> infty we must take
# the maximum within the first half of the data
def fill_surface_from_z_maximum(data: EnergyGrid) -> EnergyGrid:
    points = np.array(data["points"], dtype=float)
    max_arg = np.argmax(points[:, :, :10], axis=2, keepdims=True)
    max_val = np.max(points[:, :, :10], axis=2, keepdims=True)

    z_index = np.indices(dimensions=points.shape)[2]
    should_use_max = z_index < max_arg
    points = np.where(should_use_max, max_val, points)

    return {
        "points": points.tolist(),
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "z_points": data["z_points"],
    }


def truncate_energy(
    data: EnergyGrid, *, cutoff=2e-17, n: int = 1, offset: float = 2e-18
) -> EnergyGrid:
    points = np.array(data["points"], dtype=float)
    truncated_points = (
        cutoff * np.log(1 + ((points + offset) / cutoff) ** n) ** (1 / n) - offset
    )
    return {
        "points": truncated_points.tolist(),
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "z_points": data["z_points"],
    }


def repeat_original_data(
    data: EnergyGrid, x1_padding: int = 1, x2_padding: int = 1
) -> EnergyGrid:
    """Repeat the original data using the x, y symmetry to improve the interpolation convergence"""

    points = np.array(data["points"])

    x1_tile = 1 + 2 * x1_padding
    x2_tile = 1 + 2 * x2_padding
    xy_extended_data = np.tile(points, (x1_tile, x2_tile, 1))

    return {
        "points": xy_extended_data.tolist(),
        "delta_x1": (data["delta_x1"][0] * x1_tile, data["delta_x1"][1] * x1_tile),
        "delta_x2": (data["delta_x2"][0] * x1_tile, data["delta_x2"][1] * x2_tile),
        "z_points": data["z_points"],
    }


def extend_z_data(data: EnergyGrid, extend_by: int = 2) -> EnergyGrid:
    old_points = np.array(data["points"])
    old_shape = old_points.shape
    z_len = old_shape[2] + 2 * extend_by
    z_extended_points = np.zeros(shape=(old_shape[0], old_shape[1], z_len))

    z_extended_points[:, :, extend_by:-extend_by] = old_points
    for x in range(extend_by):
        z_extended_points[:, :, x] = old_points[:, :, 0]
        z_extended_points[:, :, -(x + 1)] = old_points[:, :, -1]

    z_points = np.zeros(shape=(z_len))
    z_points[extend_by:-extend_by] = np.array(data["z_points"])

    dz0 = data["z_points"][1] - data["z_points"][0]
    z_points[:extend_by] = np.linspace(
        data["z_points"][0] - (extend_by * dz0),
        data["z_points"][0],
        extend_by,
        endpoint=False,
    )
    dz1 = data["z_points"][-1] - data["z_points"][-2]
    z_points[-extend_by:] = np.linspace(
        data["z_points"][-1],
        data["z_points"][-1] + (extend_by * dz1),
        extend_by + 1,
    )[1:]

    return {
        "points": z_extended_points.tolist(),
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "z_points": z_points.tolist(),
    }


def add_back_symmetry_points(
    data: List[List[List[float]]],
) -> List[List[List[float]]]:
    points = np.array(data)
    nx = points.shape[0] + 1
    ny = points.shape[1] + 1
    extended_shape = (nx, ny, points.shape[2])
    extended_points = np.zeros(shape=extended_shape)
    extended_points[:-1, :-1] = points
    extended_points[-1, :-1] = points[0, :]
    extended_points[:-1, -1] = points[:, 0]
    extended_points[-1, -1] = points[0, 0]

    return extended_points.tolist()


def generate_interpolator(
    data: EnergyGrid,
) -> scipy.interpolate.RegularGridInterpolator:
    fixed_data = extend_z_data(repeat_original_data(data))
    points = np.array(add_back_symmetry_points(fixed_data["points"]))
    if (data["delta_x1"][1] != 0) or (data["delta_x2"][0] != 0):
        raise AssertionError("Not orthogonal grid")

    x_points = np.linspace(
        -data["delta_x1"][0], 2 * data["delta_x1"][0], points.shape[0]
    )
    y_points = np.linspace(
        -data["delta_x2"][0], 2 * data["delta_x2"][0], points.shape[1]
    )
    return scipy.interpolate.RegularGridInterpolator(
        [x_points, y_points, fixed_data["z_points"]],
        fixed_data["points"],
    )


def interpolate_energies_grid(
    data: EnergyGrid, shape: Tuple[int, int, int] = (40, 40, 100)
) -> EnergyGrid:
    """
    Use the 3D cubic spline method to interpolate points.

    Note this requires that the two unit vectors in the xy plane are orthogonal
    """

    interpolator = generate_interpolator(data)

    z_points = list(np.linspace(data["z_points"][0], data["z_points"][-1], shape[2]))
    new_grid: EnergyGrid = {
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "z_points": z_points,
        "points": np.empty(shape).tolist(),
    }
    test_points = get_energy_grid_coordinates(new_grid)

    points = interpolator(test_points.reshape(np.prod(shape), 3), method="quintic")

    return {
        "points": points.reshape(*shape).tolist(),
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "z_points": z_points,
    }


def interpolate_energies_spline(data: EnergyGrid, nz: int = 100) -> EnergyGrid:
    """
    Uses spline interpolation to increase the Z resolution,
    spacing z linearly
    """
    old_points = np.array(data["points"])
    z_points = list(np.linspace(data["z_points"][0], data["z_points"][-1], nz))

    points = np.empty((old_points.shape[0], old_points.shape[1], nz))
    xt, yt = np.meshgrid(
        range(old_points.shape[0]), range(old_points.shape[1]), indexing="ij"
    )
    old_xy_points = np.array([xt.ravel(), yt.ravel()]).T
    for (x, y) in old_xy_points:
        old_energies = data["points"][x][y]
        tck = scipy.interpolate.splrep(data["z_points"], old_energies, s=0)
        new_energy = scipy.interpolate.splev(z_points, tck, der=0)
        points[x, y] = new_energy

    return {
        "points": points.tolist(),
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "z_points": z_points,
    }


def interpolate_points_fourier(
    points: List[List[float]], shape: Tuple[int, int]
) -> List[List[float]]:
    """
    Given a uniform grid of points in the unit cell interpolate
    a grid of points with the given shape using the fourier transform
    """
    ft_potential = np.fft.ifft2(points)

    ft_indices = np.indices(ft_potential.shape).transpose((1, 2, 0))
    # List of list of [x1_phase, x2_phase]
    ft_phases = 2 * np.pi * ft_indices

    # List of [x1_frac, x2_frac] for the interpolated grid
    fractions = get_point_fractions(shape, endpoint=False)

    # List of (List of list of [x1_phase, x2_phase] for the interpolated grid)
    interpolated_phases = np.multiply(
        fractions[:, np.newaxis, np.newaxis, :],
        ft_phases[np.newaxis, :, :, :],
    )
    # Sum over phase from x and y, raise to exp(-i * phi)
    summed_phases = np.exp(-1j * np.sum(interpolated_phases, axis=-1))
    # Multiply the exponential by the prefactor form the fourier transform
    # Add the contribution from each ikx1, ikx2
    interpolated_points = np.sum(
        np.multiply(ft_potential[np.newaxis, :, :], summed_phases), axis=(1, 2)
    )
    return np.real_if_close(interpolated_points).reshape(shape).tolist()


def interpolate_energy_grid_xy_fourier(
    data: EnergyGrid, shape: Tuple[int, int] = (40, 40)
) -> EnergyGrid:
    """
    Makes use of a fourier transform to increase the number of points
    in the xy plane of the energy grid
    """
    old_points = np.array(data["points"])
    points = np.empty((shape[0], shape[1], old_points.shape[2]))
    for iz in range(old_points.shape[2]):
        points[:, :, iz] = interpolate_points_fourier(
            old_points[:, :, iz].tolist(), shape
        )
    return {
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "points": points.tolist(),
        "z_points": data["z_points"],
    }


def interpolate_energy_grid_fourier(
    data: EnergyGrid, shape: Tuple[int, int, int] = (40, 40, 40)
) -> EnergyGrid:
    """
    Interpolate an energy grid using the fourier method

    Makes use of a fourier transform to increase the number of points
    in the xy plane of the energy grid, and a cubic spline to interpolate in the z direction
    """
    xy_interpolation = interpolate_energy_grid_xy_fourier(data, (shape[0], shape[1]))
    return interpolate_energies_spline(xy_interpolation, shape[2])
