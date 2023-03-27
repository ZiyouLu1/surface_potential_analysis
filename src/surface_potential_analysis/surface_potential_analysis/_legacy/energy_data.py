import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import scipy.interpolate
from numpy.typing import NDArray

from .interpolation import interpolate_real_points_along_axis_fourier
from .surface_config import (
    SurfaceConfig,
    get_surface_coordinates,
    get_surface_xy_points,
)


class EnergyPoints(TypedDict):
    x_points: list[float]
    y_points: list[float]
    z_points: list[float]
    points: list[float]


def load_energy_points(path: Path) -> EnergyPoints:
    with path.open("r") as f:
        return json.load(f)


def save_energy_points(data: EnergyPoints, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f)


def get_energy_points_xy_locations(data: EnergyPoints) -> list[tuple[float, float]]:
    x_points = np.array(data["x_points"])
    return [
        (x, y)
        for x in np.unique(x_points)
        for y in np.unique(np.array(data["y_points"])[x_points == x])
    ]


class EnergyGrid(SurfaceConfig):
    """
    A grid of energy points uniformly spaced in the x1,x2 direction
    And possibly unevenly spaced in the z direction
    """

    z_points: list[float]
    points: list[list[list[float]]]


class EnergyGridRaw(TypedDict):
    """
    A grid of energy points uniformly spaced in the x1,x2 direction
    And possibly unevenly spaced in the z direction
    """

    delta_x0: list[float]
    delta_x1: list[float]
    z_points: list[float]
    points: list[list[list[float]]]


def load_energy_grid(path: Path) -> EnergyGrid:
    with path.open("r") as f:
        out: EnergyGridRaw = json.load(f)
        return {
            "delta_x0": (out["delta_x0"][0], out["delta_x0"][1]),
            "delta_x1": (out["delta_x1"][0], out["delta_x1"][1]),
            "points": out["points"],
            "z_points": out["z_points"],
        }


def save_energy_grid(data: EnergyGrid, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f)


def get_energy_grid_xy_points(grid: EnergyGrid) -> NDArray:
    shape = np.shape(grid["points"])
    return get_surface_xy_points(grid, (shape[0], shape[1])).reshape(-1, 2)


def get_energy_grid_coordinates(
    grid: EnergyGrid, *, offset: tuple[float, float] = (0.0, 0.0)
) -> NDArray:
    points = np.array(grid["points"])

    return get_surface_coordinates(
        grid, (points.shape[0], points.shape[1]), grid["z_points"], offset=offset
    )


class EnergyInterpolation(TypedDict):
    dz: float
    # Note - points should exclude the 'nth' point
    points: list[list[list[float]]]


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
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "z_points": data["z_points"],
    }


# Attempt to fill from one corner.
# We don't have enough points (or the Hydrogen is not fixed enough)
# to be able to 'fill' the whole region we want
def fill_subsurface_from_corner(data: EnergyGrid) -> EnergyGrid:
    points = np.array(data["points"], dtype=float)
    points_to_fill: set[tuple[int, int, int]] = set([(0, 0, 0)])
    fill_level = 1.6

    while len(points_to_fill) > 0:
        current_point = points_to_fill.pop()

        if points[current_point] >= fill_level:
            continue

        points[current_point] = 1000
        for dx, dy, dz in [
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
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
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
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
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
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "z_points": data["z_points"],
    }


def truncate_energy(
    data: EnergyGrid, *, cutoff=2e-17, n: int = 1, offset: float = 2e-18
) -> EnergyGrid:
    """
    Reduce the maximum energy by taking the transformation

    :math:`cutoff * np.log(1 + ((E + offset) / cutoff) ** n) ** (1 / n) - offset`

    For :math:`E << Cutoff` the energy is left unchanged. This can be useful to
    prevent the energy interpolation process from producing rabid oscillations
    """
    points = np.array(data["points"], dtype=float)
    truncated_points = (
        cutoff * np.log(1 + ((points + offset) / cutoff) ** n) ** (1 / n) - offset
    )
    return {
        "points": truncated_points.tolist(),
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "z_points": data["z_points"],
    }


def undo_truncate_energy(
    data: EnergyGrid, *, cutoff=2e-17, n: int = 1, offset: float = 2e-18
) -> EnergyGrid:
    """
    The reverse of truncate_energy
    """
    truncated_points = np.array(data["points"], dtype=float)
    points = (
        cutoff * (np.exp((truncated_points + offset) / cutoff) - 1) ** (1 / n) - offset
    )
    return {
        "points": points.tolist(),
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
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
        "delta_x0": (data["delta_x0"][0] * x1_tile, data["delta_x0"][1] * x1_tile),
        "delta_x1": (data["delta_x1"][0] * x1_tile, data["delta_x1"][1] * x2_tile),
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
    # Randomly spaced z points to reduce oscillation
    z_points[-extend_by:] = np.sort(
        np.random.uniform(
            low=data["z_points"][-1],
            high=data["z_points"][-1] + dz1,
            size=extend_by,
        )
    )

    return {
        "points": z_extended_points.tolist(),
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "z_points": z_points.tolist(),
    }


def add_back_symmetry_points(
    data: list[list[list[float]]],
) -> list[list[list[float]]]:
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

    if (data["delta_x0"][1] != 0) or (data["delta_x1"][0] != 0):
        raise AssertionError("Not orthogonal grid")

    x_points = np.linspace(
        -data["delta_x0"][0], 2 * data["delta_x0"][0], points.shape[0]
    )
    y_points = np.linspace(
        -data["delta_x1"][1], 2 * data["delta_x1"][1], points.shape[1]
    )
    return scipy.interpolate.RegularGridInterpolator(
        [x_points, y_points, fixed_data["z_points"]],
        points,
    )


def interpolate_energy_grid_3D_spline(
    data: EnergyGrid, shape: tuple[int, int, int] = (40, 40, 100)
) -> EnergyGrid:
    """
    Use the 3D cubic spline method to interpolate points.

    Note this requires that the two unit vectors in the xy plane are orthogonal
    """

    interpolator = generate_interpolator(data)

    z_points = list(np.linspace(data["z_points"][0], data["z_points"][-1], shape[2]))
    new_grid: EnergyGrid = {
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "z_points": z_points,
        "points": np.empty(shape).tolist(),
    }
    test_points = get_energy_grid_coordinates(new_grid)

    points = interpolator(test_points.reshape(np.prod(shape), 3), method="quintic")

    return {
        "points": points.reshape(*shape).tolist(),
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "z_points": z_points,
    }


def interpolate_energy_grid_z_spline(data: EnergyGrid, nz: int = 100) -> EnergyGrid:
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
    for x, y in old_xy_points:
        old_energies = data["points"][x][y]
        tck = scipy.interpolate.splrep(data["z_points"], old_energies, s=0)
        new_energy = scipy.interpolate.splev(z_points, tck, der=0)
        points[x, y] = new_energy

    return {
        "points": points.tolist(),
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "z_points": z_points,
    }


def get_ft_indexes(shape: tuple[int, int]):
    """
    Get a list of list of [x1_phase, x2_phase] for the fourier transform
    """
    ft_indices = np.indices(shape).transpose((1, 2, 0))

    x_indices = ft_indices[:, :, 0]
    above_halfway = x_indices > shape[0] // 2
    x_indices[above_halfway] = x_indices[above_halfway] - shape[0]
    ft_indices[:, :, 0] = x_indices

    y_indices = ft_indices[:, :, 1]
    above_halfway = y_indices > shape[1] // 2
    y_indices[above_halfway] = y_indices[above_halfway] - shape[1]
    ft_indices[:, :, 1] = y_indices

    # List of list of [x1_phase, x2_phase]
    return ft_indices


def get_ft_phases(shape: tuple[int, int]):
    """
    Get a list of list of [x1_phase, x2_phase] for the fourier transform
    """

    # List of list of [x1_phase, x2_phase]
    ft_phases = 2 * np.pi * get_ft_indexes(shape)
    return ft_phases


def interpolate_energy_grid_xy_fourier(
    data: EnergyGrid, shape: tuple[int, int] = (40, 40)
) -> EnergyGrid:
    """
    Makes use of a fourier transform to increase the number of points
    in the xy plane of the energy grid
    """
    old_points = np.array(data["points"])
    x_interp = interpolate_real_points_along_axis_fourier(old_points, shape[0], axis=0)
    y_interp = interpolate_real_points_along_axis_fourier(x_interp, shape[1], axis=1)
    return {
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "points": y_interp.tolist(),
        "z_points": data["z_points"],
    }


def interpolate_energy_grid_fourier(
    data: EnergyGrid, shape: tuple[int, int, int] = (40, 40, 40)
) -> EnergyGrid:
    """
    Interpolate an energy grid using the fourier method

    Makes use of a fourier transform to increase the number of points
    in the xy plane of the energy grid, and a cubic spline to interpolate in the z direction
    """

    xy_interpolation = interpolate_energy_grid_xy_fourier(data, (shape[0], shape[1]))
    return interpolate_energy_grid_z_spline(xy_interpolation, shape[2])
