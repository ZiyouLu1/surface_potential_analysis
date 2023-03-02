from typing import List, Tuple

import numpy as np

from surface_potential_analysis.brillouin_zone import get_coordinate_fractions
from surface_potential_analysis.energy_data import (
    EnergyGrid,
    EnergyPoints,
    get_ft_phases,
    interpolate_energy_grid_fourier,
    interpolate_energy_grid_z_spline,
    load_energy_grid,
    load_energy_grid_legacy,
    load_energy_points,
    normalize_energy,
    save_energy_grid,
    truncate_energy,
    undo_truncate_energy,
)
from surface_potential_analysis.surface_config import get_surface_xy_points

from .surface_data import get_data_path


def load_raw_data() -> EnergyPoints:
    path = get_data_path("raw_data.json")
    return load_energy_points(path)


def load_raw_data_reciprocal_grid():
    path = get_data_path("raw_data_reciprocal_spacing.json")
    return load_energy_grid(path)


def load_john_interpolation() -> EnergyGrid:
    path = get_data_path("john_interpolated_data.json")
    return load_energy_grid_legacy(path)


def load_interpolated_grid() -> EnergyGrid:
    path = get_data_path("interpolated_data.json")
    return load_energy_grid(path)


def load_interpolated_john_grid() -> EnergyGrid:
    path = get_data_path("interpolated_data_john_grid.json")
    return load_energy_grid(path)


def generate_raw_unit_cell_data() -> None:
    data = load_raw_data()
    x_points = np.array(data["x_points"])
    y_points = np.array(data["y_points"])
    z_points = np.array(data["z_points"])

    x_c = np.sort(np.unique(x_points))
    y_c = np.sort(np.unique(y_points))
    z_c = np.sort(np.unique(z_points))
    points = np.array(data["points"])

    is_top = np.logical_and(x_points == x_c[0], y_points == y_c[0])
    top_points = [points[np.logical_and(is_top, z_points == z)][0] for z in z_c]

    is_top_hcp = np.logical_and(x_points == x_c[0], y_points == y_c[2])
    top_hcp_points = [points[np.logical_and(is_top_hcp, z_points == z)][0] for z in z_c]

    is_hcp = np.logical_and(x_points == x_c[0], y_points == y_c[4])
    hcp_points = [points[np.logical_and(is_hcp, z_points == z)][0] for z in z_c]

    is_top_fcc = np.logical_and(x_points == x_c[1], y_points == y_c[1])
    top_fcc_points = [points[np.logical_and(is_top_fcc, z_points == z)][0] for z in z_c]

    is_fcc_hcp = np.logical_and(x_points == x_c[1], y_points == y_c[3])
    fcc_hcp_points = [points[np.logical_and(is_fcc_hcp, z_points == z)][0] for z in z_c]

    is_fcc = np.logical_and(x_points == x_c[2], y_points == y_c[2])
    fcc_points = [points[np.logical_and(is_fcc, z_points == z)][0] for z in z_c]

    # Turns out we don't have enough points to produce an 'out' grid in real space.
    # We therefore have to use the reciprocal grid of points
    hh = hcp_points
    hf = fcc_hcp_points
    ff = fcc_points
    tf = top_fcc_points
    tt = top_points
    th = top_hcp_points

    reciprocal_points = [
        [ff, tf, tt, th, hh, hf],
        [hf, th, tf, hf, th, tf],
        [hh, hf, ff, tf, tt, th],
        [th, tf, hf, th, tf, hf],
        [tt, th, hh, hf, ff, tf],
        [tf, hf, th, tf, hf, th],
    ]

    length = np.max(y_points) - np.min(y_points)
    grid: EnergyGrid = {
        "delta_x0": (3 * length * (np.sqrt(3) / 2), 3 * length * (1 / 2)),
        "delta_x1": (0, 3 * length),
        "points": reciprocal_points,
        "z_points": z_c.tolist(),
    }
    path = get_data_path("raw_data_reciprocal_spacing.json")
    save_energy_grid(grid, path)


def interpolate_points_fourier_nickel(
    points: List[List[float]],
    delta_x0_reciprocal: Tuple[float, float],
    delta_x1_reciprocal: Tuple[float, float],
    delta_x0_real: Tuple[float, float],
    delta_x1_real: Tuple[float, float],
    shape: Tuple[int, int],
) -> List[List[float]]:
    """
    Given a uniform grid of points in the reciprocal spacing interpolate
    a grid of points with the given shape into the real spacing using the fourier transform
    """
    ft_potential = np.fft.fft2(points, axes=(0, 1), norm="forward")
    ft_phases = get_ft_phases((ft_potential.shape[0], ft_potential.shape[1]))

    # List of [x1_frac, x2_frac] for the interpolated grid
    coordinates = get_surface_xy_points(
        {
            "delta_x0": delta_x0_real,
            "delta_x1": delta_x1_real,
        },
        shape=shape,
    ).reshape(-1, 2)
    fractions = get_coordinate_fractions(
        delta_x0_reciprocal, delta_x1_reciprocal, coordinates
    )

    # List of (List of list of [x1_phase, x2_phase] for the interpolated grid)
    interpolated_phases = np.multiply(
        fractions[:, np.newaxis, np.newaxis, :],
        ft_phases[np.newaxis, :, :, :],
    )
    # Sum over phase from x and y, raise to exp(-i * phi)
    summed_phases = np.exp(1j * np.sum(interpolated_phases, axis=-1))
    # Multiply the exponential by the prefactor form the fourier transform
    # Add the contribution from each ikx1, ikx2
    interpolated_points = np.sum(
        np.multiply(ft_potential[np.newaxis, :, :], summed_phases), axis=(1, 2)
    )
    np.testing.assert_array_almost_equal(
        interpolated_points, np.abs(interpolated_points)
    )
    return np.abs(interpolated_points).reshape(shape).tolist()


def interpolate_energy_grid_xy_fourier_nickel(
    data: EnergyGrid,
    delta_x0_real: Tuple[float, float],
    delta_x1_real: Tuple[float, float],
    shape: Tuple[int, int] = (40, 40),
) -> EnergyGrid:
    """
    Makes use of a fourier transform to increase the number of points
    in the xy plane of the energy grid
    """
    old_points = np.array(data["points"])
    points = np.empty((shape[0], shape[1], old_points.shape[2]))

    for iz in range(old_points.shape[2]):
        points[:, :, iz] = interpolate_points_fourier_nickel(
            old_points[:, :, iz].tolist(),
            data["delta_x0"],
            data["delta_x1"],
            delta_x0_real,
            delta_x1_real,
            shape,
        )
    return {
        "delta_x0": delta_x0_real,
        "delta_x1": delta_x1_real,
        "points": points.tolist(),
        "z_points": data["z_points"],
    }


def interpolate_energy_grid_fourier_nickel(
    data: EnergyGrid,
    delta_x0_real: Tuple[float, float],
    delta_x1_real: Tuple[float, float],
    shape: Tuple[int, int, int] = (40, 40, 40),
) -> EnergyGrid:
    """
    Interpolate an energy grid using the fourier method, but in the xy direction we
    ignore the initial lattice constants
    """
    xy_interpolation = interpolate_energy_grid_xy_fourier_nickel(
        data, delta_x0_real, delta_x1_real, (shape[0], shape[1])
    )
    return interpolate_energy_grid_z_spline(xy_interpolation, shape[2])


def load_cleaned_energy_grid():
    data = load_raw_data_reciprocal_grid()
    normalized = normalize_energy(data)
    # cutoff=4E-19, n=6, offset = 1e-20
    # cutoff=3.2e-18, n=1, offset=1e-20
    return truncate_energy(normalized, cutoff=0.4e-18, n=1, offset=1e-20)


def generate_interpolated_data():
    raw_grid = load_raw_data_reciprocal_grid()
    normalized = normalize_energy(raw_grid)
    truncated = truncate_energy(normalized, cutoff=0.4e-18, n=1, offset=1e-20)

    raw_data = load_raw_data()
    x_width = np.max(raw_data["x_points"]) - np.min(raw_data["x_points"])

    delta_x0_real = (2 * x_width, 0)
    delta_x1_real = (0.5 * delta_x0_real[0], np.sqrt(3) * delta_x0_real[0] / 2)

    data = interpolate_energy_grid_fourier_nickel(
        truncated, delta_x0_real, delta_x1_real, (60, 60, 100)
    )
    fixed_data = undo_truncate_energy(data, cutoff=0.4e-18, n=1, offset=1e-20)
    path = get_data_path("interpolated_data.json")
    save_energy_grid(fixed_data, path)


def generate_interpolated_data_john_grid():
    raw = load_raw_data_reciprocal_grid()
    normalized = normalize_energy(raw)
    truncated = truncate_energy(normalized, cutoff=0.4e-18, n=1, offset=1e-20)

    raw_data = load_raw_data()
    x_width = np.max(raw_data["x_points"]) - np.min(raw_data["x_points"])
    y_width = np.max(raw_data["y_points"]) - np.min(raw_data["y_points"])

    delta_x0_real = (2 * x_width, 0)
    delta_x1_real = (0, 3 * y_width)

    data = interpolate_energy_grid_fourier_nickel(
        truncated, delta_x0_real, delta_x1_real, (60, 60, 100)
    )
    fixed_data = undo_truncate_energy(data, cutoff=0.4e-18, n=1, offset=1e-20)
    path = get_data_path("interpolated_data_john_grid.json")
    save_energy_grid(fixed_data, path)


def generate_interpolated_data_reciprocal():
    raw = load_raw_data_reciprocal_grid()
    normalized = normalize_energy(raw)
    truncated = truncate_energy(normalized, cutoff=0.4e-18, n=1, offset=1e-20)

    data = interpolate_energy_grid_fourier(truncated, (48, 48, 100))
    fixed_data = undo_truncate_energy(data, cutoff=0.4e-18, n=1, offset=1e-20)
    path = get_data_path("interpolated_data_reciprocal.json")
    save_energy_grid(fixed_data, path)
