import math
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from surface_potential_analysis.energy_data import (
    EnergyGrid,
    load_energy_grid,
    normalize_energy,
)
from surface_potential_analysis.energy_data_plot import (
    animate_energy_grid_3D_in_x1z,
    animate_energy_grid_3D_in_xy,
    compare_energy_grid_to_all_raw_points,
    plot_all_energy_points_z,
    plot_energy_grid_locations,
    plot_energy_grid_points,
    plot_energy_point_locations_on_grid,
    plot_energy_points_location,
    plot_potential_minimum_along_path,
    plot_z_direction_energy_comparison_111,
    plot_z_direction_energy_data,
    plot_z_direction_energy_data_111,
)
from surface_potential_analysis.sho_wavefunction_plot import plot_sho_wavefunctions

from .s1_potential import (
    load_cleaned_energy_grid,
    load_interpolated_grid,
    load_interpolated_john_grid,
    load_john_interpolation,
    load_raw_data,
    load_raw_data_reciprocal_grid,
)
from .surface_data import get_data_path, save_figure


def plot_raw_data_points():
    data = load_raw_data()
    fig, _, _ = plot_energy_points_location(data)
    fig.show()
    save_figure(fig, "nickel_raw_points.png")

    fig, ax = plot_all_energy_points_z(data)
    ax.set_ylim(0, 3 * 10**-19)

    ax.legend()
    fig.show()
    save_figure(fig, "nickel_raw_points_z.png")
    input()


def get_111_locations_nickel_reciprocal(grid: EnergyGrid):
    points = np.array(grid["points"], dtype=float)
    return {
        "HCP Site": (math.floor(points.shape[0] / 3), 0),
        "Bridge Site": (
            math.floor(points.shape[0] / 6),
            0,
        ),
        "Top Site": (0, math.floor(points.shape[0] / 3)),
        "FCC Site": (0, 0),
    }


def plot_z_direction_energy_data_nickel_reciprocal_points(
    grid: EnergyGrid, *, ax: Axes | None = None
) -> Tuple[Figure, Axes, Tuple[Line2D, Line2D, Line2D, Line2D]]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    locations = get_111_locations_nickel_reciprocal(grid)
    fig, ax, lines = plot_z_direction_energy_data(grid, locations, ax=ax)

    return fig, ax, (lines[0], lines[1], lines[2], lines[3])


def plot_raw_energy_grid_points():
    grid = normalize_energy(load_raw_data_reciprocal_grid())

    fig, _, _ = plot_energy_grid_points(grid)
    fig.show()

    fig, ax, _ = plot_z_direction_energy_data_nickel_reciprocal_points(grid)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    fig, _, _ani = animate_energy_grid_3D_in_xy(grid)
    fig.show()

    cleaned = load_cleaned_energy_grid()
    fig, ax = plot_z_direction_energy_comparison_111(cleaned, grid)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    ft_points = np.abs(np.fft.ifft2(grid["points"], axes=(0, 1)))
    ft_points[0, 0] = 0
    ft_grid: EnergyGrid = {
        "delta_x0": grid["delta_x0"],
        "delta_x1": grid["delta_x1"],
        "points": ft_points.tolist(),
        "z_points": grid["z_points"],
    }
    fig, ax, _ani1 = animate_energy_grid_3D_in_xy(ft_grid)
    # TODO: delta is wrong, plot generic points and then factor out into ft.
    ax.set_title("Plot of the ft of the raw potential")
    fig.show()
    input()


def plot_interpolated_energy_grid_points():
    grid = load_interpolated_grid()

    fig, ax, _ = plot_energy_grid_points(grid)
    fig.show()

    fig, ax, _ani = animate_energy_grid_3D_in_xy(grid, clim_max=1e-18)
    path = get_data_path("raw_data_reciprocal_spacing.json")
    raw_grid = load_energy_grid(path)
    plot_energy_grid_locations(raw_grid, ax=ax)
    fig.show()

    fig, ax, _ = plot_z_direction_energy_data_111(grid)
    ax.set_ylim(0, 5e-18)
    raw_grid = normalize_energy(load_raw_data_reciprocal_grid())
    _, _, lines = plot_z_direction_energy_data_nickel_reciprocal_points(raw_grid, ax=ax)
    for ln in lines:
        ln.set_marker("x")
        ln.set_linestyle("")
    fig.show()

    ft_points = np.abs(np.fft.ifft2(grid["points"], axes=(0, 1)))
    ft_points[0, 0] = 0
    ft_grid: EnergyGrid = {
        "delta_x0": grid["delta_x0"],
        "delta_x1": grid["delta_x1"],
        "points": ft_points.tolist(),
        "z_points": grid["z_points"],
    }
    fig, ax, _ani1 = animate_energy_grid_3D_in_xy(ft_grid)
    # TODO: delta is wrong, plot generic points and then factor out into ft.
    ax.set_title("Plot of the ft of the interpolated potential")
    fig.show()
    input()


def plot_interpolated_energy_grid_reciprocal():
    path = get_data_path("interpolated_data_reciprocal.json")
    grid = load_energy_grid(path)
    points = np.array(grid["points"])
    points[points < 0] = np.max(points)
    grid["points"] = points.tolist()

    fig, ax, _ = plot_energy_grid_points(grid)
    fig.show()

    fig, ax, _ani = animate_energy_grid_3D_in_xy(grid)
    path = get_data_path("raw_data_reciprocal_spacing.json")
    raw_grid = load_energy_grid(path)
    plot_energy_grid_locations(raw_grid, ax=ax)
    fig.show()

    fig, ax, _ = plot_z_direction_energy_data_111(grid)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    ft_points = np.abs(np.fft.ifft2(grid["points"], axes=(0, 1)))
    ft_points[0, 0] = 0
    ft_grid: EnergyGrid = {
        "delta_x0": grid["delta_x0"],
        "delta_x1": grid["delta_x1"],
        "points": ft_points.tolist(),
        "z_points": grid["z_points"],
    }
    fig, ax, _ani1 = animate_energy_grid_3D_in_xy(ft_grid)
    # TODO: delta is wrong, plot generic points and then factor out into ft.
    ax.set_title("Plot of the ft of the interpolated potential")
    fig.show()
    input()


def get_john_point_locations(grid: EnergyGrid):

    points = np.array(grid["points"], dtype=float)
    return {
        "Top Site": (0, 0),
        "Bridge Site": (0, math.floor(points.shape[1] / 2)),
        "FCC Site": (0, math.floor(points.shape[1] / 3)),
        "HCP Site": (0, math.floor(2 * points.shape[1] / 3)),
    }


def plot_z_direction_energy_data_john(
    grid: EnergyGrid, *, ax: Axes | None = None
) -> Tuple[Figure, Axes, Tuple[Line2D, Line2D, Line2D, Line2D]]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    locations = get_john_point_locations(grid)
    return plot_z_direction_energy_data(grid, locations, ax=ax)


def plot_john_interpolated_points():
    data = load_john_interpolation()

    fig, ax, _anim1 = animate_energy_grid_3D_in_xy(data)
    ax.set_title(
        "Plot of the interpolated Nickel surface potential\n" "through the x plane"
    )
    fig.show()

    fig, ax, _anim2 = animate_energy_grid_3D_in_x1z(data)
    ax.set_title(
        "Plot of the interpolated Nickel surface potential\n" "through the y plane"
    )

    fig.show()

    fig, ax, _ = plot_z_direction_energy_data_john(data)
    ax.set_ylim(0, 0.3e-18)
    fig.show()
    save_figure(fig, "john_interpolation_z.png")
    input()


def compare_john_interpolation():
    raw_points = load_raw_data()
    john_interpolation = load_john_interpolation()

    fig, ax, _anim2 = plot_energy_point_locations_on_grid(
        raw_points, john_interpolation
    )
    fig.show()

    fig, ax = compare_energy_grid_to_all_raw_points(raw_points, john_interpolation)
    ax.set_ylim(0, 3 * 10**-19)
    ax.set_title("Comparison between raw and interpolated potential for Nickel")
    fig.show()
    save_figure(fig, "raw_interpolation_comparison.png")

    fig, ax, _ = plot_z_direction_energy_data_john(john_interpolation)
    my_interpolation = load_interpolated_grid()
    plot_z_direction_energy_data_111(my_interpolation, ax=ax)
    plot_sho_wavefunctions(
        my_interpolation["z_points"],
        sho_omega=195636899474736.66,
        mass=1.6735575e-27,
        first_n=16,
        ax=ax,
    )
    ax.set_ylim(0, 0.5e-18)
    fig.show()
    save_figure(fig, "original_and_new_interpolation_comparison.png")
    input()


def plot_potential_minimum_along_diagonal():

    fig, ax = plt.subplots()

    interpolation = load_interpolated_grid()
    path = [(x, x) for x in range(np.shape(interpolation["points"])[0])]
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("My Interpolation")

    john_interpolation = load_john_interpolation()
    path = [(0, y) for y in range(np.shape(john_interpolation["points"])[1])]
    _, _, line = plot_potential_minimum_along_path(john_interpolation, path, ax=ax)
    line.set_label("John Interpolation")

    ax.set_title(
        "comparison of energy along the classical trajectory\n"
        "in the FCC-HCP-TOP direction"
    )
    ax.legend()
    fig.show()
    save_figure(fig, "classical_trajectory_comparison.png")
    input()


def test_potential_fourier_transform():
    """
    Since we are sampling in units of the bz we expect the potential to be the same at the origin
    as this just represents the 'average potential'.

    We also expect the off center to be equal,
    but the irrational unit vectors prevent us from testing this
    """
    interpolation = load_interpolated_grid()
    fft_me = np.fft.ifft2(interpolation["points"], axes=(0, 1))
    ftt_origin_me = fft_me[0, 0, np.argmin(np.abs(interpolation["z_points"]))]

    print(ftt_origin_me, np.min(np.abs(interpolation["z_points"])))

    x0_norm = np.linalg.norm(interpolation["delta_x0"])
    x1_norm = np.linalg.norm(interpolation["delta_x1"])
    denom = (
        interpolation["delta_x0"][0] * interpolation["delta_x1"][1]
        - interpolation["delta_x0"][1] * interpolation["delta_x1"][0]
    )
    fix_factor = x0_norm * x1_norm / (denom)
    print(ftt_origin_me / fix_factor)

    john_grid_interpolation = load_interpolated_john_grid()
    fft_john = np.fft.ifft2(john_grid_interpolation["points"], axes=(0, 1))
    ftt_origin_john = fft_john[
        0, 0, np.argmin(np.abs(john_grid_interpolation["z_points"]))
    ]

    print(ftt_origin_john, np.min(np.abs(john_grid_interpolation["z_points"])))

    # Good enough
    # Max absolute difference: 3.31267687e-21
    # Max relative difference: 0.00026576
    np.testing.assert_allclose(fft_john[0, 0, :], fft_me[0, 0, :])


def test_symmetry_point_interpolation():
    """Does the interpolation contain the same points at x=0 and x=L"""
    raw_points = load_raw_data()
    interpolation = load_john_interpolation()
    points = np.array(interpolation["points"])

    try:
        np.testing.assert_array_equal(points[0, :, :], points[-1, :, :])
    except AssertionError:
        print("Endpoint are not the same")
    else:
        print("Endpoint the same")

    delta_x = 2 * (np.max(raw_points["x_points"]) - np.min(raw_points["x_points"]))
    # These are calculated assuming no symmetry point!
    delta_x_john = interpolation["delta_x0"][0]

    delta_y = 2 * (np.max(raw_points["y_points"]) - np.min(raw_points["y_points"]))
    delta_y_john = interpolation["delta_x1"][1]
    # True - we have excluded the symmetry points properly!
    print(np.allclose([delta_y, delta_x], [delta_x_john, delta_y_john]))
