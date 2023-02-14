import math
from typing import List, Tuple

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
    plot_energy_in_z_direction,
    plot_energy_point_locations_on_grid,
    plot_energy_points_location,
    plot_z_direction_energy_comparison_111,
    plot_z_direction_energy_data_111,
)

from .s1_potential import (
    load_cleaned_energy_grid,
    load_interpolated_grid,
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
    lines: List[Line2D] = []
    for (label, xy_ind) in locations.items():
        _, _, line = plot_energy_in_z_direction(grid, xy_ind, ax=ax)
        line.set_label(label)
        lines.append(line)

    ax.set_title("Plot of energy at the Top and Hollow sites")
    ax.set_ylabel("Energy / J")
    ax.set_xlabel("relative z position /m")

    ax.legend()

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
    input()


def compare_john_interpolation():
    raw_points = load_raw_data()
    interpolation = load_john_interpolation()
    print(raw_points["y_points"])
    print(raw_points["x_points"])

    fig, ax, _anim2 = plot_energy_point_locations_on_grid(raw_points, interpolation)
    fig.show()

    fig, ax = compare_energy_grid_to_all_raw_points(raw_points, interpolation)
    ax.set_ylim(0, 3 * 10**-19)
    ax.set_title("Comparison between raw and interpolated potential for Nickel")
    fig.show()
    input()
    save_figure(fig, "raw_interpolation_comparison.png")


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
