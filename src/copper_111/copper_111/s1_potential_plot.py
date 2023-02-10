import numpy as np

from surface_potential_analysis.energy_data import EnergyGrid, normalize_energy
from surface_potential_analysis.energy_data_plot import (
    animate_energy_grid_3D_in_x1z,
    animate_energy_grid_3D_in_xy,
    compare_energy_grid_to_all_raw_points,
    plot_all_energy_points_z,
    plot_energy_grid_points,
    plot_energy_point_locations_on_grid,
    plot_energy_points_location,
    plot_z_direction_energy_comparison_111,
    plot_z_direction_energy_data_111,
)

from .s1_potential import (
    load_cleaned_data_grid,
    load_interpolated_grid,
    load_john_interpolation,
    load_raw_data,
    load_raw_data_grid,
)
from .surface_data import save_figure


def plot_raw_data_points():
    data = load_raw_data()
    fig, ax, _ = plot_energy_points_location(data)

    amin = np.argmin(data["points"])
    x_min = data["x_points"][amin]
    y_min = data["y_points"][amin]
    ax.text(x_min, y_min, "hcp (lowest E)")

    fig.show()
    save_figure(fig, "nickel_raw_points.png")

    fig, ax = plot_all_energy_points_z(data)
    ax.set_ylim(0, 3 * 10**-19)

    ax.legend()
    fig.show()
    save_figure(fig, "nickel_raw_points_z.png")

    input()


def plot_raw_energy_grid_points():
    grid = normalize_energy(load_raw_data_grid())

    fig, _, _ = plot_energy_grid_points(grid)
    fig.show()

    fig, ax, _ = plot_z_direction_energy_data_111(grid)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    fig, _, _ani = animate_energy_grid_3D_in_xy(grid)
    fig.show()

    cleaned = load_cleaned_data_grid()
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

    raw = load_cleaned_data_grid()
    fig, ax = plot_z_direction_energy_comparison_111(grid, raw)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    fig, ax, _ani = animate_energy_grid_3D_in_xy(grid)
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
    ax.set_title("Plot of the ft of the interpolated potential")
    fig.show()
    input()


def plot_john_interpolated_points():
    data = load_john_interpolation()

    fig, ax, _anim1 = animate_energy_grid_3D_in_xy(data)
    ax.set_title(
        "Plot of the interpolated Copper surface potential\n" "through the z plane"
    )
    fig.show()

    fig, ax, _anim2 = animate_energy_grid_3D_in_x1z(data)
    ax.set_title(
        "Plot of the interpolated Copper surface potential\n" "through the y plane"
    )

    fig.show()
    input()


def compare_john_interpolation():
    raw_points = load_raw_data()
    interpolation = load_john_interpolation()

    fig, ax, _anim2 = plot_energy_point_locations_on_grid(raw_points, interpolation)
    fig.show()

    fig, ax = compare_energy_grid_to_all_raw_points(raw_points, interpolation)
    ax.set_ylim(0, 3 * 10**-19)
    ax.set_title("Comparison between raw and interpolated potential for Nickel")
    fig.show()
    input()
    save_figure(fig, "raw_interpolation_comparison.png")


def calculate_raw_fcc_hcp_energy_jump():
    raw_points = load_raw_data()

    x_points = np.array(raw_points["x_points"])
    y_points = np.array(raw_points["y_points"])
    points = np.array(raw_points["points"])

    p1 = points[(x_points == np.max(x_points))]
    p2 = points[np.logical_and(x_points == 0, y_points == np.max(y_points))]
    p3 = points[np.logical_and(x_points == 0, y_points == 0)]

    print(np.min(p1))  # 0.0
    print(np.min(p2))  # 2.95E-21J
    print(np.min(p3))  # 9.67E-20J
