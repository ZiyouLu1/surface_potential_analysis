import numpy as np

from surface_potential_analysis.energy_data_plot import (
    compare_energy_grid_to_all_raw_points,
    plot_all_energy_points_z,
    plot_energy_grid_3D_xy,
    plot_energy_grid_3D_xz,
    plot_energy_point_locations_on_grid,
    plot_energy_points_location,
)

from .potential import load_john_interpolation, load_raw_data
from .surface_data import save_figure


def plot_raw_data_points():
    data = load_raw_data()
    fig, _, _ = plot_energy_points_location(data)
    fig.show()
    save_figure(fig, "nickel_raw_points.png")

    fig, ax = plot_all_energy_points_z(data)
    ax.set_ylim(0, 3 * 10**-19)

    fig.legend()
    fig.show()
    save_figure(fig, "nickel_raw_points_z.png")
    input()


def plot_john_interpolated_points():
    data = load_john_interpolation()

    fig, ax, _anim1 = plot_energy_grid_3D_xy(data)
    ax.set_title(
        "Plot of the interpolated Copper surface potential\n" "through the z plane"
    )
    fig.show()

    fig, ax, _anim2 = plot_energy_grid_3D_xz(data)
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
