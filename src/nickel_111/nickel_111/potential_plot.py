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
        "Plot of the interpolated Nickel surface potential\n" "through the x plane"
    )
    fig.show()

    fig, ax, _anim2 = plot_energy_grid_3D_xz(data)
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

    dx = interpolation["x_points"][-1] - interpolation["x_points"][0]
    delta_x = 2 * (np.max(raw_points["x_points"]) - np.min(raw_points["x_points"]))
    dx_required = (
        delta_x * (len(interpolation["x_points"]) - 1) / len(interpolation["x_points"])
    )

    dy = interpolation["y_points"][-1] - interpolation["y_points"][0]
    delta_y = 2 * (np.max(raw_points["y_points"]) - np.min(raw_points["y_points"]))
    dy_required = (
        delta_y * (len(interpolation["y_points"]) - 1) / len(interpolation["y_points"])
    )
    # True - we have excluded the symmetry points properly!
    print(np.allclose([dx_required, dy_required], [dx, dy]))
