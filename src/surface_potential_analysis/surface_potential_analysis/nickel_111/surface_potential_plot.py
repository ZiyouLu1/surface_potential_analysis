from ..energy_data.energy_data_plot import (
    compare_energy_grid_to_all_raw_points,
    plot_all_energy_points_z,
    plot_energy_grid_3D_xy,
    plot_energy_grid_3D_xz,
    plot_energy_point_locations_on_grid,
    plot_energy_points_location,
)
from .surface_data import save_figure
from .surface_potential import load_john_interpolation, load_raw_data


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

    fig, ax, _anim2 = plot_energy_point_locations_on_grid(raw_points, interpolation)
    fig.show()

    fig, ax = compare_energy_grid_to_all_raw_points(raw_points, interpolation)
    ax.set_ylim(0, 3 * 10**-19)
    ax.set_title("Comparison between raw and interpolated potential for Nickel")
    fig.show()
    input()
    save_figure(fig, "raw_interpolation_comparison.png")
