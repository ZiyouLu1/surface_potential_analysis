import matplotlib.pyplot as plt
import numpy as np
import scipy.constants

from surface_potential_analysis.energy_data import (
    load_energy_grid_legacy,
    normalize_energy,
)
from surface_potential_analysis.energy_data_plot import (
    plot_xz_plane_energy,
    plot_z_direction_energy_comparison_100,
    plot_z_direction_energy_data_100,
)

from .potential import (
    load_9h_copper_data,
    load_interpolated_copper_data,
    load_interpolated_relaxed_data,
    load_nc_raw_copper_data,
    load_raw_copper_data,
    load_relaxed_copper_data,
    load_spline_interpolated_relaxed_data,
)
from .surface_data import get_data_path, save_figure


def plot_copper_raw_data():
    data = load_raw_copper_data()
    data = normalize_energy(data)

    fig, ax, _ = plot_z_direction_energy_data_100(data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "copper_raw_data_z_direction.png")

    plot_xz_plane_energy(data)


def plot_copper_nc_data():
    data = normalize_energy(load_nc_raw_copper_data())

    fig, ax, _ = plot_z_direction_energy_data_100(data)
    ax.set_ylim(bottom=-0.1e-18, top=1e-18)
    fig.show()
    input()
    save_figure(fig, "copper_raw_data_z_direction_nc.png")


def plot_copper_9h_data():
    data = normalize_energy(load_9h_copper_data())

    data_7h = load_raw_copper_data()
    data_7h_norm = normalize_energy(data_7h)

    fig, ax, _ = plot_z_direction_energy_data_100(data)
    _, _, _ = plot_z_direction_energy_data_100(data_7h_norm, ax=ax)
    ax.set_ylim(bottom=-0.1e-18, top=1e-18)

    fig.show()
    input()
    save_figure(fig, "copper_raw_data_z_direction_9h.png")


def plot_copper_relaxed_data():

    data_7h = load_raw_copper_data()
    data_7h_norm = normalize_energy(data_7h)

    data_relaxed = load_relaxed_copper_data()
    data_relaxed_norm = normalize_energy(data_relaxed)

    fig, ax = plt.subplots()
    _, _, _ = plot_z_direction_energy_data_100(data_relaxed_norm, ax=ax)
    _, _, _ = plot_z_direction_energy_data_100(data_7h_norm, ax=ax)

    ax.set_ylim(bottom=-0.1e-18, top=1e-18)

    fig.show()
    input()
    save_figure(fig, "copper_raw_data_z_direction_vs_relaxed.png")


def plot_copper_relaxed_interpolated_data():
    data = load_interpolated_relaxed_data()
    raw_data = normalize_energy(load_relaxed_copper_data())

    fig, ax = plot_z_direction_energy_comparison_100(data, raw_data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "relaxed_interpolated_data_comparison.png")

    fig = plot_xz_plane_energy(data)
    fig.show()
    save_figure(fig, "relaxed_interpolated_data_xy.png")

    spline_data = load_spline_interpolated_relaxed_data()
    raw_data = normalize_energy(load_relaxed_copper_data())

    fig, ax = plot_z_direction_energy_comparison_100(spline_data, raw_data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "relaxed_interpolated_data_comparison.png")

    fig = plot_xz_plane_energy(spline_data)
    fig.show()
    save_figure(fig, "relaxed_interpolated_data_xy.png")

    input()


def plot_copper_interpolated_data():
    data = load_interpolated_copper_data()
    raw_data = normalize_energy(load_raw_copper_data())

    fig, ax = plot_z_direction_energy_comparison_100(data, raw_data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "copper_interpolated_data_comparison.png")

    fig = plot_xz_plane_energy(data)
    fig.show()
    input()
    save_figure(fig, "copper_interpolated_data_xy.png")


def compare_bridge_hollow_energy():
    data = load_interpolated_copper_data()
    points = np.array(data["points"])
    print(points.shape)

    print(np.min(points[points.shape[0] // 2, 0, :]))
    print(np.min(points[points.shape[0] // 2, points.shape[1] // 2, :]))
    print(np.min(points[0, 0, :]))
    print(np.max(points[:, :, -1]))
    print(
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
    )

    data = normalize_energy(load_raw_copper_data())
    points = np.array(data["points"])
    print(points.shape)

    print(np.min(points[points.shape[0] // 2, 0, :]))
    print(np.min(points[points.shape[0] // 2, points.shape[1] // 2, :]))
    print(np.min(points[0, 0, :]))
    print(np.max(points[:, :, -1]))
    print(
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
    )


def calculate_hollow_free_energy_jump():
    path = get_data_path("copper_relaxed_raw_energies.json")
    data = load_energy_grid_legacy(path)

    points = np.array(data["points"], dtype=float)
    middle_x_index = points.shape[0] // 2
    middle_y_index = points.shape[1] // 2
    hollow_points = points[middle_x_index, middle_y_index]

    min_index = np.argmin(hollow_points)
    min_value = data["points"][middle_x_index][middle_y_index][min_index]

    max_index = np.argmax(hollow_points[hollow_points < 0])
    max_value = hollow_points[hollow_points < 0][max_index]

    print(
        min_index,
        f"{min_value} J",
        f"{min_value / scipy.constants.elementary_charge} eV",
    )

    print(
        max_index,
        f"{max_value} J",
        f"{max_value / scipy.constants.elementary_charge} eV",
    )
