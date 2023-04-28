from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants

from .surface_data import get_data_path, save_figure


def plot_copper_raw_data():
    data = load_raw_copper_potential()
    data = normalize_potential(data)

    fig, ax, _ = plot_z_direction_energy_data_100(data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "copper_raw_data_z_direction.png")

    plot_xz_plane_energy_copper_100(data)


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

    fig = plot_xz_plane_energy_copper_100(data)
    fig.show()
    save_figure(fig, "relaxed_interpolated_data_xy.png")

    fig, ax, _ani0 = animate_energy_grid_3d_in_xy(data)
    plot_energy_grid_locations(raw_data, ax=ax)
    fig.show()

    spline_data = load_spline_interpolated_relaxed_data()
    raw_data = normalize_energy(load_relaxed_copper_data())

    fig, ax = plot_z_direction_energy_comparison_100(spline_data, raw_data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "relaxed_interpolated_data_comparison.png")

    fig = plot_xz_plane_energy_copper_100(spline_data)
    fig.show()
    save_figure(fig, "relaxed_interpolated_data_xy.png")

    fig, ax, _ani1 = animate_energy_grid_3d_in_xy(data)
    plot_energy_grid_locations(raw_data, ax=ax)
    fig.show()

    input()


def plot_copper_interpolated_data():
    path = get_data_path("copper_interpolated_energies.json")
    data = load_energy_grid(path)

    raw_data = normalize_energy(load_raw_copper_data())

    fig, ax = plot_z_direction_energy_comparison_100(data, raw_data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "copper_interpolated_data_comparison.png")

    fig = plot_xz_plane_energy_copper_100(data)
    fig.show()
    input()
    save_figure(fig, "copper_interpolated_data_xy.png")


def plot_interpolation_with_sho_wavefunctions():
    """
    Is is possible that the SHO wavefunctions lie outside the interpolated potential
    or have energies for which they can see the truncation process.

    Plotting them alongside the interpolation in the hZ direction will allow us to
    diagnose these issues
    """
    grid = load_interpolated_copper_data()
    fig, ax = plt.subplots()
    plot_z_direction_energy_data_100(grid, ax=ax)
    plot_sho_wavefunctions(
        grid["z_points"],
        sho_omega=117905964225836.06,
        mass=1.6735575e-27,
        first_n=16,
        ax=ax,
    )
    ax.set_ylim(0, 0.5e-18)
    fig.show()

    save_figure(fig, "sho_wavefunctions_alongside_potential.png")
    input()


def compare_bridge_hollow_energy():
    print("--------------------------------------")
    print("Non-relaxed")
    data = load_interpolated_copper_data()
    points = np.array(data["points"])
    print(points.shape)

    print("Bridge ", np.min(points[points.shape[0] // 2, 0, :]))
    print("Hollow ", np.min(points[points.shape[0] // 2, points.shape[1] // 2, :]))
    print("Top ", np.min(points[0, 0, :]))
    print("Free ", np.max(points[:, :, -1]))
    print(
        "Max free E variation",
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
        * 100,
        "%",
    )

    data = normalize_energy(load_raw_copper_data())
    points = np.array(data["points"])
    print(points.shape)

    print("Bridge ", np.min(points[points.shape[0] // 2, 0, :]))
    print("Hollow ", np.min(points[points.shape[0] // 2, points.shape[1] // 2, :]))
    print("Top ", np.min(points[0, 0, :]))
    print("Free ", np.max(points[:, :, -1]))
    print(
        "Max free E variation",
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
        * 100,
        "%",
    )
    print("--------------------------------------")

    print("--------------------------------------")
    print("Relaxed")
    data = load_interpolated_relaxed_data()
    points = np.array(data["points"])
    print(points.shape)

    print("Bridge ", np.min(points[points.shape[0] // 2, 0, :]))
    print("Hollow ", np.min(points[points.shape[0] // 2, points.shape[1] // 2, :]))
    print("Top ", np.min(points[0, 0, :]))
    print("Free ", np.max(points[:, :, -1]))
    print(
        "Max free E variation",
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
        * 100,
        "%",
    )

    data = normalize_energy(load_relaxed_copper_data())
    points = np.array(data["points"])
    print(points.shape)

    print("Bridge ", np.min(points[points.shape[0] // 2, 0, :]))
    print("Hollow ", np.min(points[points.shape[0] // 2, points.shape[1] // 2, :]))
    print("Top ", np.min(points[0, 0, :]))
    print("Free ", np.max(points[:, :, -1]))
    print(
        "Max free E variation",
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
        * 100,
        "%",
    )
    print("--------------------------------------")


def calculate_hollow_free_energy_jump():
    path = get_data_path("copper_relaxed_raw_energies.json")
    data = load_energy_grid(path)

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
