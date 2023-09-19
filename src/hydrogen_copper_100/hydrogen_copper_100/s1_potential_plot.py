from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from surface_potential_analysis.axis.plot import plot_explicit_basis_states_x
from surface_potential_analysis.axis.util import BasisUtil
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x2_comparison_100,
    plot_potential_2d_x,
)
from surface_potential_analysis.potential.plot_uneven_potential import (
    plot_uneven_potential_z_comparison_100,
)
from surface_potential_analysis.potential.potential import (
    mock_even_potential,
    normalize_potential,
)
from surface_potential_analysis.stacked_basis.plot import (
    plot_fundamental_x_in_plane_projected_2d,
)
from surface_potential_analysis.stacked_basis.sho_basis import (
    infinate_sho_axis_3d_from_config,
)

from .s1_potential import (
    get_interpolated_potential,
    get_interpolated_potential_relaxed,
    load_9h_copper_potential,
    load_nc_raw_copper_potential,
    load_raw_copper_potential,
    load_relaxed_copper_potential,
)
from .surface_data import save_figure


def plot_copper_raw_data() -> None:
    data = load_raw_copper_potential()
    data = normalize_potential(data)

    fig, ax, _ = plot_uneven_potential_z_comparison_100(data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "copper_raw_data_z_direction.png")

    fig, _, _ = plot_potential_2d_x(mock_even_potential(data))
    fig.show()
    input()


def plot_copper_nc_data() -> None:
    data = normalize_potential(load_nc_raw_copper_potential())

    fig, ax, _ = plot_uneven_potential_z_comparison_100(data)
    ax.set_ylim(bottom=-0.1e-18, top=1e-18)
    fig.show()
    input()
    save_figure(fig, "copper_raw_data_z_direction_nc.png")


def plot_copper_9h_data() -> None:
    data = normalize_potential(load_9h_copper_potential())

    data_7h = load_raw_copper_potential()
    data_7h_norm = normalize_potential(data_7h)

    fig, ax, _ = plot_uneven_potential_z_comparison_100(data)
    _, _, _ = plot_uneven_potential_z_comparison_100(data_7h_norm, ax=ax)
    ax.set_ylim(bottom=-0.1e-18, top=1e-18)

    fig.show()
    input()
    save_figure(fig, "copper_raw_data_z_direction_9h.png")


def plot_copper_relaxed_data() -> None:
    data_7h = load_raw_copper_potential()
    data_7h_norm = normalize_potential(data_7h)

    data_relaxed = load_relaxed_copper_potential()
    data_relaxed_norm = normalize_potential(data_relaxed)

    fig, ax = plt.subplots()
    plot_uneven_potential_z_comparison_100(data_relaxed_norm, ax=ax)
    plot_uneven_potential_z_comparison_100(data_7h_norm, ax=ax)

    ax.set_ylim(bottom=-0.1e-18, top=1e-18)

    fig.show()
    input()
    save_figure(fig, "copper_raw_data_z_direction_vs_relaxed.png")


def plot_copper_relaxed_interpolated_data() -> None:
    data = get_interpolated_potential_relaxed((50, 50, 250))
    raw_data = normalize_potential(load_relaxed_copper_potential())

    fig, ax, _ = plot_potential_1d_x2_comparison_100(data)
    plot_uneven_potential_z_comparison_100(raw_data, ax=ax)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "relaxed_interpolated_data_comparison.png")

    fig, ax, _ = plot_potential_2d_x(data)
    fig.show()
    save_figure(fig, "relaxed_interpolated_data_xy.png")

    fig, ax, _ani0 = plot_potential_2d_x(data)
    plot_fundamental_x_in_plane_projected_2d(
        mock_even_potential(raw_data)["basis"], (0, 1), (0,), ax=ax
    )
    fig.show()

    raw_data = normalize_potential(load_relaxed_copper_potential())

    fig, ax, _ani1 = plot_potential_2d_x(data)
    plot_fundamental_x_in_plane_projected_2d(
        mock_even_potential(raw_data)["basis"], (0, 1), (0,), ax=ax
    )
    fig.show()

    input()


def plot_copper_interpolated_data() -> None:
    data = get_interpolated_potential((50, 50, 100))

    raw_data = normalize_potential(load_raw_copper_potential())

    fig, ax, _ = plot_potential_1d_x2_comparison_100(data)
    plot_uneven_potential_z_comparison_100(raw_data, ax=ax)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "copper_interpolated_data_comparison.png")

    fig, ax, _ = plot_potential_2d_x(data)
    fig.show()
    input()
    save_figure(fig, "copper_interpolated_data_xy.png")


def plot_interpolation_with_sho_wavefunctions() -> None:
    # Is is possible that the SHO wavefunctions lie outside the interpolated potential
    # or have energies for which they can see the truncation process.

    # Plotting them alongside the interpolation in the hZ direction will allow us to
    # diagnose these issues

    potential = get_interpolated_potential((50, 50, 100))
    fig, ax = plt.subplots()
    plot_potential_1d_x2_comparison_100(potential, ax=ax)
    plot_explicit_basis_states_x(
        infinate_sho_axis_3d_from_config(
            potential["basis"][2],
            {
                "mass": 1.6735575e-27,
                "sho_omega": 117905964225836.06,
                "x_origin": np.array([0, 0, -1.840551985155284e-10]),
            },
            16,
        ),
        ax=ax,
    )

    ax.set_ylim(0, 0.5e-18)
    fig.show()

    save_figure(fig, "sho_wavefunctions_alongside_potential.png")
    input()


def compare_bridge_hollow_energy() -> None:
    print("--------------------------------------")  # noqa: T201
    print("Non-relaxed")  # noqa: T201
    data = get_interpolated_potential((50, 50, 100))
    points = np.array(data["data"])
    print(points.shape)  # noqa: T201

    print("Bridge ", np.min(points[points.shape[0] // 2, 0, :]))  # noqa: T201
    print(  # noqa: T201
        "Hollow ", np.min(points[points.shape[0] // 2, points.shape[1] // 2, :])
    )
    print("Top ", np.min(points[0, 0, :]))  # noqa: T201
    print("Free ", np.max(points[:, :, -1]))  # noqa: T201
    print(  # noqa: T201
        "Max free E variation",
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
        * 100,
        "%",
    )

    raw_data = normalize_potential(load_raw_copper_potential())
    util = BasisUtil(raw_data["basis"])
    points = np.array(raw_data["data"]).reshape(util.shape)

    print(points.shape)  # noqa: T201

    print("Bridge ", np.min(points[points.shape[0] // 2, 0, :]))  # noqa: T201
    print(  # noqa: T201
        "Hollow ", np.min(points[points.shape[0] // 2, points.shape[1] // 2, :])
    )
    print("Top ", np.min(points[0, 0, :]))  # noqa: T201
    print("Free ", np.max(points[:, :, -1]))  # noqa: T201
    print(  # noqa: T201
        "Max free E variation",
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
        * 100,
        "%",
    )
    print("--------------------------------------")  # noqa: T201

    print("--------------------------------------")  # noqa: T201
    print("Relaxed")  # noqa: T201
    data = get_interpolated_potential_relaxed((50, 50, 100))
    util2 = BasisUtil(data["basis"])
    points = np.array(data["data"]).reshape(util2.shape)
    print(points.shape)  # noqa: T201

    print("Bridge ", np.min(points[points.shape[0] // 2, 0, :]))  # noqa: T201
    print(  # noqa: T201
        "Hollow ", np.min(points[points.shape[0] // 2, points.shape[1] // 2, :])
    )
    print("Top ", np.min(points[0, 0, :]))  # noqa: T201
    print("Free ", np.max(points[:, :, -1]))  # noqa: T201
    print(  # noqa: T201
        "Max free E variation",
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
        * 100,
        "%",
    )

    relaxed_data = normalize_potential(load_relaxed_copper_potential())

    util3 = BasisUtil(relaxed_data["basis"][0:2])
    points = np.array(relaxed_data["data"]).reshape(util3.shape)
    print(points.shape)  # noqa: T201

    print("Bridge ", np.min(points[points.shape[0] // 2, 0, :]))  # noqa: T201
    print(  # noqa: T201
        "Hollow ", np.min(points[points.shape[0] // 2, points.shape[1] // 2, :])
    )
    print("Top ", np.min(points[0, 0, :]))  # noqa: T201
    print("Free ", np.max(points[:, :, -1]))  # noqa: T201
    print(  # noqa: T201
        "Max free E variation",
        np.max(
            np.abs(points[:, :, -1] - np.max(points[:, :, -1]))
            / np.max(points[:, :, -1])
        )
        * 100,
        "%",
    )
    print("--------------------------------------")  # noqa: T201


def calculate_hollow_free_energy_jump() -> None:
    data = load_relaxed_copper_potential()
    util = BasisUtil(data["basis"])

    points = data["data"].reshape(util.shape)
    middle_x_index = points.shape[0] // 2
    middle_y_index = points.shape[1] // 2
    hollow_points = points[middle_x_index, middle_y_index]

    min_index = np.argmin(hollow_points)
    min_value = points[middle_x_index][middle_y_index][min_index]

    max_index = np.argmax(hollow_points[hollow_points < 0])
    max_value = hollow_points[hollow_points < 0][max_index]

    print(  # noqa: T201
        min_index,
        f"{min_value} J",
        f"{min_value / scipy.constants.elementary_charge} eV",
    )

    print(  # noqa: T201
        max_index,
        f"{max_value} J",
        f"{max_value / scipy.constants.elementary_charge} eV",
    )
