from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.axis.plot import plot_explicit_basis_states_x
from surface_potential_analysis.basis.plot import (
    plot_fundamental_x_in_plane_projected_2d,
)
from surface_potential_analysis.basis.sho_basis import (
    infinate_sho_axis_3d_from_config,
)
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.potential.plot import (
    animate_potential_x0x1,
    plot_potential_1d_x2_comparison_111,
    plot_potential_minimum_along_path,
)
from surface_potential_analysis.potential.plot_point_potential import (
    get_point_potential_xy_locations,
    plot_point_potential_all_z,
    plot_point_potential_location_xy,
)
from surface_potential_analysis.potential.plot_uneven_potential import (
    plot_uneven_potential_z_comparison_111,
)
from surface_potential_analysis.potential.potential import (
    mock_even_potential,
    normalize_potential,
    truncate_potential,
)

from .s1_potential import (
    get_interpolated_potential,
    get_reflected_potential,
    load_raw_data,
)
from .surface_data import save_figure


def plot_raw_data_points() -> None:
    data = load_raw_data()
    fig, ax, _ = plot_point_potential_location_xy(data)

    locations = get_point_potential_xy_locations(data)
    e_min: list[float] = []
    for x, y in locations.T:
        idx = np.argwhere(
            np.logical_and(
                np.array(data["x_points"]) == x,
                np.array(data["y_points"]) == y,
            )
        )
        e_min.append(np.min(np.array(data["points"])[idx]))

    amin = np.argsort(e_min)
    x_min = locations[0][amin[0]]
    y_min = locations[1][amin[0]]
    ax.text(x_min, y_min, "FCC (lowest E)")

    x_min = locations[0][amin[1]]
    y_min = locations[1][amin[1]]
    ax.text(x_min, y_min, "HCP (second lowest E)")

    for i in range(2, 9):
        x_min = locations[0][amin[i]]
        y_min = locations[1][amin[i]]
        ax.text(x_min, y_min, f"({i})")

    x_min = locations[0][amin[-1]]
    y_min = locations[1][amin[-1]]
    ax.text(x_min, y_min, "Top (largest E)")

    fig.show()
    save_figure(fig, "nickel_raw_points.png")

    fig, ax = plot_point_potential_all_z(data)
    ax.set_ylim(0, 3 * 10**-19)

    ax.legend()
    fig.show()
    save_figure(fig, "nickel_raw_points_z.png")

    input()


def plot_raw_potential_points() -> None:
    potential = normalize_potential(get_reflected_potential())
    mocked = mock_even_potential(potential)

    fig, _, _ = plot_fundamental_x_in_plane_projected_2d(mocked["basis"], (0, 1), (0,))
    fig.show()

    fig, ax, _ = plot_potential_1d_x2_comparison_111(mocked)
    fig.show()

    fig, _, _ani = animate_potential_x0x1(mocked)
    fig.show()

    truncated = truncate_potential(potential, cutoff=2e-19, n=1, offset=1e-20)
    fig, ax, _ = plot_uneven_potential_z_comparison_111(truncated)
    plot_potential_1d_x2_comparison_111(mocked, ax=ax)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    input()


def plot_interpolated_energy_grid_points() -> None:
    potential = get_interpolated_potential((70, 70, 100))

    fig, ax, _ = plot_fundamental_x_in_plane_projected_2d(
        potential["basis"], (0, 1), (0,)
    )
    fig.show()

    raw = normalize_potential(get_reflected_potential())
    fig, ax, _ = plot_uneven_potential_z_comparison_111(raw)
    plot_potential_1d_x2_comparison_111(potential, ax=ax)
    fig.show()

    fig, ax, _ani = animate_potential_x0x1(potential, clim=(0, 0.2e-18))
    z_energies = np.min(
        potential["vector"].reshape(AxisWithLengthBasisUtil(potential["basis"]).shape),
        axis=2,
    )
    xy_min = np.unravel_index(np.argmin(z_energies), z_energies.shape)
    x0_min = xy_min[0] / (1 + z_energies.shape[0])
    x1_min = xy_min[1] / (1 + z_energies.shape[1])
    (line,) = ax.plot(
        x0_min * potential["basis"][0].delta_x[0]
        + x1_min * potential["basis"][1].delta_x[0],
        x0_min * potential["basis"][0].delta_x[1]
        + x1_min * potential["basis"][1].delta_x[1],
    )
    line.set_marker("x")

    fig.show()
    input()


def calculate_raw_fcc_hcp_energy_jump() -> None:
    raw_points = load_raw_data()

    x_points = np.array(raw_points["x_points"])
    y_points = np.array(raw_points["y_points"])
    points = np.array(raw_points["points"])

    p1 = points[(x_points == np.max(x_points))]
    p2 = points[np.logical_and(x_points == 0, y_points == np.max(y_points))]
    p3 = points[np.logical_and(x_points == 0, y_points == 0)]

    print(np.min(p1))  # 0.0  # noqa: T201
    print(np.min(p2))  # 2.95E-21J # noqa: T201
    print(np.min(p3))  # 9.67E-20J # noqa: T201


def plot_interpolation_with_sho_wavefunctions() -> None:
    """
    Investigate the extent to which SHO wavefunctions lie outside the potential.

    Is is possible that the SHO wavefunctions lie outside the interpolated potential
    or have energies for which they can see the truncation process.

    Plotting them alongside the interpolation in the hZ direction will allow us to
    diagnose these issues
    """
    potential = get_interpolated_potential((70, 70, 100))
    fig, ax = plt.subplots()
    plot_potential_1d_x2_comparison_111(potential, ax=ax)
    plot_explicit_basis_states_x(
        infinate_sho_axis_3d_from_config(
            potential["basis"][2],
            {
                "mass": 1.6735575e-27,
                "sho_omega": 179704637926161.6,
                "x_origin": np.array([0, 0, -9.848484848484871e-11]),
            },
            16,
        ),
        ax=ax,
    )

    ax.set_ylim(0, 0.5e-18)
    fig.show()

    save_figure(fig, "sho_wavefunctions_alongside_potential.png")
    input()


def plot_potential_minimum_along_diagonal() -> None:
    fig, ax = plt.subplots()

    interpolation = get_interpolated_potential((70, 70, 100))
    shape = AxisWithLengthBasisUtil(interpolation["basis"]).shape
    path = np.array([(x, x) for x in range(shape[0])]).T
    _, _, _ = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    fig.show()
    save_figure(fig, "classical_trajectory_comparison.png")

    input()


def plot_potential_minimum_along_edge() -> None:
    interpolation = get_interpolated_potential((70, 70, 100))
    fig, ax = plt.subplots()
    shape = AxisWithLengthBasisUtil(interpolation["basis"]).shape
    path = np.array([(shape[0] - (x), x) for x in range(shape[0])]).T
    # Add a fake point here so they line up. path[0] is not included in the unit cell
    path[:, 0] = path[:, 2]
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("diagonal")

    path = np.array([(x, 0) for x in range(shape[0])]).T
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("x1=0")

    path = np.array([(0, y) for y in range(shape[1])]).T
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("x0=0")

    ax.legend()
    fig.show()
    ax.set_title(
        "plot of the potential along the edge and off diagonal."
        "\nAll three should be identical"
    )
    input()
