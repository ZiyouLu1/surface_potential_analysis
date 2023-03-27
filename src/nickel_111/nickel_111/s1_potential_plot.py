import math
from typing import Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from surface_potential_analysis.basis_config_plot import plot_projected_coordinates_2D
from surface_potential_analysis.potential import (
    Potential,
    UnevenPotential,
    normalize_potential,
)
from surface_potential_analysis.potential.plot_point_potential import (
    plot_point_potential_all_z,
    plot_point_potential_location_xy,
)
from surface_potential_analysis.potential.plot_potential import (
    animate_potential_x0x1,
    plot_potential_1D_comparison,
    plot_potential_minimum_along_path,
    plot_potential_x0x1,
)
from surface_potential_analysis.potential.plot_uneven_potential import (
    plot_uneven_potential_z_comparison,
)
from surface_potential_analysis.potential.potential import mock_even_potential

from .s1_potential import (
    load_cleaned_energy_grid,
    load_interpolated_grid,
    load_interpolated_john_grid,
    load_interpolated_reciprocal_grid,
    load_john_interpolation,
    load_raw_data,
    load_raw_data_reciprocal_grid,
)
from .surface_data import save_figure


def get_nickel_comparison_points_x0x1(
    potential: Potential[Any, Any, Any]
) -> dict[str, tuple[tuple[int, int], int]]:
    points = potential["points"]
    return {
        "Top Site": (
            (math.floor(2 * points.shape[0] / 3), math.floor(2 * points.shape[1] / 3)),
            2,
        ),
        "Bridge Site": (
            (math.floor(points.shape[0] / 6), math.floor(points.shape[1] / 6)),
            2,
        ),
        "FCC Site": (
            (0, 0),
            2,
        ),
        "HCP Site": (
            (math.floor(points.shape[0] / 3), math.floor(points.shape[1] / 3)),
            2,
        ),
    }


def get_nickel_reciprocal_comparison_points_x0x1(
    potential: Potential[Any, Any, Any]
) -> dict[str, tuple[tuple[int, int], int]]:
    points = potential["points"]
    return {
        "HCP Site": ((math.floor(points.shape[0] / 3), 0), 2),
        "Bridge Site": ((math.floor(points.shape[0] / 6), 0), 2),
        "Top Site": ((0, math.floor(points.shape[0] / 3)), 2),
        "FCC Site": ((0, 0), 2),
    }


def plot_raw_data_points() -> None:
    data = load_raw_data()
    fig, _, _ = plot_point_potential_location_xy(data)
    fig.show()
    save_figure(fig, "nickel_raw_points.png")

    fig, ax = plot_point_potential_all_z(data)
    ax.set_ylim(0, 3 * 10**-19)

    ax.legend()
    fig.show()
    save_figure(fig, "nickel_raw_points_z.png")
    input()


def plot_z_direction_energy_data_nickel_reciprocal_points(
    potential: UnevenPotential[Any, Any, Any], *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    mocked = mock_even_potential(potential)
    locations = get_nickel_reciprocal_comparison_points_x0x1(mocked)
    locations_uneven = {k: v[0] for (k, v) in locations.items()}
    fig, ax = plot_uneven_potential_z_comparison(potential, locations_uneven, ax=ax)

    return fig, ax


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def plot_raw_energy_grid_points() -> None:
    potential = normalize_potential(load_raw_data_reciprocal_grid())
    mocked_potential = mock_even_potential(potential)

    fig, _, _ = plot_projected_coordinates_2D(mocked_potential["basis"], 0, 2)
    fig.show()

    fig, ax = plot_z_direction_energy_data_nickel_reciprocal_points(potential)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    fig, _, _ani = animate_potential_x0x1(mocked_potential)
    fig.show()

    cleaned = load_cleaned_energy_grid()
    mocked = mock_even_potential(cleaned)
    locations = get_nickel_comparison_points_x0x1(mocked)
    locations_uneven = {k: v[0] for (k, v) in locations.items()}
    fig, ax = plot_uneven_potential_z_comparison(cleaned, locations_uneven)
    for ln in ax.lines:
        ln.set_marker("x")
        ln.set_linestyle("")
    locations = get_nickel_comparison_points_x0x1(mocked_potential)
    locations_uneven = {k: v[0] for (k, v) in locations.items()}
    plot_uneven_potential_z_comparison(potential, locations_uneven)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    # ft_points = np.abs(np.fft.ifft2(grid["points"], axes=(0, 1)))
    # ft_points[0, 0] = 0
    # ft_grid: EnergyGrid = {
    #     "delta_x0": grid["delta_x0"],
    #     "delta_x1": grid["delta_x1"],
    #     "points": ft_points.tolist(),
    #     "z_points": grid["z_points"],
    # }
    # fig, ax, _ani1 = animate_energy_grid_3D_in_xy(ft_grid)
    # # TODO: delta is wrong, plot generic points and then factor out into ft.
    # ax.set_title("Plot of the ft of the raw potential")
    # fig.show()
    input()


def plot_interpolated_energy_grid_points() -> None:
    potential = load_interpolated_grid()

    fig, ax, _ = plot_projected_coordinates_2D(potential["basis"], 0, 2)
    fig.show()

    fig, ax, _ani = animate_potential_x0x1(potential)
    # TODO: clim_max=1e-18

    raw_potential = load_raw_data_reciprocal_grid()
    mocked_raw_potential = mock_even_potential(raw_potential)
    plot_projected_coordinates_2D(mocked_raw_potential["basis"], 0, 2, ax=ax)
    fig.show()

    locations = get_nickel_comparison_points_x0x1(potential)
    raw_grid = normalize_potential(load_raw_data_reciprocal_grid())
    fig, ax = plot_z_direction_energy_data_nickel_reciprocal_points(raw_grid, ax=ax)
    for ln in ax.lines:
        ln.set_marker("x")
        ln.set_linestyle("")

    _, _ = plot_potential_1D_comparison(potential, locations)
    ax.set_ylim(0, 5e-18)

    fig.show()

    # ft_points = np.abs(np.fft.ifft2(grid["points"], axes=(0, 1)))
    # ft_points[0, 0] = 0
    # ft_grid: EnergyGrid = {
    #     "delta_x0": grid["delta_x0"],
    #     "delta_x1": grid["delta_x1"],
    #     "points": ft_points.tolist(),
    #     "z_points": grid["z_points"],
    # }
    # fig, ax, _ani1 = animate_energy_grid_3D_in_xy(ft_grid)
    # TODO: delta is wrong, plot generic points and then factor out into ft.
    ax.set_title("Plot of the ft of the interpolated potential")
    fig.show()
    input()


def plot_interpolated_energy_grid_reciprocal() -> None:
    potential = load_interpolated_reciprocal_grid()
    points = potential["points"]
    points[points < 0] = np.max(points)
    potential["points"] = points

    fig, ax, _ = plot_projected_coordinates_2D(potential["basis"], 0, 2)
    fig.show()

    fig, ax, _ani = animate_potential_x0x1(potential)

    raw_grid = load_raw_data_reciprocal_grid()
    mocked_raw_grid = mock_even_potential(raw_grid)
    plot_projected_coordinates_2D(mocked_raw_grid["basis"], 0, 2, ax=ax)
    fig.show()

    comparison_points = get_nickel_comparison_points_x0x1(potential)
    fig, ax = plot_potential_1D_comparison(potential, comparison_points)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    # ft_points = np.abs(np.fft.ifft2(grid["points"], axes=(0, 1)))
    # ft_points[0, 0] = 0
    # ft_grid: EnergyGrid = {
    #     "delta_x0": grid["delta_x0"],
    #     "delta_x1": grid["delta_x1"],
    #     "points": ft_points.tolist(),
    #     "z_points": grid["z_points"],
    # }
    # fig, ax, _ani1 = animate_energy_grid_3D_in_xy(ft_grid)
    # # TODO: delta is wrong, plot generic points and then factor out into ft.
    # ax.set_title("Plot of the ft of the interpolated potential")
    # fig.show()
    input()


def get_john_point_locations(
    grid: UnevenPotential[Any, Any, Any]
) -> dict[str, tuple[int, int]]:
    points = np.array(grid["points"], dtype=float)
    return {
        "Top Site": (0, 0),
        "Bridge Site": (0, math.floor(points.shape[1] / 2)),
        "FCC Site": (0, math.floor(points.shape[1] / 3)),
        "HCP Site": (0, math.floor(2 * points.shape[1] / 3)),
    }


def plot_z_direction_energy_data_john(
    grid: UnevenPotential[Any, Any, Any], *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    locations = get_john_point_locations(grid)
    return plot_uneven_potential_z_comparison(grid, locations, ax=ax)


def plot_john_interpolated_points() -> None:
    data = load_john_interpolation()
    mocked_data = mock_even_potential(data)

    fig, ax, _anim1 = animate_potential_x0x1(mocked_data)
    ax.set_title(
        "Plot of the interpolated Nickel surface potential\n" "through the x plane"
    )
    fig.show()

    fig, ax, _anim2 = animate_potential_x0x1(mocked_data)
    ax.set_title(
        "Plot of the interpolated Nickel surface potential\n" "through the y plane"
    )

    fig.show()

    fig, ax = plot_z_direction_energy_data_john(data)
    ax.set_ylim(0, 0.3e-18)
    fig.show()
    save_figure(fig, "john_interpolation_z.png")
    input()


def compare_john_interpolation() -> None:
    raw_points = load_raw_data()
    john_interpolation = load_john_interpolation()
    mocked_interpolation = mock_even_potential(john_interpolation)

    fig, ax, _anim2 = animate_potential_x0x1(mocked_interpolation)
    plot_point_potential_location_xy(raw_points, ax=ax)
    fig.show()

    fig, ax = compare_energy_grid_to_all_raw_points(raw_points, john_interpolation)
    ax.set_ylim(0, 3 * 10**-19)
    ax.set_title("Comparison between raw and interpolated potential for Nickel")
    fig.show()
    save_figure(fig, "raw_interpolation_comparison.png")

    fig, ax = plot_z_direction_energy_data_john(john_interpolation)
    my_interpolation = load_interpolated_grid()
    comparison_points = get_nickel_comparison_points_x0x1(my_interpolation)
    fig, ax = plot_potential_1D_comparison(my_interpolation, comparison_points)
    ax.set_ylim(0, 0.5e-18)
    fig.show()
    save_figure(fig, "original_and_new_interpolation_comparison.png")
    input()


def plot_interpolation_with_sho_wavefunctions() -> None:
    """
    Is is possible that the SHO wavefunctions lie outside the interpolated potential
    or have energies for which they can see the truncation process.

    Plotting them alongside the interpolation in the hZ direction will allow us to
    diagnose these issues
    """

    grid = load_interpolated_grid()
    fig, ax = plt.subplots()
    plot_z_direction_energy_data_111(grid, ax=ax)
    plot_sho_wavefunctions(
        grid["z_points"],
        sho_omega=195636899474736.66,
        mass=1.6735575e-27,
        first_n=16,
        ax=ax,
    )
    ax.set_ylim(0, 0.5e-18)
    fig.show()

    save_figure(fig, "sho_wavefunctions_alongside_potential.png")
    input()


def plot_potential_minimum_along_diagonal() -> None:
    fig, ax = plt.subplots()

    interpolation = load_interpolated_grid()
    path = np.array([(x, x) for x in range(np.shape(interpolation["points"])[0])])
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("My Interpolation")

    john_interpolation = mock_even_potential(load_john_interpolation())
    path = np.array([(0, y) for y in range(np.shape(john_interpolation["points"])[1])])
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


def plot_potential_minimum_along_edge() -> None:
    interpolation = load_interpolated_grid()
    fig, ax = plt.subplots()

    # Note we are 'missing' two points here!
    path = np.array(
        [
            (np.shape(interpolation["points"])[0] - (x), x)
            for x in range(np.shape(interpolation["points"])[0])
        ]
    ).T
    # Add a fake point here so they line up. path[0] is not included in the unit cell
    path[0] = path[2]
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("diagonal")

    path = np.array([(x, 0) for x in range(np.shape(interpolation["points"])[0])]).T
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("x1=0")

    path = np.array([(0, y) for y in range(np.shape(interpolation["points"])[1])]).T
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("x0=0")

    ax.legend()
    fig.show()
    ax.set_title(
        "plot of the potential along the edge and off diagonal. All three should be identical"
    )
    input()


def plot_potential_minimum_along_edge_reciprocal() -> None:
    """
    Is it an issue with how we lay out the raw reciprocal data,
    or is it a problem with the interpolation procedure?

    """
    potentail = load_raw_data_reciprocal_grid()
    potentail_mock = mock_even_potential(potentail)

    fig, _, _ = plot_potential_x0x1(potentail_mock, x3_idx=0)
    fig.show()

    fig, ax = plt.subplots()

    path = np.array([(x, x) for x in range(potentail_mock["points"].shape[0])]).T
    _, _, line = plot_potential_minimum_along_path(potentail_mock, path, ax=ax)
    line.set_label("x0=0")

    path = np.array(
        [
            ((np.shape(potentail_mock["points"])[1] - x) // 2, x)
            for x in range(np.shape(potentail_mock["points"])[0])
            if x % 2 == 0
        ]
    ).T
    _, _, line = plot_potential_minimum_along_path(potentail_mock, path, ax=ax)
    line.set_label("x1=0")

    path = np.array(
        [
            (y // 2, (np.shape(potentail_mock["points"])[1] - y))
            for y in range(np.shape(potentail_mock["points"])[1] + 1)
            if y % 2 == 0
        ][1:]
    ).T
    _, _, line = plot_potential_minimum_along_path(potentail_mock, path, ax=ax)
    line.set_label("diagonal")

    ax.legend()
    fig.show()
    ax.set_title(
        "plot of the potential along the edge and off diagonal.\n"
        "All three directions are identical"
    )
    input()


def test_potential_fourier_transform() -> None:
    """
    Since we are sampling in units of the bz we expect the potential to be the same at the origin
    as this just represents the 'average potential'.

    We also expect the off center to be equal,
    but the irrational unit vectors prevent us from testing this
    """
    interpolation = load_interpolated_grid()
    fft_me = np.fft.ifft2(interpolation["points"], axes=(0, 1))
    ftt_origin_me = fft_me[0, 0, np.argmin(np.abs(interpolation["basis"][2]))]

    print(ftt_origin_me, np.min(np.abs(interpolation["basis"][2])))

    x0_norm = np.linalg.norm(interpolation["basis"][0]["delta_x"])
    x1_norm = np.linalg.norm(interpolation["basis"][1]["delta_x"])
    denom = (
        interpolation["basis"][0]["delta_x"][0]
        * interpolation["basis"][1]["delta_x"][1]
        - interpolation["basis"][0]["delta_x"][1]
        * interpolation["basis"][1]["delta_x"][0]
    )
    fix_factor = x0_norm * x1_norm / (denom)
    print(ftt_origin_me / fix_factor)

    john_grid_interpolation = load_interpolated_john_grid()
    fft_john = np.fft.ifft2(john_grid_interpolation["points"], axes=(0, 1))
    ftt_origin_john = fft_john[
        0, 0, np.argmin(np.abs(john_grid_interpolation["basis"][2]))
    ]

    print(ftt_origin_john, np.min(np.abs(john_grid_interpolation["basis"][2])))

    # Good enough
    # Max absolute difference: 3.31267687e-21
    # Max relative difference: 0.00026576
    np.testing.assert_allclose(fft_john[0, 0, :], fft_me[0, 0, :])


def test_symmetry_point_interpolation() -> None:
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

    delta_x = 2 * (np.max(raw_points["x_points"]) - np.min(raw_points["x_points"]))  # type: ignore
    # These are calculated assuming no symmetry point!
    delta_x_john = interpolation["basis"][0]["delta_x"]

    delta_y = 2 * (np.max(raw_points["y_points"]) - np.min(raw_points["y_points"]))  # type: ignore
    delta_y_john = interpolation["basis"][1]["delta_x"]
    # True - we have excluded the symmetry points properly!
    print(np.allclose([delta_y, delta_x], [delta_x_john, delta_y_john]))
