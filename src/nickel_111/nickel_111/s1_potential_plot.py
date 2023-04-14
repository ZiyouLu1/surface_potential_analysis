import math
from typing import Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.constants import electron_volt
from surface_potential_analysis.basis_config.plot import plot_projected_coordinates_2d
from surface_potential_analysis.interpolation import (
    interpolate_points_fftn,
    interpolate_points_rfft,
    interpolate_points_rfftn,
    pad_ft_points,
)
from surface_potential_analysis.potential import (
    Potential,
    UnevenPotential,
    normalize_potential,
)
from surface_potential_analysis.potential.plot import (
    animate_potential_x0x1,
    plot_potential_1d_comparison,
    plot_potential_2d,
    plot_potential_difference_2d,
    plot_potential_minimum_along_path,
    plot_potential_x0x1,
)
from surface_potential_analysis.potential.plot_point_potential import (
    plot_point_potential_all_z,
    plot_point_potential_location_xy,
)
from surface_potential_analysis.potential.plot_uneven_potential import (
    plot_uneven_potential_z_comparison,
)
from surface_potential_analysis.potential.potential import mock_even_potential

from .s1_potential import (
    get_interpolated_nickel_potential,
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
) -> dict[str, tuple[tuple[int, int], Literal[2]]]:
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
) -> dict[str, tuple[tuple[int, int], Literal[2]]]:
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

    fig, _, _ = plot_projected_coordinates_2d(mocked_potential["basis"], 0, 2)
    fig.show()

    fig, ax = plot_z_direction_energy_data_nickel_reciprocal_points(potential)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    fig, ax, _ani = animate_potential_x0x1(mocked_potential, clim=(0, 0.2e-18))
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
    plot_uneven_potential_z_comparison(potential, locations_uneven, ax=ax)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    input()


def plot_interpolated_energy_grid_points() -> None:
    potential = load_interpolated_grid()

    fig, ax, _ = plot_projected_coordinates_2d(potential["basis"], 0, 2)
    fig.show()

    fig, ax, _ani = animate_potential_x0x1(potential, clim=(0, 2e-19))

    raw_potential = load_raw_data_reciprocal_grid()
    mocked_raw_potential = mock_even_potential(raw_potential)
    plot_projected_coordinates_2d(mocked_raw_potential["basis"], 0, 2, ax=ax)
    fig.show()

    locations = get_nickel_comparison_points_x0x1(potential)
    raw_grid = normalize_potential(load_raw_data_reciprocal_grid())
    raw_grid["basis"] = (
        raw_grid["basis"][0],
        raw_grid["basis"][1],
        raw_grid["basis"][2] - raw_grid["basis"][2][0],
    )
    fig, ax = plot_z_direction_energy_data_nickel_reciprocal_points(raw_grid)
    for ln in ax.lines:
        ln.set_marker("x")
        ln.set_linestyle("")
    _, _ = plot_potential_1d_comparison(potential, locations, ax=ax)
    ax.set_ylim(0, 5e-18)

    fig.show()

    # ft_grid: EnergyGrid = {
    # TODO: delta is wrong, plot generic points and then factor out into ft.
    input()


def plot_interpolated_energy_grid_reciprocal() -> None:
    potential = load_interpolated_reciprocal_grid()
    points = potential["points"]
    points[points < 0] = np.max(points)
    potential["points"] = points

    fig, ax, _ = plot_projected_coordinates_2d(potential["basis"], 0, 2)
    fig.show()

    fig, ax, _ani = animate_potential_x0x1(potential)

    raw_grid = load_raw_data_reciprocal_grid()
    mocked_raw_grid = mock_even_potential(raw_grid)
    plot_projected_coordinates_2d(mocked_raw_grid["basis"], 0, 2, ax=ax)
    fig.show()

    comparison_points = get_nickel_comparison_points_x0x1(potential)
    fig, ax = plot_potential_1d_comparison(potential, comparison_points)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    # ft_grid: EnergyGrid = {
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
        "Plot of the interpolated Nickel surface potential\nthrough the x plane"
    )
    fig.show()

    fig, ax, _anim2 = animate_potential_x0x1(mocked_data)
    ax.set_title(
        "Plot of the interpolated Nickel surface potential\nthrough the y plane"
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

    fig, ax = plot_potential_2d(mocked_interpolation, 0, 2)
    plot_point_potential_location_xy(raw_points, ax=ax)
    ax.set_ylim(0, 3 * 10**-19)
    ax.set_title("Comparison between raw and interpolated potential for Nickel")
    fig.show()
    save_figure(fig, "raw_interpolation_comparison.png")

    fig, ax = plot_z_direction_energy_data_john(john_interpolation)
    my_interpolation = load_interpolated_grid()
    comparison_points = get_nickel_comparison_points_x0x1(my_interpolation)
    fig, ax = plot_potential_1d_comparison(my_interpolation, comparison_points)
    ax.set_ylim(0, 0.5e-18)
    fig.show()
    save_figure(fig, "original_and_new_interpolation_comparison.png")
    input()


def plot_interpolation_with_sho_wavefunctions() -> None:
    """
    Investigate the extent to which SHO wavefunctions lie outside the potential.

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
    Investigate symmetry in the potential minimum along the edge.

    Is it an issue with how we lay out the raw reciprocal data,
    or is it a problem with the interpolation procedure?.

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
    # Since we are sampling in units of the bz we expect the potential to be the same at the origin
    # as this just represents the 'average potential'.

    # We also expect the off center to be equal,
    # but the irrational unit vectors prevent us from testing this

    interpolation = load_interpolated_grid()
    fft_me = np.fft.ifft2(interpolation["points"], axes=(0, 1))
    ftt_origin_me = fft_me[0, 0, np.argmin(np.abs(interpolation["basis"][2]))]

    print(ftt_origin_me, np.min(np.abs(interpolation["basis"][2])))  # noqa: T201

    x0_norm = np.linalg.norm(interpolation["basis"][0]["delta_x"])
    x1_norm = np.linalg.norm(interpolation["basis"][1]["delta_x"])
    denom = (
        interpolation["basis"][0]["delta_x"][0]
        * interpolation["basis"][1]["delta_x"][1]
        - interpolation["basis"][0]["delta_x"][1]
        * interpolation["basis"][1]["delta_x"][0]
    )
    fix_factor = x0_norm * x1_norm / (denom)
    print(ftt_origin_me / fix_factor)  # noqa: T201

    john_grid_interpolation = load_interpolated_john_grid()
    fft_john = np.fft.ifft2(john_grid_interpolation["points"], axes=(0, 1))
    ftt_origin_john = fft_john[
        0, 0, np.argmin(np.abs(john_grid_interpolation["basis"][2]))
    ]

    print(  # noqa: T201
        ftt_origin_john, np.min(np.abs(john_grid_interpolation["basis"][2]))
    )

    # Good enough
    # Max absolute difference: 3.31267687e-21
    # Max relative difference: 0.00026576
    np.testing.assert_allclose(fft_john[0, 0, :], fft_me[0, 0, :])


def test_symmetry_point_interpolation() -> None:
    """Check if the interpolation contain the same points at x=0 and x=L."""
    raw_points = load_raw_data()
    interpolation = load_john_interpolation()
    points = np.array(interpolation["points"])

    try:
        np.testing.assert_array_equal(points[0, :, :], points[-1, :, :])
    except AssertionError:
        print("Endpoint are not the same")  # noqa: T201
    else:
        print("Endpoint the same")  # noqa: T201

    delta_x = 2 * (np.max(raw_points["x_points"]) - np.min(raw_points["x_points"]))  # type: ignore[operator]
    # These are calculated assuming no symmetry point!
    delta_x_john = interpolation["basis"][0]["delta_x"]

    delta_y = 2 * (np.max(raw_points["y_points"]) - np.min(raw_points["y_points"]))  # type: ignore[operator]
    delta_y_john = interpolation["basis"][1]["delta_x"]
    # True - we have excluded the symmetry points properly!
    print(np.allclose([delta_y, delta_x], [delta_x_john, delta_y_john]))  # noqa: T201


def plot_interpolated_potential_difference() -> None:
    potential0 = get_interpolated_nickel_potential((2 * 23, 2 * 23, 500))
    potential1 = get_interpolated_nickel_potential((10 * 23, 10 * 23, 500))
    potential1_0_basis: Potential[int, int, int] = {
        "basis": potential0["basis"],
        "points": potential1["points"][::5, ::5, :],
    }

    fig, _, _ = plot_potential_difference_2d(
        potential0, potential1_0_basis, idx=0, z_axis=2
    )
    fig.show()
    np.testing.assert_array_almost_equal(
        potential0["points"], potential1_0_basis["points"]
    )
    # Max absolute difference: 5.42341872e-31
    # Max relative difference: 6.79388726e-14
    # i.e. we see no difference in the potential (as expected)

    potential0_1_basis: Potential[int, int, int] = {
        "basis": potential0["basis"],
        "points": interpolate_points_rfftn(  # type:ignore[typeddict-item]
            potential0["points"],  # type:ignore[arg-type]
            s=(potential1["points"].shape[0], potential1["points"].shape[1]),
            axes=(0, 1),
        ),
    }
    np.testing.assert_array_almost_equal(
        potential1["points"], potential0_1_basis["points"]
    )
    # Max absolute difference: 4.17565566e-21
    # Max relative difference: 1.33803642
    # i.e. we see no difference in the potential
    # (as expected, since the larger potential is more 'smooth')

    np.testing.assert_array_almost_equal(
        potential0["points"], potential0_1_basis["points"][::5, ::5, :]
    )
    # Max absolute difference: 6.24892436e-23
    # Max relative difference: 0.23842189

    # this is bad - we should expect almost no error on the original points


def plot_interpolated_potential_difference_rfft() -> None:
    potential0 = get_interpolated_nickel_potential((2 * 23, 2 * 23, 500))
    potential0_1_basis: Potential[int, int, int] = {
        "basis": potential0["basis"],
        "points": interpolate_points_rfftn(  # type:ignore[typeddict-item]
            potential0["points"],  # type:ignore[arg-type]
            s=(5 * potential0["points"].shape[0], 5 * potential0["points"].shape[1]),
            axes=(0, 1),
        )[::5, ::5, :],
    }
    fig, _, _ = plot_potential_difference_2d(
        potential0, potential0_1_basis, idx=0, z_axis=2
    )
    fig.show()

    potential0_1_basis1: Potential[int, int, int] = {
        "basis": potential0["basis"],
        "points": np.real(
            interpolate_points_fftn(  # type:ignore[typeddict-item]
                potential0["points"],  # type:ignore[arg-type]
                s=(
                    5 * potential0["points"].shape[0],
                    5 * potential0["points"].shape[1],
                ),
                axes=(0, 1),
            )
        )[::5, ::5, :],
    }
    fig, _, _ = plot_potential_difference_2d(
        potential0, potential0_1_basis1, idx=0, z_axis=2
    )
    fig.show()

    fig, _, _ = plot_potential_2d(potential0, idx=0, z_axis=2)
    fig.show()
    input()


def compare_potential_rfft() -> None:
    potential0 = get_interpolated_nickel_potential((2 * 23, 2 * 23, 500))
    potential1 = get_interpolated_nickel_potential((10 * 23, 10 * 23, 500))

    potential0_1_basis: Potential[int, int, int] = {
        "basis": potential0["basis"],
        "points": interpolate_points_rfft(  # type:ignore[typeddict-item]
            interpolate_points_rfft(
                potential0["points"],  # type:ignore[arg-type]
                potential1["points"].shape[0],
                axis=0,
            ),
            potential1["points"].shape[1],
            axis=1,
        ),
    }
    np.testing.assert_array_almost_equal(
        potential1["points"], potential0_1_basis["points"]
    )
    # Max absolute difference: 4.17415007e-21
    # Max relative difference: 1.44881035

    np.testing.assert_array_almost_equal(
        potential0["points"], potential0_1_basis["points"][::5, ::5, :]
    )
    # Max absolute difference: 1.04279313e-22
    # Max relative difference: 0.39221331

    potential0_1_basis2: Potential[int, int, int] = {
        "basis": potential0["basis"],
        "points": interpolate_points_rfft(
            interpolate_points_rfft(
                potential0["points"] / electron_volt,
                potential1["points"].shape[0],
                axis=0,
            ),
            potential1["points"].shape[1],
            axis=1,
        )
        * electron_volt,
    }

    np.testing.assert_array_almost_equal(
        potential0["points"], potential0_1_basis2["points"][::5, ::5, :]
    )
    # Max absolute difference: 1.04279313e-22
    # Max relative difference: 0.39221331

    potential0_1_basis3: Potential[int, int, int] = {
        "basis": potential0["basis"],
        "points": interpolate_points_fftn(  # type:ignore[typeddict-item]
            potential0["points"],  # type:ignore[arg-type]
            s=(potential1["points"].shape[0], potential1["points"].shape[1]),
            axes=(0, 1),
        ),
    }
    np.testing.assert_array_almost_equal(
        potential0["points"], potential0_1_basis3["points"][::5, ::5, :]
    )
    # Max absolute difference: 7.48043224e-32
    # Max relative difference: 4.33468487e-13
    # We get the right answer!!


def analyze_interpolated_potential_difference_momentum() -> None:
    potential0 = get_interpolated_nickel_potential((2 * 23, 2 * 23, 100))
    potential1 = get_interpolated_nickel_potential((10 * 23, 10 * 23, 100))
    potential1_0_basis: Potential[int, int, int] = {
        "basis": potential0["basis"],
        "points": potential1["points"][::5, ::5, :],
    }

    ft_points0 = np.fft.fft2(potential0["points"], axes=(0, 1), norm="forward")
    ft_points1 = np.fft.fft2(potential1_0_basis["points"], axes=(0, 1), norm="forward")
    np.testing.assert_array_almost_equal(ft_points0, ft_points1)
    # Max absolute difference: 6.31088724e-30
    # Max relative difference: 4.83872206e-05

    ft_points1_original = pad_ft_points(
        np.fft.fft2(potential1["points"], axes=(0, 1), norm="forward"),
        s=(ft_points0.shape[0], ft_points0.shape[1]),
        axes=(0, 1),
    )

    np.testing.assert_array_almost_equal(ft_points0, ft_points1_original)
    np.testing.assert_array_equal(ft_points0, ft_points1_original)
    # Max absolute difference: 2.15832122e-23
    # Max relative difference: 4558.62118163

    # The two potentials still agree on the value at the 'key' points
    # in position basis, but in momentum basis the story is not so simple
    # 2.15832122e-23 is actually a pretty large 'error'
    # Maybe try rescaling, look at how this changes the interval between points!


def plot_potential_difference_very_large_resolution() -> None:
    p0 = get_interpolated_nickel_potential((100, 100, 250))
    p1 = get_interpolated_nickel_potential((250, 250, 250))
    potential0: Potential[int, int, int] = {
        "basis": (
            {"_type": "position", "delta_x": p0["basis"][0]["delta_x"], "n": 50},
            {"_type": "position", "delta_x": p0["basis"][1]["delta_x"], "n": 50},
            p0["basis"][2],
        ),
        "points": p0["points"][::2, ::2, :],
    }
    potential1: Potential[int, int, int] = {
        "basis": potential0["basis"],
        "points": p1["points"][::5, ::5, :],
    }
    amax = np.unravel_index(
        np.argmax(np.abs(potential1["points"] - potential0["points"])),
        potential0["points"].shape,
    )
    fig, _, _ = plot_potential_difference_2d(potential0, potential1, int(amax[2]), 2)
    fig.show()

    fig, _, _ = plot_potential_difference_2d(potential0, potential1, int(amax[0]), 0)
    fig.show()
    input()
