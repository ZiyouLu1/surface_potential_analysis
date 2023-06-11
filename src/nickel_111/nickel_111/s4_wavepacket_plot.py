from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.axis.axis import FundamentalMomentumAxis3d
from surface_potential_analysis.axis.conversion import (
    axis_as_fundamental_position_axis,
    axis_as_single_point_axis,
)
from surface_potential_analysis.basis.plot import (
    plot_fundamental_x_at_index_projected_2d,
)
from surface_potential_analysis.basis.util import Basis3dUtil
from surface_potential_analysis.state_vector.conversion import (
    convert_eigenstate_to_basis,
)
from surface_potential_analysis.state_vector.plot import (
    animate_eigenstate_x1x2,
    plot_eigenstate_along_path,
    plot_eigenstate_x0x1,
)
from surface_potential_analysis.util.util import slice_along_axis
from surface_potential_analysis.wavepacket.get_eigenstate import get_eigenstate
from surface_potential_analysis.wavepacket.localization import (
    get_wavepacket_two_points,
)
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_energies_momentum,
    plot_wavepacket_energies_position,
    plot_wavepacket_sample_frequencies,
    plot_wavepacket_x0x1,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_unfurled_basis,
    get_wavepacket_sample_fractions,
    load_wavepacket,
)

from nickel_111.s4_wavepacket import (
    MAXIMUM_POINTS,
    load_nickel_wavepacket,
    load_normalized_nickel_wavepacket_momentum,
    load_two_point_normalized_nickel_wavepacket_momentum,
)

from .surface_data import get_data_path, save_figure

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleIndexLike3d
    from surface_potential_analysis.state_vector.state_vector import StateVector3d


def flaten_eigenstate_x(
    eigenstate: StateVector3d[Any],
    idx: SingleIndexLike3d,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
) -> StateVector3d[Any]:
    """
    Flatten the eigenstate in the z direction, at the given index in position basis.

    Parameters
    ----------
    eigenstate : EigenstateWithBasis[_A3d0Inv, _A3d1Inv, _A3d2Inv]
    idx : int
        index in position basis to flatten
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis along which to flatten

    Returns
    -------
    EigenstateWithBasis[Any, Any, Any]
        _description_
    """
    position_basis = (
        eigenstate["basis"][0],
        eigenstate["basis"][1],
        axis_as_fundamental_position_axis(eigenstate["basis"][2]),
    )
    util = Basis3dUtil(position_basis)
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    converted = convert_eigenstate_to_basis(eigenstate, position_basis)
    flattened = (
        converted["vector"]
        .reshape(*util.shape)[slice_along_axis(idx, z_axis)]
        .reshape(-1)
    )
    basis = (
        eigenstate["basis"][0],
        eigenstate["basis"][1],
        axis_as_single_point_axis(eigenstate["basis"][2]),
    )
    return {"basis": basis, "vector": flattened}


def plot_nickel_wavepacket_points() -> None:
    wavepacket = load_nickel_wavepacket(0)

    fig, _, _ = plot_wavepacket_sample_frequencies(wavepacket)
    fig.show()

    input()


def plot_nickel_wavepacket_energies() -> None:
    for i in range(10):
        wavepacket = load_nickel_wavepacket(i)
        fig, _, _ = plot_wavepacket_energies_momentum(wavepacket)
        fig.show()

        fig, _, _ = plot_wavepacket_energies_position(wavepacket)
        fig.show()
    input()


def animate_wavepacket_eigenstates_x1x2() -> None:
    wavepacket = load_nickel_wavepacket(0)

    eigenstate: StateVector3d[Any] = get_eigenstate(wavepacket, (0, 0))
    eigenstate["basis"] = (
        FundamentalMomentumAxis3d(
            eigenstate["basis"][0].delta_x, eigenstate["basis"][0].n
        ),
        FundamentalMomentumAxis3d(
            eigenstate["basis"][1].delta_x, eigenstate["basis"][1].n
        ),
        eigenstate["basis"][2],
    )
    fig, _, _anim0 = animate_eigenstate_x1x2(eigenstate, measure="real")
    fig.show()

    path = get_data_path("wavepacket_1.npy")
    wavepacket = load_wavepacket(path)

    eigenstate2: StateVector3d[Any] = get_eigenstate(wavepacket, (0, 0))
    eigenstate2["basis"] = (
        FundamentalMomentumAxis3d(
            eigenstate2["basis"][0].delta_x, eigenstate2["basis"][0].n
        ),
        FundamentalMomentumAxis3d(
            eigenstate2["basis"][1].delta_x, eigenstate2["basis"][1].n
        ),
        eigenstate2["basis"][2],
    )
    fig, _, _anim1 = animate_eigenstate_x1x2(eigenstate2, measure="real")
    fig.show()

    path = get_data_path("wavepacket_2.npy")
    wavepacket = load_wavepacket(path)

    eigenstate3: StateVector3d[Any] = get_eigenstate(wavepacket, (0, 0))
    eigenstate3["basis"] = (
        FundamentalMomentumAxis3d(
            eigenstate3["basis"][0].delta_x, eigenstate3["basis"][0].n
        ),
        FundamentalMomentumAxis3d(
            eigenstate3["basis"][1].delta_x, eigenstate3["basis"][1].n
        ),
        eigenstate3["basis"][2],
    )
    fig, _, _anim2 = animate_eigenstate_x1x2(eigenstate3, measure="real")
    fig.show()
    input()


def plot_wavepacket_points_john() -> None:
    a = np.array(
        [
            [+0, +2],
            [-1, +2],
            [+1, +1],
            [-2, +2],
            [+0, +1],
            [+2, +0],
            [-1, +1],
            [+1, +0],
            [-2, +1],
            [+0, +0],
            [+2, -1],
            [-1, +0],
            [+1, -1],
            [-2, +0],
            [+0, -1],
            [+2, -2],
            [-1, -1],
            [+1, -2],
            [+0, -2],
        ]
    )
    # = 2 x 2pi / delta_y
    g1 = 2.9419 * 10**10
    n_q_dim = 4

    qx_points = g1 * ((np.sqrt(3) / 2) * a[:, 0] + 0 * a[:, 1]) / n_q_dim
    qy_points = g1 * ((1 / 2) * a[:, 0] + a[:, 1]) / n_q_dim

    fig, ax = plt.subplots()
    (line,) = ax.plot(qx_points, qy_points)
    line.set_linestyle("")
    line.set_marker("x")
    ax.set_title("Plot of points as chosen by John")
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("ky")
    ax.set_xlabel("kx")
    fig.show()
    input()
    save_figure(fig, "john_wavepacket_points.png")


def plot_wavepacket_points_me() -> None:
    fractions = get_wavepacket_sample_fractions(np.array([10, 10]))
    fig, ax = plt.subplots()
    (line,) = ax.plot(*fractions[0:2])
    line.set_linestyle("")
    line.set_marker("x")
    ax.set_title("Plot of points as chosen by Me")
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("ky")
    ax.set_xlabel("kx")
    fig.show()
    input()
    save_figure(fig, "my_wavepacket_points.png")


def animate_nickel_wavepacket() -> None:
    wavepacket = load_normalized_nickel_wavepacket_momentum(0, (0, 0, 117), 0)

    fig, _, _anim0 = animate_wavepacket_x0x1(wavepacket, scale="symlog")
    fig.show()

    wavepacket = load_normalized_nickel_wavepacket_momentum(1, (8, 8, 118), 0)

    fig, _, _anim1 = animate_wavepacket_x0x1(wavepacket, scale="symlog")
    fig.show()

    input()


def plot_wavepacket_at_maximum_points_x2() -> None:
    for band in range(0, 6):
        max_point = MAXIMUM_POINTS[band]
        # normalized = load_normalized_nickel_wavepacket_momentum(band, max_point, 0)  # noqa: ERA001
        # normalized = load_two_point_normalized_nickel_wavepacket_momentum(band, 0) # noqa: ERA001
        normalized = load_two_point_normalized_nickel_wavepacket_momentum(band)

        fig, ax, _ = plot_wavepacket_x0x1(normalized, max_point[2], measure="abs")
        fig.show()
        ax.set_title(f"Plot of abs(wavefunction) for ix2={max_point[2]}")
        save_figure(fig, f"./wavepacket/wavepacket_{band}.png")
    input()


def plot_two_point_wavepacket_with_idx() -> None:
    offset = (2, 1)
    for band in range(0, 6):
        normalized = load_two_point_normalized_nickel_wavepacket_momentum(band, offset)

        fig, ax = plt.subplots()

        idx0, idx1 = get_wavepacket_two_points(normalized, offset)
        unfurled_basis = get_unfurled_basis(normalized["basis"], normalized["shape"])
        plot_fundamental_x_at_index_projected_2d(unfurled_basis, idx0, ax=ax)
        plot_fundamental_x_at_index_projected_2d(unfurled_basis, idx1, ax=ax)

        plot_wavepacket_x0x1(normalized, idx0[2], measure="abs", ax=ax)

        fig.show()
        ax.set_title(f"Plot of abs(wavefunction) for ix2={idx0[2]}")
        save_figure(fig, f"./wavepacket/wavepacket_{band}.png")
    input()


def plot_nickel_wavepacket_eigenstate() -> None:
    for band in range(20):
        wavepacket = load_nickel_wavepacket(band)
        eigenstate = get_eigenstate(wavepacket, 0)

        fig, ax, _ = plot_eigenstate_x0x1(
            eigenstate, MAXIMUM_POINTS[band][2], measure="abs"
        )
        fig.show()

    input()


def plot_phase_around_origin() -> None:
    wavepacket = load_nickel_wavepacket(band=2)
    eigenstate = get_eigenstate(wavepacket, 0)

    flat = flaten_eigenstate_x(eigenstate, 124, 2)
    flat["basis"][0]["parent"]["n"] = 92  # type: ignore[typeddict-item]
    flat["basis"][1]["parent"]["n"] = 92  # type: ignore[typeddict-item]
    # 2, 21, 124
    path = np.array(
        [
            [8, 6, 4, 2, 89, 86, 83, 80],
            [80, 83, 86, 89, 2, 4, 6, 8],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    idx = (path[0], path[1], path[2])
    fig, ax, _ = plot_eigenstate_x0x1(flat, 0)
    plot_fundamental_x_at_index_projected_2d(flat["basis"], idx, z_axis=2, ax=ax)
    fig.show()

    fig, ax, _ = plot_eigenstate_x0x1(flat, 0, measure="real")
    plot_fundamental_x_at_index_projected_2d(flat["basis"], idx, z_axis=2, ax=ax)
    fig.show()

    fig, ax, _ = plot_eigenstate_along_path(flat, path, wrap_distances=True)
    ax.set_title("plot of abs against distance for the eigenstate")
    fig.show()

    fig, ax, _ = plot_eigenstate_along_path(
        flat, path, wrap_distances=True, measure="angle"
    )
    ax.set_title("plot of angle against distance for the eigenstate")
    fig.show()

    fig, ax, _ = plot_eigenstate_along_path(
        flat, path, wrap_distances=True, measure="real"
    )
    ax.set_title("plot of real against distance for the eigenstate")
    fig.show()

    fig, ax, _ = plot_eigenstate_along_path(
        flat, path, wrap_distances=True, measure="imag"
    )
    ax.set_title("plot of imag against distance for the eigenstate")
    fig.show()
    input()
