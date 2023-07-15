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
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from surface_potential_analysis.state_vector.plot import (
    animate_eigenstate_x1x2,
    plot_eigenstate_2d_x,
    plot_eigenstate_x0x1,
    plot_state_vector_along_path,
)
from surface_potential_analysis.util.util import slice_along_axis
from surface_potential_analysis.wavepacket.get_eigenstate import get_eigenstate
from surface_potential_analysis.wavepacket.localization import (
    get_wavepacket_two_points,
)
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_eigenvalues_2d_k,
    plot_wavepacket_eigenvalues_2d_x,
    plot_wavepacket_sample_frequencies,
    plot_wavepacket_x0x1,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_unfurled_basis,
    get_wavepacket_sample_fractions,
)

from .s4_wavepacket import (
    get_two_point_normalized_wavepacket_hydrogen,
    get_wavepacket_hydrogen,
)
from .surface_data import save_figure

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleIndexLike3d
    from surface_potential_analysis.basis.basis import AxisWithLengthBasis
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
    util = BasisUtil(position_basis)
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    converted = convert_state_vector_to_basis(eigenstate, position_basis)
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
    wavepacket = get_wavepacket_hydrogen(0)

    fig, _, _ = plot_wavepacket_sample_frequencies(wavepacket)
    fig.show()

    input()


def plot_nickel_wavepacket_energies() -> None:
    for i in range(10):
        wavepacket = get_wavepacket_hydrogen(i)
        fig, _, _ = plot_wavepacket_eigenvalues_2d_k(wavepacket)
        fig.show()

        fig, _, _ = plot_wavepacket_eigenvalues_2d_x(wavepacket)
        fig.show()
    input()


def animate_wavepacket_eigenstates_x1x2() -> None:
    wavepacket = get_wavepacket_hydrogen(0)

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

    wavepacket = get_wavepacket_hydrogen(1)

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

    wavepacket = get_wavepacket_hydrogen(2)

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
    wavepacket = get_two_point_normalized_wavepacket_hydrogen(0)
    fig, _, _anim0 = animate_wavepacket_x0x1(wavepacket, scale="symlog")
    fig.show()

    wavepacket = get_two_point_normalized_wavepacket_hydrogen(1)
    fig, _, _anim1 = animate_wavepacket_x0x1(wavepacket, scale="symlog")
    fig.show()
    input()


def plot_hydrogen_wavepacket_at_x2_max() -> None:
    for band in range(0, 6):
        normalized = get_two_point_normalized_wavepacket_hydrogen(band)
        _, _, x2_max = BasisUtil(normalized["basis"]).get_stacked_index(
            np.argmax(np.abs(normalized["vectors"][0]))
        )
        fig, ax, _ = plot_wavepacket_x0x1(normalized, x2_max, scale="symlog")
        fig.show()
        ax.set_title(f"Plot of abs(wavefunction) for ix2={x2_max}")
        save_figure(fig, f"./wavepacket/wavepacket_{band}.png")
    input()


def plot_two_point_wavepacket_with_idx() -> None:
    offset = (2, 1)
    for band in range(0, 6):
        normalized = get_two_point_normalized_wavepacket_hydrogen(band, offset)

        fig, ax = plt.subplots()

        idx0, idx1 = get_wavepacket_two_points(normalized, offset)
        unfurled_basis: AxisWithLengthBasis[Literal[3]] = get_unfurled_basis(
            normalized["basis"], normalized["shape"]
        )
        plot_fundamental_x_at_index_projected_2d(unfurled_basis, idx0, (0, 1), ax=ax)
        plot_fundamental_x_at_index_projected_2d(unfurled_basis, idx1, (0, 1), ax=ax)

        plot_wavepacket_x0x1(normalized, idx0[2], measure="abs", ax=ax)

        fig.show()
        ax.set_title(f"Plot of abs(wavefunction) for ix2={idx0[2]}")
        save_figure(fig, f"./wavepacket/wavepacket_{band}.png")
    input()


def plot_nickel_wavepacket_eigenstate() -> None:
    for band in range(20):
        wavepacket = get_wavepacket_hydrogen(band)
        eigenstate = get_eigenstate(wavepacket, 0)
        _, _, x2_max = BasisUtil(eigenstate["basis"]).get_stacked_index(
            np.argmax(np.abs(eigenstate["vector"]))
        )

        fig, ax, _ = plot_eigenstate_2d_x(eigenstate, (0, 1), (x2_max,), measure="abs")
        fig.show()

    input()


def plot_phase_around_origin() -> None:
    wavepacket = get_wavepacket_hydrogen(band=2)
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
    plot_fundamental_x_at_index_projected_2d(flat["basis"], idx, (0, 1), ax=ax)
    fig.show()

    fig, ax, _ = plot_eigenstate_x0x1(flat, 0, measure="real")
    plot_fundamental_x_at_index_projected_2d(flat["basis"], idx, (0, 1), ax=ax)
    fig.show()

    fig, ax, _ = plot_state_vector_along_path(flat, path, wrap_distances=True)
    ax.set_title("plot of abs against distance for the eigenstate")
    fig.show()

    fig, ax, _ = plot_state_vector_along_path(
        flat, path, wrap_distances=True, measure="angle"
    )
    ax.set_title("plot of angle against distance for the eigenstate")
    fig.show()

    fig, ax, _ = plot_state_vector_along_path(
        flat, path, wrap_distances=True, measure="real"
    )
    ax.set_title("plot of real against distance for the eigenstate")
    fig.show()

    fig, ax, _ = plot_state_vector_along_path(
        flat, path, wrap_distances=True, measure="imag"
    )
    ax.set_title("plot of imag against distance for the eigenstate")
    fig.show()
    input()
