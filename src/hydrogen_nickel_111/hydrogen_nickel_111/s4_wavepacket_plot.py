from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.axis.axis import FundamentalTransformedPositionAxis3d
from surface_potential_analysis.basis.plot import (
    plot_fundamental_x_at_index_projected_2d,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.plot import (
    animate_state_x1x2,
    plot_state_2d_x,
    plot_state_2d_x_max,
    plot_state_along_path,
)
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_shape
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_state_vector,
    get_tight_binding_state,
)
from surface_potential_analysis.wavepacket.localization import (
    localize_wavepacket_wannier90_many_band,
)
from surface_potential_analysis.wavepacket.localization._tight_binding import (
    get_wavepacket_two_points,
)
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_2d_x_max,
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
    get_single_point_projection_localized_wavepacket_hydrogen,
    get_tight_binding_projection_localized_wavepacket_hydrogen,
    get_two_point_localized_wavepacket_hydrogen,
    get_wannier90_localized_wavepacket_hydrogen,
    get_wavepacket_hydrogen,
)
from .surface_data import save_figure

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import AxisWithLengthBasis
    from surface_potential_analysis.state_vector.state_vector import StateVector3d


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

    state_1: StateVector3d[Any] = get_state_vector(wavepacket, (0, 0))
    state_1["basis"] = (
        FundamentalTransformedPositionAxis3d(
            state_1["basis"][0].delta_x, state_1["basis"][0].n
        ),
        FundamentalTransformedPositionAxis3d(
            state_1["basis"][1].delta_x, state_1["basis"][1].n
        ),
        state_1["basis"][2],
    )
    fig, _, _anim0 = animate_state_x1x2(state_1, measure="real")
    fig.show()

    wavepacket = get_wavepacket_hydrogen(1)

    state_2: StateVector3d[Any] = get_state_vector(wavepacket, (0, 0))
    state_2["basis"] = (
        FundamentalTransformedPositionAxis3d(
            state_2["basis"][0].delta_x, state_2["basis"][0].n
        ),
        FundamentalTransformedPositionAxis3d(
            state_2["basis"][1].delta_x, state_2["basis"][1].n
        ),
        state_2["basis"][2],
    )
    fig, _, _anim1 = animate_state_x1x2(state_2, measure="real")
    fig.show()

    wavepacket = get_wavepacket_hydrogen(2)

    state_3: StateVector3d[Any] = get_state_vector(wavepacket, (0, 0))
    state_3["basis"] = (
        FundamentalTransformedPositionAxis3d(
            state_3["basis"][0].delta_x, state_3["basis"][0].n
        ),
        FundamentalTransformedPositionAxis3d(
            state_3["basis"][1].delta_x, state_3["basis"][1].n
        ),
        state_3["basis"][2],
    )
    fig, _, _anim2 = animate_state_x1x2(state_3, measure="real")
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
    wavepacket = get_two_point_localized_wavepacket_hydrogen(0)
    fig, _, _anim0 = animate_wavepacket_x0x1(wavepacket, scale="symlog")
    fig.show()

    wavepacket = get_two_point_localized_wavepacket_hydrogen(1)
    fig, _, _anim1 = animate_wavepacket_x0x1(wavepacket, scale="symlog")
    fig.show()
    input()


def plot_hydrogen_wavepacket_at_x2_max() -> None:
    for band in range(0, 6):
        normalized = get_two_point_localized_wavepacket_hydrogen(band)
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
        normalized = get_two_point_localized_wavepacket_hydrogen(band, offset)

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
        state = get_state_vector(wavepacket, 0)

        fig, ax, _ = plot_state_2d_x_max(state, (0, 1), measure="abs")
        fig.show()

    input()


def plot_phase_around_origin() -> None:
    wavepacket = get_wavepacket_hydrogen(2)
    eigenstate = get_state_vector(wavepacket, 0)

    path = np.array(
        [
            [8, 6, 4, 2, 89, 86, 83, 80],
            [80, 83, 86, 89, 2, 4, 6, 8],
            [124, 124, 124, 124, 124, 124, 124, 124],
        ]
    )
    idx = (path[0], path[1], path[2])
    fig, ax, _ = plot_state_2d_x(eigenstate, (0, 1), (124,))
    plot_fundamental_x_at_index_projected_2d(eigenstate["basis"], idx, (0, 1), ax=ax)
    fig.show()

    fig, ax, _ = plot_state_2d_x(eigenstate, (0, 1), (124,), measure="real")
    plot_fundamental_x_at_index_projected_2d(eigenstate["basis"], idx, (0, 1), ax=ax)
    fig.show()

    fig, ax, _ = plot_state_along_path(eigenstate, path, wrap_distances=True)
    ax.set_title("plot of abs against distance for the eigenstate")
    fig.show()

    fig, ax, _ = plot_state_along_path(
        eigenstate, path, wrap_distances=True, measure="angle"
    )
    ax.set_title("plot of angle against distance for the eigenstate")
    fig.show()

    fig, ax, _ = plot_state_along_path(
        eigenstate, path, wrap_distances=True, measure="real"
    )
    ax.set_title("plot of real against distance for the eigenstate")
    fig.show()

    fig, ax, _ = plot_state_along_path(
        eigenstate, path, wrap_distances=True, measure="imag"
    )
    ax.set_title("plot of imag against distance for the eigenstate")
    fig.show()
    input()


def plot_tight_binding_projection_localized_wavepacket_hydrogen() -> None:
    for band in [4]:
        wavepacket = get_tight_binding_projection_localized_wavepacket_hydrogen(band)
        tight_binding_state = get_tight_binding_state(wavepacket)
        fig, ax, _ = plot_state_2d_x_max(tight_binding_state, (0, 1), scale="symlog")
        fig.show()
        input()

        fig, ax, _ = plot_wavepacket_2d_x_max(wavepacket, (0, 1), scale="symlog")
        fig.show()
        input()


def plot_two_point_localized_wavepacket_hydrogen() -> None:
    for band in [0, 3]:
        wavepacket = get_two_point_localized_wavepacket_hydrogen(band)
        fig, ax, _ = plot_wavepacket_2d_x_max(wavepacket, (0, 1), scale="symlog")
        fig.show()
        input()


def plot_single_point_projection_localized_wavepacket_hydrogen() -> None:
    for band in [0, 3]:
        wavepacket = get_single_point_projection_localized_wavepacket_hydrogen(band)
        fig, ax, _ = plot_wavepacket_2d_x_max(wavepacket, (0, 1), scale="symlog")
        fig.show()
        input()


def plot_wannier90_localized_wavepacket_hydrogen() -> None:
    for band in [0, 1, 2, 3, 4, 5]:
        wavepacket = get_wannier90_localized_wavepacket_hydrogen(band)
        fig, _, _ = plot_wavepacket_2d_x_max(wavepacket, (0, 1), scale="symlog")
        fig.show()

        fig, _, _ = plot_wavepacket_2d_x_max(wavepacket, (1, 2), scale="symlog")
        fig.show()
        input()


def plot_wannier90_many_band_localized_wavepacket_hydrogen() -> None:
    wavepackets = [
        convert_wavepacket_to_shape(get_wavepacket_hydrogen(2), (4, 4, 1)),
        convert_wavepacket_to_shape(get_wavepacket_hydrogen(3), (4, 4, 1)),
        convert_wavepacket_to_shape(get_wavepacket_hydrogen(6), (4, 4, 1)),
    ]
    localized = localize_wavepacket_wannier90_many_band(wavepackets)

    fig, _, _ = plot_wavepacket_2d_x_max(localized[0], (0, 1), scale="symlog")
    fig.show()

    fig, _, _ = plot_wavepacket_2d_x_max(localized[0], (1, 2), scale="symlog")
    fig.show()

    fig, _, _ = plot_wavepacket_2d_x_max(localized[1], (0, 1), scale="symlog")
    fig.show()

    fig, _, _ = plot_wavepacket_2d_x_max(localized[1], (1, 2), scale="symlog")
    fig.show()
    input()
