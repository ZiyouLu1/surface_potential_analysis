from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.axis.util import BasisUtil
from surface_potential_analysis.stacked_basis.plot import (
    plot_fundamental_x_at_index_projected_2d,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.state_vector.plot import (
    plot_state_2d_k,
    plot_state_2d_x,
    plot_state_along_path,
    plot_state_difference_2d_x,
)
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.util.interpolation import pad_ft_points
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_shape
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_tight_binding_state,
    get_wavepacket_state_vector,
)
from surface_potential_analysis.wavepacket.localization import (
    localize_single_point_projection,
    localize_wavepacket_wannier90_many_band,
)
from surface_potential_analysis.wavepacket.localization._tight_binding import (
    get_wavepacket_two_points,
)
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_3d_x,
    plot_wavepacket_2d_k,
    plot_wavepacket_2d_x,
    plot_wavepacket_eigenvalues_2d_k,
    plot_wavepacket_eigenvalues_2d_x,
    plot_wavepacket_sample_frequencies,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    get_unfurled_basis,
    get_wavepacket_sample_fractions,
)

from .s4_wavepacket import (
    get_all_wavepackets_hydrogen,
    get_single_point_projection_localized_wavepacket_hydrogen,
    get_tight_binding_projection_localized_wavepacket_hydrogen,
    get_two_point_localized_wavepacket_hydrogen,
    get_wannier90_localized_wavepacket_hydrogen,
    get_wavepacket_hydrogen,
)
from .surface_data import get_data_path, save_figure

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        ExplicitBasis,
        FundamentalBasis,
        TransformedPositionBasis,
    )
    from surface_potential_analysis.axis.stacked_axis import (
        StackedBasisLike,
    )


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


def plot_wavepacket_eigenstates() -> None:
    for band in range(20):
        wavepacket = get_wavepacket_hydrogen(band)
        state = get_wavepacket_state_vector(wavepacket, 0)

        fig, _, _ = plot_state_2d_x(state, (0, 1), measure="abs")
        fig.show()

    input()


def animate_wavepacket_eigenstates() -> None:
    for band in [0, 1, 2]:
        wavepacket = get_wavepacket_hydrogen(band)
        state = get_wavepacket_state_vector(wavepacket, (0, 0))
        fig, _, _anim0 = animate_wavepacket_3d_x(state, (1, 2), measure="real")  # type: ignore[type-var]
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
    fig, _, _anim0 = animate_wavepacket_3d_x(wavepacket, scale="symlog")
    fig.show()

    wavepacket = get_two_point_localized_wavepacket_hydrogen(1)
    fig, _, _anim1 = animate_wavepacket_3d_x(wavepacket, scale="symlog")
    fig.show()
    input()


def plot_two_point_wavepacket_with_idx() -> None:
    offset = (2, 1)
    for band in range(6):
        normalized = get_two_point_localized_wavepacket_hydrogen(band, offset)

        fig, ax = plt.subplots()

        idx0, idx1 = get_wavepacket_two_points(normalized, offset)
        unfurled_basis: StackedBasisLike = get_unfurled_basis(
            normalized["list_basis"], normalized["basis"]
        )
        plot_fundamental_x_at_index_projected_2d(unfurled_basis, idx0, (0, 1), ax=ax)
        plot_fundamental_x_at_index_projected_2d(unfurled_basis, idx1, (0, 1), ax=ax)

        plot_wavepacket_2d_x(normalized, idx0[2], measure="abs", ax=ax)

        fig.show()
        ax.set_title(f"Plot of abs(wavefunction) for ix2={idx0[2]}")
        save_figure(fig, f"./wavepacket/wavepacket_{band}.png")
    input()


def plot_phase_around_origin() -> None:
    wavepacket = get_wavepacket_hydrogen(2)
    eigenstate = get_wavepacket_state_vector(wavepacket, 0)

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
        fig, _, _ = plot_state_2d_x(tight_binding_state, (0, 1), scale="symlog")
        fig.show()

        fig, _, _ = plot_wavepacket_2d_x(wavepacket, (0, 1), scale="symlog")
        fig.show()
        input()


def plot_two_point_localized_wavepacket_hydrogen() -> None:
    for band in [0, 3]:
        wavepacket = get_two_point_localized_wavepacket_hydrogen(band)
        fig, ax, _ = plot_wavepacket_2d_x(wavepacket, (0, 1), scale="symlog")
        fig.show()
        input()


def plot_single_point_projection_localized_wavepacket_hydrogen() -> None:
    for band in [3]:
        wavepacket = get_single_point_projection_localized_wavepacket_hydrogen(band)
        vectors = wavepacket["data"].reshape(12, 12, 1, -1)
        vectors[::2, ::] = 0
        vectors[::, ::2] = 0
        wavepacket["data"] = vectors.reshape(12 * 12 * 1, -1)
        fig, _, _ = plot_wavepacket_2d_x(
            wavepacket, (0, 1), scale="symlog", measure="real"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_x(
            wavepacket, (1, 2), scale="symlog", measure="real"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(
            wavepacket, (1, 0), scale="symlog", measure="real"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(
            wavepacket, (2, 1), scale="symlog", measure="real"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(
            wavepacket, (0, 2), scale="symlog", measure="real"
        )
        fig.show()
        input()


@npy_cached(get_data_path("wavepacket/many_band_localized_test4.npy"), load_pickle=True)
def get_localized_extrapolated() -> (
    Wavepacket[
        StackedBasisLike[
            FundamentalBasis[Literal[6]],
            FundamentalBasis[Literal[6]],
            FundamentalBasis[Literal[4]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
            TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
            ExplicitBasis[Literal[250], Literal[16], Literal[3]],
        ],
    ]
):
    wavepacket_0 = get_all_wavepackets_hydrogen()[3]
    return localize_single_point_projection(wavepacket_0)


@npy_cached(
    get_data_path("wavepacket/many_band_localized_test4_0.npy"), load_pickle=True
)
def get_localized_extrapolated_0() -> (
    Wavepacket[
        StackedBasisLike[
            FundamentalBasis[Literal[6]],
            FundamentalBasis[Literal[6]],
            FundamentalBasis[Literal[4]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
            TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
            ExplicitBasis[Literal[250], Literal[16], Literal[3]],
        ],
    ]
):
    wavepacket_0 = get_all_wavepackets_hydrogen()[0]
    return localize_single_point_projection(wavepacket_0)


def plot_compare_point_projection_localized_wavepacket_hydrogen() -> None:
    for _band in [3]:
        wavepacket_0 = get_localized_extrapolated_0()
        localized_0 = unfurl_wavepacket(wavepacket_0)
        converted_0 = convert_state_vector_to_position_basis(localized_0)

        wavepacket_1 = convert_wavepacket_to_shape(
            get_localized_extrapolated(), (12, 12, 1)
        )
        localized_1 = unfurl_wavepacket(wavepacket_1)
        converted_1 = convert_state_vector_to_position_basis(localized_1)
        util_1 = BasisUtil(converted_1["basis"])

        util_0 = BasisUtil(converted_0["basis"])
        converted_0["data"] = pad_ft_points(
            converted_0["data"].reshape(util_0.shape), util_1.shape, (0, 1, 2)
        ).reshape(-1)
        converted_0["basis"] = converted_1["basis"]

        max_idx = util_1.get_stacked_index(np.argmax(np.abs(converted_1["data"])))

        fig, _, _ = plot_state_2d_x(converted_0, (0, 1), (max_idx[2],), scale="symlog")
        fig.show()
        fig, _, _ = plot_state_2d_k(localized_0, (1, 0), scale="symlog")
        fig.show()

        fig, _, _ = plot_state_2d_x(converted_1, (0, 1), (max_idx[2],), scale="symlog")
        fig.show()
        fig, _, _ = plot_state_difference_2d_x(
            converted_0,
            converted_1,
            (0, 1),
            (max_idx[2],),
            scale="symlog",
            measure="real",
        )
        fig.show()
        input()


def plot_wannier90_localized_wavepacket_hydrogen() -> None:
    for band in [0, 1, 2, 3, 4, 5]:
        wavepacket = get_wannier90_localized_wavepacket_hydrogen(band)
        fig, _, _ = plot_wavepacket_2d_x(wavepacket, (0, 1), scale="symlog")
        fig.show()

        fig, _, _ = plot_wavepacket_2d_x(wavepacket, (1, 2), scale="symlog")
        fig.show()
        input()


# @npy_cached(get_data_path("wavepacket/many_band_localized_test2.npy"), load_pickle=True)
def get_many_band_localized() -> (
    list[
        Wavepacket[
            StackedBasisLike[
                FundamentalBasis[Literal[6]],
                FundamentalBasis[Literal[6]],
                FundamentalBasis[Literal[4]],
            ],
            StackedBasisLike[
                TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
                TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
                ExplicitBasis[Literal[250], Literal[16], Literal[3]],
            ],
        ]
    ]
):
    wavepackets = get_all_wavepackets_hydrogen()[3:4]
    return localize_wavepacket_wannier90_many_band(wavepackets)


def plot_wannier90_many_band_localized_wavepacket_hydrogen() -> None:
    localized = get_many_band_localized()

    for wavepacket in localized:
        fig, _, _ = plot_wavepacket_2d_x(wavepacket, (0, 1), scale="symlog")
        fig.show()

        fig, _, _ = plot_wavepacket_2d_x(wavepacket, (1, 2), scale="symlog")
        fig.show()
    input()
