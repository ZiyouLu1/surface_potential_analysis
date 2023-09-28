from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.stacked_basis.plot import (
    plot_fundamental_x_at_index_projected_2d,
)
from surface_potential_analysis.state_vector.plot import (
    plot_state_2d_x,
    plot_state_along_path,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_wavepacket_state_vector,
)
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_3d_x,
    plot_wavepacket_2d_k,
    plot_wavepacket_eigenvalues_2d_k,
    plot_wavepacket_eigenvalues_2d_x,
    plot_wavepacket_sample_frequencies,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_wavepacket,
    wavepacket_list_into_iter,
)

from .s4_wavepacket import (
    get_wannier90_localized_split_bands_wavepacket_hydrogen,
    get_wannier90_localized_wavepacket_hydrogen,
    get_wavepacket_hydrogen,
)
from .surface_data import save_figure


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


def animate_nickel_wavepacket() -> None:
    wavepackets = get_wannier90_localized_split_bands_wavepacket_hydrogen()
    wavepacket = get_wavepacket(wavepackets, 0)
    fig, _, _anim0 = animate_wavepacket_3d_x(wavepacket, scale="symlog")
    fig.show()

    wavepacket = get_wavepacket(wavepackets, 1)
    fig, _, _anim1 = animate_wavepacket_3d_x(wavepacket, scale="symlog")
    fig.show()
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


def plot_wannier90_localized_wavepacket_hydrogen() -> None:
    wavepackets = get_wannier90_localized_split_bands_wavepacket_hydrogen()
    for wavepacket in wavepacket_list_into_iter(wavepackets):
        fig, _, _ = plot_wavepacket_2d_k(wavepacket, (0, 1), scale="linear")
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(wavepacket, (1, 2), scale="symlog")
        fig.show()
    input()

    wavepackets = get_wannier90_localized_wavepacket_hydrogen(0, 8)
    for wavepacket in wavepacket_list_into_iter(wavepackets):
        fig, _, _ = plot_wavepacket_2d_k(wavepacket, (0, 1), scale="linear")
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(wavepacket, (1, 2), scale="linear")
        fig.show()
    input()
