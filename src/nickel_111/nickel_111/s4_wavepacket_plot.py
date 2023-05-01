from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.eigenstate.conversion import (
    convert_sho_eigenstate_to_fundamental_xy,
)
from surface_potential_analysis.eigenstate.plot import (
    animate_eigenstate_x1x2,
    plot_eigenstate_x0x1,
)
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_energies_momentum,
    plot_wavepacket_energies_position,
    plot_wavepacket_sample_frequencies,
    plot_wavepacket_x0x1,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_eigenstate,
    get_wavepacket_sample_fractions,
    load_wavepacket,
)

from nickel_111.s4_wavepacket import (
    MAXIMUM_POINTS,
    load_nickel_wavepacket,
    load_normalized_nickel_wavepacket_momentum,
)

from .surface_data import get_data_path, save_figure


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

    eigenstate = get_eigenstate(wavepacket, (0, 0))
    fundamental = convert_sho_eigenstate_to_fundamental_xy(eigenstate)
    fig, _, _anim0 = animate_eigenstate_x1x2(fundamental, measure="real")
    fig.show()

    path = get_data_path("wavepacket_1.npy")
    wavepacket = load_wavepacket(path)

    eigenstate = get_eigenstate(wavepacket, (0, 0))
    fundamental = convert_sho_eigenstate_to_fundamental_xy(eigenstate)
    fig, _, _anim1 = animate_eigenstate_x1x2(fundamental, measure="real")
    fig.show()

    path = get_data_path("wavepacket_2.npy")
    wavepacket = load_wavepacket(path)

    eigenstate = get_eigenstate(wavepacket, (0, 0))
    fundamental = convert_sho_eigenstate_to_fundamental_xy(eigenstate)
    fig, _, _anim2 = animate_eigenstate_x1x2(fundamental, measure="real")
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
    (line,) = ax.plot(*fractions.reshape(2, -1))
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


def plot_wavepacket_at_maximum_points() -> None:
    for band in range(20):
        max_point = MAXIMUM_POINTS[band]
        normalized = load_normalized_nickel_wavepacket_momentum(band, max_point, 0)

        fig, ax, _ = plot_wavepacket_x0x1(normalized, max_point[2], measure="abs")
        fig.show()
        ax.set_title(f"Plot of abs(wavefunction) for ix2={max_point[2]}")
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
