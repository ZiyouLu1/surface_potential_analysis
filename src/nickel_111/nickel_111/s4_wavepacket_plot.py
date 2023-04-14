import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.eigenstate.eigenstate import (
    convert_sho_eigenstate_to_position_basis,
)
from surface_potential_analysis.eigenstate.plot import animate_eigenstate_x1x2
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_sample_frequencies,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_wavepacket_sample_fractions,
    load_wavepacket,
    normalize_wavepacket,
    select_wavepacket_eigenstate,
)

from .surface_data import get_data_path, save_figure


def plot_wavepacket_points() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    wavepacket = load_wavepacket(path)

    fig, _, _ = plot_wavepacket_sample_frequencies(wavepacket)
    fig.show()

    input()


def animate_wavepacket_eigenstates_x1x2() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    wavepacket = load_wavepacket(path)

    eigenstate = select_wavepacket_eigenstate(wavepacket, (0, 0))
    eigenstate_position = convert_sho_eigenstate_to_position_basis(eigenstate)

    fig, _, _anim0 = animate_eigenstate_x1x2(eigenstate_position, measure="real")
    fig.show()

    path = get_data_path("eigenstates_grid_1.json")
    wavepacket = load_wavepacket(path)

    eigenstate = select_wavepacket_eigenstate(wavepacket, (0, 0))
    eigenstate_position = convert_sho_eigenstate_to_position_basis(eigenstate)

    fig, _, _anim1 = animate_eigenstate_x1x2(eigenstate_position, measure="real")
    fig.show()

    path = get_data_path("eigenstates_grid_2.json")
    wavepacket = load_wavepacket(path)

    eigenstate = select_wavepacket_eigenstate(wavepacket, (0, 0))
    eigenstate_position = convert_sho_eigenstate_to_position_basis(eigenstate)

    fig, _, _anim2 = animate_eigenstate_x1x2(eigenstate_position, measure="real")
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
    G1 = 2.9419 * 10**10
    nqdim = 4

    qx_points = G1 * ((np.sqrt(3) / 2) * a[:, 0] + 0 * a[:, 1]) / nqdim
    qy_points = G1 * ((1 / 2) * a[:, 0] + a[:, 1]) / nqdim

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


def plot_wavepacket_grid_all_equal() -> None:
    """
    Does the imaginary oscillation in the imaginary part of the wavefunction happen
    if we choose a constant bloch wavefunction for all k.
    """
    # eigenstates["eigenvectors"] = [
    #     eigenvector.tolist() for _ in eigenstates["eigenvectors"]

    input()


def plot_wavepacket_grid() -> None:
    path = get_data_path(f"eigenstates_grid_{0}.json")
    wavepacket = load_wavepacket(path)
    normalized = normalize_wavepacket(wavepacket, 0, 0)

    fig, _, _anim0 = animate_wavepacket_x0x1(normalized, scale="symlog")
    fig.show()

    input()

    path = get_data_path(f"eigenstates_grid_{1}.json")
    wavepacket = load_wavepacket(path)
    normalized = normalize_wavepacket(wavepacket, 0, 0)
    # TODO:
    #         0,
    fig, _, _anim1 = animate_wavepacket_x0x1(normalized, scale="symlog")
    fig.show()

    input()
