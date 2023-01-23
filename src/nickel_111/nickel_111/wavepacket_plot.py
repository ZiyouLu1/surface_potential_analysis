import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from nickel_111.hamiltonian import generate_hamiltonian
from nickel_111.wavepacket import get_brillouin_points_nickel_111
from surface_potential_analysis.energy_eigenstate import (
    EigenstateConfigUtil,
    EnergyEigenstates,
    EnergyEigenstatesRaw,
    get_eigenstate_list,
    load_energy_eigenstates,
    normalize_eigenstate_phase,
)
from surface_potential_analysis.plot_eigenstate import (
    plot_eigenstate_3D,
    plot_eigenstate_in_xy,
)
from surface_potential_analysis.plot_energy_eigenstates import plot_eigenstate_positions
from surface_potential_analysis.wavepacket_grid import (
    calculate_wavepacket_grid_copper,
    load_wavepacket_grid,
    save_wavepacket_grid,
)
from surface_potential_analysis.wavepacket_grid_plot import (
    plot_wavepacket_grid_y_2D,
    plot_wavepacket_grid_z_2D,
)

from .surface_data import get_data_path, save_figure


def load_energy_eigenstates_list(path: Path) -> List[EnergyEigenstates]:
    with path.open("r") as f:
        out: List[EnergyEigenstatesRaw] = json.load(f)

        return [
            {
                "eigenstate_config": d["eigenstate_config"],
                "eigenvalues": d["eigenvalues"],
                "eigenvectors": (
                    np.array(d["eigenvectors_re"]) + 1j * np.array(d["eigenvectors_im"])
                ).tolist(),
                "kx_points": d["kx_points"],
                "ky_points": d["ky_points"],
            }
            for d in out
        ]


def plot_wavepacket_points():
    path = get_data_path("eigenstates_grid.json")
    eigenstates = load_energy_eigenstates(path)
    fig, _, _ = plot_eigenstate_positions(eigenstates)

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates)
    fig, _, _ = plot_eigenstate_3D(
        eigenstates["eigenstate_config"], eigenstate_list[-1]
    )

    fig.show()
    input()


def plot_wavepacket_points_from_list():
    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = load_energy_eigenstates_list(path)
    fig, _, _ = plot_eigenstate_positions(eigenstates[0])

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates[0])
    fig, _, _anim1 = plot_eigenstate_3D(
        eigenstates[0]["eigenstate_config"], eigenstate_list[-1], measure="real"
    )

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates[1])
    fig, _, _anim2 = plot_eigenstate_3D(
        eigenstates[1]["eigenstate_config"], eigenstate_list[-1], measure="real"
    )

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates[2])
    fig, _, _anim3 = plot_eigenstate_3D(
        eigenstates[2]["eigenstate_config"], eigenstate_list[-1], measure="real"
    )

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates[3])
    fig, _, _anim4 = plot_eigenstate_3D(
        eigenstates[3]["eigenstate_config"], eigenstate_list[-1], measure="real"
    )

    fig.show()
    input()


def plot_wavepacket_points_at_z_origin_from_list():
    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = load_energy_eigenstates_list(path)
    fig, _, _ = plot_eigenstate_positions(eigenstates[0])

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates[0])
    fig, _, _anim1 = plot_eigenstate_in_xy(
        eigenstates[0]["eigenstate_config"], eigenstate_list[-1], measure="real"
    )

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates[1])
    fig, _, _anim2 = plot_eigenstate_in_xy(
        eigenstates[1]["eigenstate_config"], eigenstate_list[-1], measure="real"
    )

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates[2])
    fig, _, _anim3 = plot_eigenstate_in_xy(
        eigenstates[2]["eigenstate_config"], eigenstate_list[-1], measure="real"
    )

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates[3])
    fig, _, _anim4 = plot_eigenstate_in_xy(
        eigenstates[3]["eigenstate_config"], eigenstate_list[-1], measure="real"
    )

    fig.show()
    input()


def plot_energy_of_first_bands():
    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = load_energy_eigenstates_list(path)

    eigenvalues = [x["eigenvalues"][0] for x in eigenstates]
    fig, ax = plt.subplots()
    (line,) = ax.plot(eigenvalues)
    line.set_marker("x")
    line.set_linestyle("")
    ax.set_title(
        "Plot of the energy of the first 10 bands in the super-lattuice\n"
        "showing a jump after the first 4 states"
    )

    fig.show()
    save_figure(fig, "nickel_super_lattice_energy_states.png")
    input()


def select_lowest_band_eigenstates(
    eigenstates: List[EnergyEigenstates], origin_point: Tuple[float, float, float]
) -> EnergyEigenstates:

    out: EnergyEigenstates = {
        "kx_points": [eigenstates[i]["kx_points"][0] for i in range(4)],
        "ky_points": [eigenstates[i]["ky_points"][0] for i in range(4)],
        "eigenstate_config": eigenstates[0]["eigenstate_config"],
        "eigenvalues": [eigenstates[i]["eigenvalues"][0] for i in range(4)],
        "eigenvectors": [eigenstates[i]["eigenvectors"][0] for i in range(4)],
    }
    return out


def test_single_kpoint_wavepacket():
    """
    Figuring out how to produce a 'localised' wavepacket from a single k-point
    ie only one site in the super cell
    """

    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = load_energy_eigenstates_list(path)

    util = EigenstateConfigUtil(eigenstates[0]["eigenstate_config"])

    # Note we don't need to worry about 'repeats'
    # Ie at (1.0 * util.delta_x / 3, util.delta_y, 0),
    origins = [
        (0, 1.0 * util.delta_y / 3, 0),
        (0, 2.0 * util.delta_y / 3, 0),
        (util.delta_x / 2, 0.5 * util.delta_y / 3, 0),
        (util.delta_x / 2, 2.5 * util.delta_y / 3, 0),
    ]

    lowest_band = select_lowest_band_eigenstates(eigenstates, origin_point=origins[0])

    path = get_data_path("single_band_wavepacket_1.json")
    normalized1 = normalize_eigenstate_phase(lowest_band, origins[0])
    grid1 = calculate_wavepacket_grid_copper(normalized1)
    save_wavepacket_grid(grid1, path)
    grid1 = load_wavepacket_grid(path)

    path = get_data_path("single_band_wavepacket_2.json")
    normalized2 = normalize_eigenstate_phase(lowest_band, origins[1])
    grid2 = calculate_wavepacket_grid_copper(normalized2)
    save_wavepacket_grid(grid2, path)
    grid2 = load_wavepacket_grid(path)

    fig, ax, _anim1 = plot_wavepacket_grid_z_2D(grid1, norm="linear")

    fig.show()
    fig, ax, _anim2 = plot_wavepacket_grid_y_2D(grid1, norm="linear")
    fig.show()

    fig, ax, _anim3 = plot_wavepacket_grid_z_2D(grid2, norm="linear")
    fig.show()
    input()


def plot_wavepacket():
    path = get_data_path("eigenstates_wavepacket_0.json")
    grid = load_wavepacket_grid(path)
    fig, ax, _anim0 = plot_wavepacket_grid_z_2D(grid, norm="symlog")
    fig.show()

    # path = get_data_path("eigenstates_wavepacket_1.json")
    # grid = load_wavepacket_grid(path)
    # fig, ax, _anim1 = plot_wavepacket_grid_z_2D(grid, norm="linear")
    # fig.show()

    # path = get_data_path("eigenstates_wavepacket_2.json")
    # grid = load_wavepacket_grid(path)
    # fig, ax, _anim2 = plot_wavepacket_grid_z_2D(grid, norm="linear")
    fig.show()

    input()


# dkx = hamiltonian.dkx
#     (kx_points, kx_step) = np.linspace(
#         -dkx / 2, dkx / 2, 2 * grid_size, endpoint=False, retstep=True
#     )
#     dky = hamiltonian.dky
#     (ky_points, ky_step) = np.linspace(
#         -dky / 2, dky / 2, 2 * grid_size, endpoint=False, retstep=True
#     )
#     if not include_zero:
#         kx_points += kx_step / 2
#         ky_points += ky_step / 2

#     xv, yv = np.meshgrid(kx_points, ky_points)
#     k_points = np.array([xv.ravel(), yv.ravel()]).T
#     return k_points


def plot_wavepacket_points_john():

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
    G1 = 2.9419
    nqdim = 4

    qx_points = G1 * ((np.sqrt(3) / 2) * a[:, 0] + 0 * a[:, 1]) / nqdim
    qy_points = G1 * ((1 / 2) * a[:, 0] + a[:, 1]) / nqdim

    fig, ax = plt.subplots()
    (line,) = ax.plot(qx_points, qy_points)
    line.set_linestyle("")
    line.set_marker("x")
    ax.set_title("Plot of points as chosen by John")
    fig.show()
    input()
    save_figure(fig, "john_wavepacket_points.png")


def plot_wavepacket_points_me():
    hamiltonian = generate_hamiltonian()
    points = get_brillouin_points_nickel_111(hamiltonian._config, grid_size=2)
    fig, ax = plt.subplots()
    (line,) = ax.plot(points[:, 0], points[:, 1])
    line.set_linestyle("")
    line.set_marker("x")
    ax.set_title("Plot of points as chosen by Me")
    fig.show()
    input()
    save_figure(fig, "my_wavepacket_points.png")
