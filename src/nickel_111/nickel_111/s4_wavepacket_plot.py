import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.eigenstate.eigenstate import EigenstateConfigUtil
from surface_potential_analysis.eigenstate.plot import (
    animate_eigenstate_3D_in_xy,
    plot_eigenstate_in_yz,
    plot_eigenstate_x0z,
)
from surface_potential_analysis.energy_eigenstate import (
    EnergyEigenstatesLegacy,
    filter_eigenstates_band,
    get_brillouin_points_irreducible_config,
    get_eigenstate_list,
    load_energy_eigenstates,
    normalize_eigenstate_phase,
)
from surface_potential_analysis.energy_eigenstates_plot import plot_eigenstate_positions
from surface_potential_analysis.wavepacket_grid import (
    calculate_wavepacket_grid,
    calculate_wavepacket_grid_fourier,
)
from surface_potential_analysis.wavepacket_grid_plot import (
    animate_wavepacket_grid_3D_in_xy,
)

from .s2_hamiltonian import generate_hamiltonian
from .surface_data import get_data_path, save_figure


def plot_wavepacket_points():
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)

    fig, _, _ = plot_eigenstate_positions(eigenstates)
    fig.show()

    eigenstate_list = get_eigenstate_list(filter_eigenstates_band(eigenstates))
    fig, _, _ = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate_list[-1]
    )
    fig.show()
    input()


def select_single_k_eigenstates(
    eigenstates: EnergyEigenstatesLegacy, kx: float, ky: float
) -> EnergyEigenstatesLegacy:
    point_filter = np.logical_and(
        np.array(eigenstates["kx_points"]) == kx,
        np.array(eigenstates["ky_points"]) == ky,
    )
    out: EnergyEigenstatesLegacy = {
        "kx_points": np.array(eigenstates["kx_points"])[point_filter].tolist(),
        "ky_points": np.array(eigenstates["ky_points"])[point_filter].tolist(),
        "eigenstate_config": eigenstates["eigenstate_config"],
        "eigenvalues": np.array(eigenstates["eigenvalues"])[point_filter].tolist(),
        "eigenvectors": np.array(eigenstates["eigenvectors"])[point_filter].tolist(),
    }
    return out


def plot_wavepacket_points_in_yz():
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstate = get_eigenstate_list(eigenstates)[0]
    config = eigenstates["eigenstate_config"]

    fig, _, _anim1 = plot_eigenstate_in_yz(config, eigenstate, measure="real")
    fig.show()

    path = get_data_path("eigenstates_grid_1.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstate = get_eigenstate_list(eigenstates)[0]
    config = eigenstates["eigenstate_config"]

    fig, _, _anim2 = plot_eigenstate_in_yz(config, eigenstate, measure="real")
    fig.show()

    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstate = get_eigenstate_list(eigenstates)[0]
    config = eigenstates["eigenstate_config"]

    fig, _, _anim3 = plot_eigenstate_in_yz(config, eigenstate, measure="real")
    fig.show()

    input()


def plot_wavepacket_points_in_xz_from_list():
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstate = get_eigenstate_list(eigenstates)[0]
    config = eigenstates["eigenstate_config"]

    fig, _, _anim0 = plot_eigenstate_x0z(config, eigenstate, measure="real")
    fig.show()

    path = get_data_path("eigenstates_grid_1.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstate = get_eigenstate_list(eigenstates)[0]
    config = eigenstates["eigenstate_config"]

    fig, _, _anim1 = plot_eigenstate_x0z(config, eigenstate, measure="real")
    fig.show()

    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstate = get_eigenstate_list(eigenstates)[0]
    config = eigenstates["eigenstate_config"]

    fig, _, _anim2 = plot_eigenstate_x0z(config, eigenstate, measure="real")
    fig.show()

    path = get_data_path("eigenstates_grid_3.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstate = get_eigenstate_list(eigenstates)[0]
    config = eigenstates["eigenstate_config"]

    fig, _, _anim3 = plot_eigenstate_x0z(config, eigenstate, measure="real")
    fig.show()
    input()


def calculate_wavepacket_grid_nickel_111(eigenstates: EnergyEigenstatesLegacy):
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])

    return calculate_wavepacket_grid(
        eigenstates,
        util.delta_x0,
        util.delta_x1,
        np.linspace(-util.characteristic_z * 2, util.characteristic_z * 2, 11).tolist(),
        shape=(37, 25),
        offset=(0.0, 0.0),
    )


def get_value_at_point(
    eigenstates: EnergyEigenstatesLegacy, point: tuple[float, float, float]
):
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    return sum(
        util.calculate_wavefunction_fast(e, [point])[0]
        for e in get_eigenstate_list(eigenstates)
    )


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


def plot_wavepacket_points_me():
    hamiltonian = generate_hamiltonian()
    points = get_brillouin_points_irreducible_config(hamiltonian._config, size=(2, 2))
    fig, ax = plt.subplots()
    (line,) = ax.plot(points[:, 0], points[:, 1])
    line.set_linestyle("")
    line.set_marker("x")
    ax.set_title("Plot of points as chosen by Me")
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("ky")
    ax.set_xlabel("kx")
    fig.show()
    input()
    save_figure(fig, "my_wavepacket_points.png")


def plot_wavepacket_grid_all_equal():
    """
    Does the imaginary oscillation in the imaginary part of the wavefunction happen
    if we choose a constant bloch wavefunction for all k
    """
    path = get_data_path(f"eigenstates_grid_{0}.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    eigenvector = np.zeros(np.prod(util.resolution))
    eigenvector[util.get_index(0, 0, 0)] = 1
    eigenstates["eigenvectors"] = [
        eigenvector.tolist() for _ in eigenstates["eigenvectors"]
    ]

    z_points = [0.0]
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points, x0_lim=(0, 10), x1_lim=(0, 10)
    )

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(grid, norm="symlog")
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="real"
    )
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="imag"
    )
    fig.show()
    input()


def plot_wavepacket_grid():
    path = get_data_path(f"eigenstates_grid_{0}.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstates = normalize_eigenstate_phase(eigenstates, (0, 0, 0))

    z_points = [0.0]
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points, x0_lim=(0, 10), x1_lim=(0, 10)
    )

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(grid, norm="symlog")
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="real"
    )
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="imag"
    )
    fig.show()
    input()

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    path = get_data_path(f"eigenstates_grid_{1}.json")
    eigenstates = load_energy_eigenstates(path)
    eigenstates = normalize_eigenstate_phase(
        eigenstates,
        (
            (util.delta_x0[0] + util.delta_x1[0]) / 3,
            (util.delta_x0[1] + util.delta_x1[1]) / 3,
            0,
        ),
    )

    z_points = [0.0]
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points, x0_lim=(0, 10), x1_lim=(0, 10)
    )

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(grid, norm="symlog")
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="real"
    )
    fig.show()

    fig, _, _anim0 = animate_wavepacket_grid_3D_in_xy(
        grid, norm="symlog", measure="imag"
    )
    fig.show()
    input()
