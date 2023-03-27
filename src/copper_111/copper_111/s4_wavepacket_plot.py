from surface_potential_analysis.eigenstate.eigenstate import EigenstateConfigUtil
from surface_potential_analysis.energy_eigenstate import (
    load_energy_eigenstates,
    normalize_eigenstate_phase,
)
from surface_potential_analysis.energy_eigenstates_plot import plot_eigenstate_positions
from surface_potential_analysis.wavepacket_grid import (
    calculate_wavepacket_grid_fourier,
    calculate_wavepacket_grid_fourier_fourier,
)
from surface_potential_analysis.wavepacket_grid_plot import plot_wavepacket_grid_xy

from .surface_data import get_data_path, save_figure


def plot_wavepacket_points():
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    fig, _, _ = plot_eigenstate_positions(eigenstates)

    fig.show()

    input()


def compare_double_fourier_wavepacket():
    """
    Test calculate_wavepacket_grid_fourier_fourier. Since the bloch wavefunctions are
    for -n/2 to n/2 rather than 0 to n-1 the output is wrong!
    """
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    origin = (0, 0, 0)
    normalized = normalize_eigenstate_phase(eigenstates, origin)

    grid1 = calculate_wavepacket_grid_fourier(normalized, [0.0])
    fig, ax, _ = plot_wavepacket_grid_xy(grid1, measure="abs")
    fig.show()
    fig, ax, _ = plot_wavepacket_grid_xy(grid1, measure="real")
    fig.show()
    fig, ax, _ = plot_wavepacket_grid_xy(grid1, measure="imag")
    fig.show()

    grid2 = calculate_wavepacket_grid_fourier_fourier(normalized, [0.0])
    fig, ax, _ = plot_wavepacket_grid_xy(grid2, measure="abs")
    fig.show()
    fig, ax, _ = plot_wavepacket_grid_xy(grid2, measure="real")
    fig.show()
    fig, ax, _ = plot_wavepacket_grid_xy(grid2, measure="imag")
    fig.show()
    input()


def plot_wavepacket_at_z_origin():
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    origin = (0, 0, 0)
    normalized = normalize_eigenstate_phase(eigenstates, origin)
    grid = calculate_wavepacket_grid_fourier(normalized, [0.0], (-4, 4), (-4, 4))

    fig, ax, _ = plot_wavepacket_grid_xy(grid, measure="abs")
    fig.show()
    ax.set_title("Plot of abs(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin.png")

    fig, ax, _ = plot_wavepacket_grid_xy(grid, measure="real")
    fig.show()
    ax.set_title("Plot of real(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_real.png")

    fig, ax, _ = plot_wavepacket_grid_xy(grid, measure="imag")
    fig.show()
    ax.set_title("Plot of imag(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_imag.png")

    path = get_data_path("eigenstates_grid_1.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    origin = (
        (util.delta_x0[0] + util.delta_x1[0]) / 3,
        (util.delta_x0[1] + util.delta_x1[1]) / 3,
        0,
    )
    normalized = normalize_eigenstate_phase(eigenstates, origin)

    grid = calculate_wavepacket_grid_fourier(normalized, [0.0], (-4, 4), (-4, 4))

    fig, ax, _ = plot_wavepacket_grid_xy(grid, measure="abs")
    fig.show()
    ax.set_title("Plot of abs(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin.png")

    fig, ax, _ = plot_wavepacket_grid_xy(grid, measure="real")
    fig.show()
    ax.set_title("Plot of real(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_real.png")

    fig, ax, _ = plot_wavepacket_grid_xy(grid, measure="imag")
    fig.show()
    ax.set_title("Plot of imag(wavefunction) for z=0")

    input()
