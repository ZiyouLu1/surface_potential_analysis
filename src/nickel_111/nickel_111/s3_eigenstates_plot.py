import matplotlib.pyplot as plt
import numpy as np

from surface_potential_analysis.eigenstate_plot import animate_eigenstate_3D_in_xy
from surface_potential_analysis.energy_eigenstate import (
    EnergyEigenstates,
    filter_eigenstates_band,
    get_eigenstate_list,
    load_energy_eigenstates,
    load_energy_eigenstates_legacy,
)
from surface_potential_analysis.energy_eigenstates_plot import (
    plot_lowest_band_in_kx,
    plot_nth_band_in_kx,
)
from surface_potential_analysis.wavepacket_grid import calculate_wavepacket_grid
from surface_potential_analysis.wavepacket_grid_plot import (
    animate_wavepacket_grid_3D_in_xy,
)

from .surface_data import get_data_path, save_figure


def analyze_band_convergence():
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_23_23_10.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,10)")

    path = get_data_path("eigenstates_23_23_12.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,12)")

    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(25,25,16)")

    ax.legend()
    ax.set_title(
        "Plot of lowest band energies\n"
        "showing convergence for an eigenstate grid of (15,15,12)"
    )

    fig.show()
    save_figure(fig, "lowest_band_convergence.png")

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_nth_band_in_kx(eigenstates, n=0, ax=ax)
    ln.set_label("n=0")
    _, _, ln = plot_nth_band_in_kx(eigenstates, n=1, ax=ax)
    ln.set_label("n=1")

    ax.legend()
    fig.show()
    save_figure(fig, "two_lowest_bands.png")

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_23_23_12.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_nth_band_in_kx(eigenstates, n=1, ax=ax)
    ln.set_label("(23,23,12)")

    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_nth_band_in_kx(eigenstates, n=1, ax=ax)
    ln.set_label("(25,25,16)")

    ax.legend()
    fig.show()
    save_figure(fig, "second_band_convergence.png")

    input()


def analyze_band_convergence_kx1():
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_25_25_10_kx1.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(25,25,10)")

    # path = get_data_path("eigenstates_25_25_12_kx1.json")
    # eigenstates = load_energy_eigenstates(path)
    # _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    # ln.set_label("(25,25,12)")

    # path = get_data_path("eigenstates_27_27_12.json")
    # eigenstates = load_energy_eigenstates(path)
    # _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    # ln.set_label("(27,27,12)")

    # path = get_data_path("eigenstates_29_29_10.json")
    # eigenstates = load_energy_eigenstates(path)
    # _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    # ln.set_label("(29,29,10)")

    ax.legend()
    ax.set_title(
        "Plot of lowest band energies\n"
        "showing convergence for an eigenstate grid of (15,15,12)"
    )

    fig.show()
    input()
    save_figure(fig, "lowest_band_convergence.png")


def analyze_band_convergence_john():
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_john_12_12_13.json")
    eigenstates = load_energy_eigenstates_legacy(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,13)")

    path = get_data_path("eigenstates_john_12_12_14.json")
    eigenstates = load_energy_eigenstates_legacy(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,14)")

    path = get_data_path("eigenstates_john_10_10_13.json")
    eigenstates = load_energy_eigenstates_legacy(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(10,10,13)")

    path = get_data_path("eigenstates_john_14_14_13.json")
    eigenstates = load_energy_eigenstates_legacy(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(14,14,13)")

    path = get_data_path("eigenstates_john_12_12_12.json")
    eigenstates = load_energy_eigenstates_legacy(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,12)")

    # Corrupted data
    # path = get_data_path("eigenstates_john_15_15_12.json")
    # eigenstates = load_energy_eigenstates_legacy(path)
    # _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    # ln.set_label("(15,15,12)")

    path = get_data_path("eigenstates_john_12_18_12.json")
    eigenstates = load_energy_eigenstates_legacy(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,18,12)")

    ax.legend()
    ax.set_title(
        "Plot of lowest band energies\n"
        "showing convergence for an eigenstate grid of (15,15,12)"
    )

    fig.show()
    input()
    save_figure(fig, "lowest_band_convergence_john.png")


def plot_eigenstate_for_each_band():
    """
    Check to see if the eigenstates look as they are supposed to

    Spoiler: they dont :(
    """
    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates = load_energy_eigenstates(path)

    eigenstate = get_eigenstate_list(filter_eigenstates_band(eigenstates, n=0))[0]
    fig, _, _anim1 = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate
    )
    fig.show()

    eigenstate = get_eigenstate_list(filter_eigenstates_band(eigenstates, n=1))[0]
    fig, _, _anim2 = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate
    )
    fig.show()

    eigenstate = get_eigenstate_list(filter_eigenstates_band(eigenstates, n=2))[0]
    fig, _, _anim3 = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate
    )
    fig.show()

    eigenstate = get_eigenstate_list(filter_eigenstates_band(eigenstates, n=3))[0]
    fig, _, _anim4 = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate
    )
    fig.show()

    eigenstate = get_eigenstate_list(filter_eigenstates_band(eigenstates, n=4))[0]
    fig, _, _anim5 = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate
    )
    fig.show()

    eigenstate = get_eigenstate_list(filter_eigenstates_band(eigenstates, n=9))[0]
    fig, _, _anim6 = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate
    )
    fig.show()
    input()


def plot_eigenstate_john():
    path = get_data_path("eigenstates_john_12_18_12.json")
    eigenstates = load_energy_eigenstates_legacy(path)

    eigenstate = get_eigenstate_list(eigenstates)[0]
    fig, _, _anim1 = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate
    )

    fig.show()

    path = get_data_path("eigenstates_john_15_15_12.json")
    eigenstates = load_energy_eigenstates_legacy(path)

    eigenstate = get_eigenstate_list(eigenstates)[0]
    fig, _, _anim2 = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate
    )

    fig.show()
    input()
