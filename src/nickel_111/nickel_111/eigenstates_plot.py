import matplotlib.pyplot as plt

from surface_potential_analysis.energy_eigenstate import (
    get_eigenstate_list,
    load_energy_eigenstates,
)
from surface_potential_analysis.plot_eigenstate import plot_eigenstate_3D
from surface_potential_analysis.plot_energy_eigenstates import plot_lowest_band_in_kx

from .surface_data import get_data_path, save_figure


def analyze_band_convergence():
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_12_12_13.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,13)")

    path = get_data_path("eigenstates_12_12_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,14)")

    path = get_data_path("eigenstates_10_10_13.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(10,10,13)")

    path = get_data_path("eigenstates_14_14_13.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(14,14,13)")

    path = get_data_path("eigenstates_12_12_12.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,12)")

    path = get_data_path("eigenstates_15_15_12.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(15,15,12)")

    path = get_data_path("eigenstates_12_18_12.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,18,12)")

    ax.legend()
    ax.set_title(
        "Plot of lowest band energies\n"
        "showing convergence for an eigenstate grid of (15,15,12)"
    )

    fig.show()
    input()
    save_figure(fig, "lowest_band_convergence.png")


def plot_eigenstate():
    path = get_data_path("eigenstates_12_18_12.json")
    eigenstates = load_energy_eigenstates(path)

    eigenstate = get_eigenstate_list(eigenstates)[0]
    fig, _, _anim1 = plot_eigenstate_3D(eigenstates["eigenstate_config"], eigenstate)

    fig.show()

    path = get_data_path("eigenstates_15_15_12.json")
    eigenstates = load_energy_eigenstates(path)

    eigenstate = get_eigenstate_list(eigenstates)[0]
    fig, _, _anim2 = plot_eigenstate_3D(eigenstates["eigenstate_config"], eigenstate)

    fig.show()
    input()
