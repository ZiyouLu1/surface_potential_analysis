from matplotlib import pyplot as plt

from surface_potential_analysis.eigenstate_plot import animate_eigenstate_3D_in_xy
from surface_potential_analysis.energy_eigenstate import (
    get_eigenstate_list,
    load_energy_eigenstates,
)
from surface_potential_analysis.energy_eigenstates_plot import plot_lowest_band_in_kx

from .surface_data import get_data_path, save_figure


def analyze_eigenvalue_convergence():

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_14_14_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(14,14,14)")

    path = get_data_path("eigenstates_13_13_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(13,13,14)")

    path = get_data_path("eigenstates_15_15_13.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(15,15,13)")

    path = get_data_path("eigenstates_15_15_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(15,15,14)")

    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing no convergence even for a (15,15,14) grid"
    )
    ax.set_xlabel("K /$m^{-1}$")
    ax.set_ylabel("energy / J")
    ax.legend()

    fig.tight_layout()
    fig.show()
    save_figure(fig, "lowest_band_convergence.png")
    input()


def plot_lowest_eigenstate_3D_xy():
    path = get_data_path("eigenstates_15_15_14.json")
    eigenstates = load_energy_eigenstates(path)

    eigenstate = get_eigenstate_list(eigenstates)[-1]

    fig, _, _anim = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate
    )
    fig.show()
    input()
