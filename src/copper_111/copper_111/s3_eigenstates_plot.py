from matplotlib import pyplot as plt

from surface_potential_analysis.eigenstate_plot import animate_eigenstate_3D_in_xy
from surface_potential_analysis.energy_eigenstate import (
    filter_eigenstates_band,
    get_eigenstate_list,
    load_energy_eigenstates,
)
from surface_potential_analysis.energy_eigenstates_plot import plot_lowest_band_in_kx

from .surface_data import get_data_path, save_figure


def analyze_eigenvalue_convergence():

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_23_23_10.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,10)")

    path = get_data_path("eigenstates_23_23_12.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,12)")

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


def plot_eigenstate_for_each_band():
    """
    Check to see if the eigenstates look as they are supposed to
    """
    path = get_data_path("eigenstates_23_23_12.json")
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
    input()
