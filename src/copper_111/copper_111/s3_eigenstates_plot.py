from matplotlib import pyplot as plt

from surface_potential_analysis.eigenstate.eigenstate_plot import (
    animate_eigenstate_3D_in_xy,
)
from surface_potential_analysis.energy_eigenstate import (
    filter_eigenstates_band,
    get_eigenstate_list,
    load_energy_eigenstates,
)
from surface_potential_analysis.energy_eigenstates_plot import (
    plot_lowest_band_in_kx,
    plot_nth_band_in_kx,
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

    path = get_data_path("eigenstates_23_23_16.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,16)")

    path = get_data_path("eigenstates_21_21_16.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(21,21,16)")

    ax.legend()
    ax.set_title(
        "Plot of lowest band energies\n"
        "showing convergence for an eigenstate grid of (23,23,16)"
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


def plot_eigenstate_for_each_band():
    """
    Check to see if the eigenstates look as they are supposed to
    """
    path = get_data_path("eigenstates_29_29_12.json")
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
