import matplotlib.pyplot as plt

from surface_potential_analysis.eigenstate_plot import (
    animate_eigenstate_3D_in_xy,
    plot_eigenstate_through_bridge,
    plot_eigenstate_z,
)
from surface_potential_analysis.energy_eigenstate import (
    filter_eigenstates_band,
    get_eigenstate_list,
    load_energy_eigenstates,
    normalize_eigenstate_phase,
)
from surface_potential_analysis.energy_eigenstates_plot import (
    plot_lowest_band_in_kx,
    plot_nth_band_in_kx,
)

from .surface_data import get_data_path, save_figure


def analyze_eigenvalue_convergence():

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_25_25_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(25,25,14)")

    path = get_data_path("eigenstates_23_23_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,14)")

    path = get_data_path("eigenstates_23_23_15.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,15)")

    path = get_data_path("eigenstates_23_23_16.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,16)")

    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(25,25,16)")

    path = get_data_path("eigenstates_23_23_17.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,17)")

    path = get_data_path("eigenstates_23_23_18.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,18)")

    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.set_xlabel("K /$m^{-1}$")
    ax.set_ylabel("energy / J")
    ax.legend()

    fig.tight_layout()
    fig.show()
    save_figure(fig, "copper_lowest_band_convergence.png")
    input()


def analyze_eigenvalue_convergence_relaxed():

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_relaxed_17_17_15.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(17,17,15)")

    path = get_data_path("eigenstates_relaxed_21_21_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(21,21,14)")

    path = get_data_path("eigenstates_relaxed_21_21_15.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(21,21,15)")

    path = get_data_path("eigenstates_relaxed_17_17_13.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(17,17,13)")

    ax.set_title(
        "Plot of energy against kx for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.set_xlabel("K /$m^{-1}$")
    ax.set_ylabel("energy / J")
    ax.legend()

    fig.tight_layout()
    fig.show()
    save_figure(fig, "lowest_band_convergence.png")

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_relaxed_10_10_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_nth_band_in_kx(eigenstates, n=4, ax=ax)
    ln.set_label("(10,10,14)")

    path = get_data_path("eigenstates_relaxed_12_12_15.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_nth_band_in_kx(eigenstates, n=4, ax=ax)
    ln.set_label("(12,12,15)")

    ax.set_title(
        "Plot of energy against kx for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.set_xlabel("K /$m^{-1}$")
    ax.set_ylabel("energy / J")
    ax.legend()

    fig.tight_layout()
    fig.show()
    save_figure(fig, "second_band_convergence.png")
    input()


def plot_lowest_eigenstate_3D_xy():
    path = get_data_path("eigenstates_relaxed_10_10_14.json")
    eigenstates = load_energy_eigenstates(path)

    eigenstate = get_eigenstate_list(filter_eigenstates_band(eigenstates, n=0))[-1]

    fig, _, _anim = animate_eigenstate_3D_in_xy(
        eigenstates["eigenstate_config"], eigenstate, measure="real"
    )
    fig.show()
    input()


def analyze_eigenvector_convergence_not_relaxed_z():

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_25_25_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_eigenstate_z(
        eigenstates["eigenstate_config"], get_eigenstate_list(eigenstates)[0], ax=ax
    )
    ln.set_label("(25,25,14) kx=G/2")

    path = get_data_path("eigenstates_23_23_15.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, l2 = plot_eigenstate_z(
        eigenstates["eigenstate_config"], get_eigenstate_list(eigenstates)[0], ax=ax
    )
    l2.set_label("(23,23,15) kx=G/2")

    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.legend()
    fig.show()
    save_figure(fig, "copper_wfn_convergence.png")
    input()


def analyze_eigenvector_convergence_not_relaxed_through_bridge():

    path = get_data_path("eigenstates_25_25_14.json")
    eigenstates = load_energy_eigenstates(path)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    path = get_data_path("eigenstates_23_23_15.json")
    eigenstates = load_energy_eigenstates(path)
    normalized = normalize_eigenstate_phase(eigenstates)
    _, _, ln = plot_eigenstate_through_bridge(
        eigenstates["eigenstate_config"], get_eigenstate_list(normalized)[5], ax=ax
    )
    _, _, _ = plot_eigenstate_through_bridge(
        eigenstates["eigenstate_config"],
        get_eigenstate_list(eigenstates)[5],
        ax=ax2,
        view="angle",
    )
    ln.set_label("(23,23,15)")

    path = get_data_path("eigenstates_25_25_14.json")
    eigenstates = load_energy_eigenstates(path)
    normalized = normalize_eigenstate_phase(eigenstates)
    _, _, ln = plot_eigenstate_through_bridge(
        eigenstates["eigenstate_config"], get_eigenstate_list(normalized)[5], ax=ax
    )
    _, _, _ = plot_eigenstate_through_bridge(
        eigenstates["eigenstate_config"],
        get_eigenstate_list(eigenstates)[5],
        ax=ax2,
        view="angle",
    )
    ln.set_label("(25,25,15)")

    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.legend()
    fig.show()
    save_figure(fig, "copper_wfn_convergence_through_bridge.png")
    input()
