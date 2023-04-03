from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.scale import FuncScale

from nickel_111.experiment_comparison import (
    plot_tunnelling_rate_baseline,
    plot_tunnelling_rate_jianding,
    plot_tunnelling_rate_theory,
)
from surface_potential_analysis.eigenstate_plot import (
    animate_eigenstate_3D_in_xy,
    plot_eigenstate_in_xy,
    plot_eigenstate_xy_hd,
)
from surface_potential_analysis.energy_eigenstate import (
    filter_eigenstates_band,
    get_eigenstate_list,
    load_energy_eigenstates,
)

from .surface_data import get_data_path


def plot_eigenstates():
    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates = load_energy_eigenstates(path)

    fig, axs = plt.subplots(1, 2)
    eigenstate = get_eigenstate_list(filter_eigenstates_band(eigenstates, n=0))[0]
    _, _, mesh = plot_eigenstate_xy_hd(
        eigenstates["eigenstate_config"], eigenstate, ax=axs[0]
    )
    mesh.set_clim(0)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].set_title("FCC")

    axs[1].sharey(axs[0])
    eigenstate = get_eigenstate_list(filter_eigenstates_band(eigenstates, n=1))[0]
    _, _, mesh = plot_eigenstate_xy_hd(
        eigenstates["eigenstate_config"], eigenstate, ax=axs[1]
    )
    mesh.set_clim(0)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["left"].set_visible(False)
    axs[1].yaxis.set_visible(False)
    axs[1].set_title("HCP")

    fig.colorbar(mesh, ax=axs, format="%4.1e", location="bottom")

    fig.show()

    input()


def plot_final_rates():
    fig, ax, container = plot_tunnelling_rate_jianding()

    container.set_label("Experimental tunnelling rate")
    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])
    ax.set_xscale(scale)
    ax.set_yscale("log")
    _, _, line = plot_tunnelling_rate_theory(ax)
    line.set_label("Corrected Rate")
    _, _, line = plot_tunnelling_rate_baseline(ax)
    line.set_label("Coherent Tunnelling Rate")
    ax.legend()
    ax.set_ylim(5e8, None)
    fig.show()
    input()
