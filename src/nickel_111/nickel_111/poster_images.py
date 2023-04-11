import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.scale import FuncScale

from nickel_111.experiment_comparison import (
    plot_tunnelling_rate_baseline,
    plot_tunnelling_rate_jianding,
    plot_tunnelling_rate_theory,
    plot_tunnelling_rate_theory_double,
)
from nickel_111.s1_potential import (
    load_interpolated_grid,
    load_raw_data_reciprocal_grid,
)
from nickel_111.s1_potential_plot import (
    plot_z_direction_energy_data_nickel_reciprocal_points,
)
from surface_potential_analysis.eigenstate_plot import (
    animate_eigenstate_3D_in_xy,
    plot_eigenstate_in_xy,
    plot_eigenstate_xy_hd,
)
from surface_potential_analysis.energy_data import normalize_energy
from surface_potential_analysis.energy_data_plot import plot_z_direction_energy_data_111
from surface_potential_analysis.energy_eigenstate import (
    filter_eigenstates_band,
    get_eigenstate_list,
    load_energy_eigenstates,
)
from surface_potential_analysis.overlap_transform import load_overlap_transform
from surface_potential_analysis.overlap_transform_plot import plot_overlap_transform_xy

from .surface_data import get_data_path, get_out_path, save_figure

colors = ["#D6083B", "#0072CF", "#EA7125", "#55A51C"]
# [1.0, 234 / 256, 234 / 256]
# [1.0, 113 / 256, 113 / 256]
# [1.0, 37.0 / 256, 37 / 256]

# [1.0, 85.0 / 256, 85 / 256]
# [1.0, 165 / 256, 165 / 256]
# [1.0, 28.0 / 256, 28 / 256]
cdict = {
    "red": [
        [0.0, 0.0 / 256, 0.0 / 256],
        [0.5, 85.0 / 256, 85 / 256],
        [1.0, 234 / 256, 234 / 256],
    ],
    "green": [
        [0.0, 114 / 256, 114 / 256],
        [0.5, 165 / 256, 165 / 256],
        [1.0, 113 / 256, 113 / 256],
    ],
    "blue": [
        [0.0, 207 / 256, 207 / 256],
        [0.5, 28.0 / 256, 28 / 256],
        [1.0, 37.0 / 256, 37 / 256],
    ],
}

newcmp = LinearSegmentedColormap("testCmap", segmentdata=cdict, N=256)


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


def plot_overlap_factor():
    path = get_data_path("overlap_transform_orthogonal_hcp_fcc.npz")
    # path = get_data_path("overlap_transform_interpolated_hcp_fcc.npz")
    # path = get_data_path("overlap_transform_extended_hcp_fcc.npz")
    overlap = load_overlap_transform(path)
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(dpi=1000)
    fig.set_figheight(6)
    fig.set_figwidth(6)
    # for item in (
    #     [ax.title, ax.xaxis.label, ax.yaxis.label]
    #     + ax.get_xticklabels()
    #     + ax.get_yticklabels()
    # ):
    #     item.set_fontsize(12)

    # overlap = make_transform_real_at(overlap, point=(1, 1, 0))
    _, _, mesh = plot_overlap_transform_xy(overlap, ax=ax)
    ax.set_xlabel("$K_x$ /$\\mathrm{m}^{-1}$")
    ax.set_ylabel("$K_y$ /$\\mathrm{m}^{-1}$")
    print(mesh.get_cmap()(0))
    ax.set_facecolor(mesh.get_cmap()(0))
    mesh.set_clim(0)
    save_figure(fig, "poster_overlap_plot.png")


def plot_eigenstates_fcc():
    matplotlib.rcParams.update({"font.size": 14})
    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates = load_energy_eigenstates(path)

    fig, ax = plt.subplots(dpi=1000)
    ax.set_xlabel("Distance / m")
    ax.set_ylabel("Distance / m")
    fig.set_figheight(5)  # 5
    fig.set_figwidth(7)  # 7
    eigenstate = get_eigenstate_list(filter_eigenstates_band(eigenstates, n=0))[0]
    _, _, mesh = plot_eigenstate_xy_hd(
        eigenstates["eigenstate_config"],
        eigenstate,
        ax=ax,
        shape=(400, 400),
    )
    mesh.set_clim(0)
    print(mesh.get_clim())
    ax.set_yticks([0, 0.5e-10, 1e-10, 1.5e-10, 2e-10])
    mesh.set_norm(SymLogNorm(linthresh=50000, linscale=1, base=10))
    # mesh.set_cmap(newcmp)
    # axs[0].spines["top"].set_visible(False)
    # axs[0].spines["right"].set_visible(False)
    # axs[0].set_title("FCC")
    # fig.colorbar(mesh, ax=ax, format="%4.1e", location="bottom")
    save_figure(fig, "poster_fcc_eigenstates.png")


def plot_final_rates():
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(dpi=1000)
    fig.set_figheight(6)
    fig.set_figwidth(8)

    fig, ax, container = plot_tunnelling_rate_jianding(ax=ax)
    container.set_label("Experimental Tunnelling Rate")
    (line, errors, b) = container.lines
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(colors[0])
    for e in errors:
        e.set_color(colors[0])
    for d in b:
        d.set_color(colors[0])

    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])
    ax.set_xscale(scale)
    ax.set_yscale("log")
    _, _, line = plot_tunnelling_rate_theory(ax)
    line.set_label("Corrected Rate")
    line.set_color(colors[1])

    _, _, line = plot_tunnelling_rate_theory_double(ax)
    line.set_label("3x Corrected Rate")
    line.set_color(colors[2])
    line.set_linestyle("dotted")

    _, _, line = plot_tunnelling_rate_baseline(ax)
    line.set_label("High Temperature Fit")
    line.set_linestyle("--")
    line.set_color(colors[2])
    ax.legend()
    ax.set_ylim(5e8, None)
    ax.set_ylabel("Rate /s")
    ax.set_xlabel("1/Temperature 1/$\\mathrm{K}^{-1}$")
    save_figure(fig, "poster_final_rates")


def plot_fcc_hcp_z_interpolation():
    interpolation = load_interpolated_grid()
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(dpi=1000)
    fig.set_figheight(4.5)  # 5
    fig.set_figwidth(6.3)  # 7

    _, _, old_lines = plot_z_direction_energy_data_111(interpolation, ax=ax)
    for i, (a) in enumerate(old_lines):
        a.set_color(colors[i])

    ax.set_ylim(0, 0.5e-18)
    raw_grid = normalize_energy(load_raw_data_reciprocal_grid())
    _, _, lines = plot_z_direction_energy_data_nickel_reciprocal_points(raw_grid, ax=ax)
    for ln in lines:
        ln.set_marker("x")
        ln.set_linestyle("")
    lines[0].set_color(colors[3])
    lines[1].set_color(colors[1])
    lines[2].set_color(colors[0])
    lines[3].set_color(colors[2])

    ax.legend(handles=old_lines)
    ax.set_xlabel("Height 1/ M")
    fig.subplots_adjust(bottom=0.2, top=0.8)
    save_figure(fig, "poster_z_interpolation.png")
