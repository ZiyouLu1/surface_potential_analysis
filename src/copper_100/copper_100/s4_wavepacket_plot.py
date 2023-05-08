from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from surface_potential_analysis.eigenstate.plot import (
    plot_eigenstate_difference_2d_x,
    plot_eigenstate_x0x1,
)
from surface_potential_analysis.wavepacket.normalization import normalize_wavepacket
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_k0k1,
    plot_wavepacket_sample_frequencies,
    plot_wavepacket_x0,
    plot_wavepacket_x0x1,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_eigenstate,
    load_wavepacket,
)

from copper_100.s4_wavepacket import load_copper_wavepacket

from .surface_data import get_data_path, save_figure


def plot_wavepacket_points() -> None:
    wavepacket = load_copper_wavepacket(0)
    fig, _, _ = plot_wavepacket_sample_frequencies(wavepacket)

    fig.show()

    input()


def plot_wavepacket_at_z_origin() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    wavepacket = load_wavepacket(path)
    normalized = normalize_wavepacket(wavepacket)

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 0, measure="abs")
    fig.show()
    ax.set_title("Plot of abs(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin.png")

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 0, measure="real")
    fig.show()
    ax.set_title("Plot of real(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_real.png")

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 0, measure="imag")
    fig.show()
    ax.set_title("Plot of imag(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_imag.png")
    input()


def plot_wavepacket_3d_x() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    path = get_data_path("eigenstates_grid_1.json")
    eigenstates = load_wavepacket(path)
    normalized = normalize_wavepacket(eigenstates)

    fig, _, _ = animate_wavepacket_x0x1(normalized)
    fig.show()
    input()
    fig, _, _ = animate_wavepacket_x0x1(normalized)
    fig.show()
    input()


def plot_ft_hd_wavepacket_at_origin() -> None:
    path = get_data_path("relaxed_eigenstates_hd_wavepacket_flat.json")
    wavepacket = load_wavepacket(path)
    fig, _, _ = plot_wavepacket_x0x1(wavepacket, x2_idx=1, measure="real")
    fig.show()
    fig, _, _ = plot_wavepacket_k0k1(wavepacket, k2_idx=1, measure="real")


def compare_wavefunction_4_8_points_log() -> None:
    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket_8 = load_wavepacket(path)

    path = get_data_path("copper_eigenstates_wavepacket_4_point.json")
    wavepacket_4 = load_wavepacket(path)

    path = get_data_path("copper_eigenstates_wavepacket_flat_band.json")
    wavepacket_1 = load_wavepacket(path)

    fig, ax = plt.subplots()
    _, _, l2 = plot_wavepacket_x0(wavepacket_4, (48, 10), ax=ax)
    l2.set_label("4 point grid")
    _, _, l4 = plot_wavepacket_x0(wavepacket_8, (48, 10), ax=ax)
    l4.set_label("8 point grid")
    _, _, l5 = plot_wavepacket_x0(wavepacket_1, (48, 10), ax=ax)
    l5.set_label("1 point grid")

    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Log plot of the abs 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_abs_comparison.png")

    fig, ax = plt.subplots()
    _, _, l2 = plot_wavepacket_x0(wavepacket_4, (48, 10), ax=ax, measure="real")
    l2.set_label("4 point grid")
    _, _, l4 = plot_wavepacket_x0(wavepacket_8, (48, 10), ax=ax, measure="real")
    l4.set_label("8 point grid")
    _, _, l5 = plot_wavepacket_x0(wavepacket_1, (48, 10), ax=ax, measure="real")
    l5.set_label("1 point grid")

    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Log plot of the real part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_real_comparison.png")


def compare_wavefunction_4_8_points() -> None:
    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket_8 = load_wavepacket(path)

    path = get_data_path("copper_eigenstates_wavepacket_4_point.json")
    wavepacket_4 = load_wavepacket(path)

    path = get_data_path("copper_eigenstates_wavepacket_flat_band.json")
    wavepacket_1 = load_wavepacket(path)

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_x0(wavepacket_4, (48, 10), ax=ax, measure="imag")
    l1.set_label("4 point grid")
    _, _, l2 = plot_wavepacket_x0(wavepacket_8, (48, 10), ax=ax, measure="imag")
    l2.set_label("8 point grid")
    _, _, l3 = plot_wavepacket_x0(wavepacket_1, (48, 10), ax=ax, measure="imag")
    l3.set_label("1 point grid")

    ax.legend()
    ax.set_yscale("linear")
    ax.set_title("Imaginary part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_imag_comparison.png")

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_x0(wavepacket_4, (48, 12), ax=ax, measure="abs")
    l1.set_label("4 point grid")
    _, _, l2 = plot_wavepacket_x0(wavepacket_8, (48, 12), ax=ax, measure="abs")
    l2.set_label("8 point grid")
    _, _, l3 = plot_wavepacket_x0(wavepacket_1, (48, 12), ax=ax, measure="abs")
    l3.set_label("1 point grid")
    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Abs part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_abs_comparison_max_height.png")
    input()


def compare_wavefunction_eigenstate_2d() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    wavepacket = load_wavepacket(path)
    normalized = normalize_wavepacket(wavepacket)

    (ns0, ns1) = wavepacket["energies"].shape
    eigenstate_0 = get_eigenstate(normalized, (ns0 // 2, ns1 // 2))
    eigenstate_1 = get_eigenstate(normalized, (0, 0))
    eigenstate_2 = get_eigenstate(normalized, (ns0 // 2, 0))

    fig, axs = plt.subplots(2, 3)
    (_, ax, _) = plot_eigenstate_x0x1(eigenstate_0, 0, ax=axs[0][0])
    ax.set_title("(-dkx/2, -dky/2) at z=0")
    (_, ax, _) = plot_eigenstate_x0x1(eigenstate_1, 0, ax=axs[0][1])
    ax.set_title("(0,0) at z=0")
    (_, ax, _) = plot_eigenstate_x0x1(eigenstate_2, 0, ax=axs[0][2])
    ax.set_title("(-dkx/2, 0) at z=0")

    (_, ax, _) = plot_eigenstate_x0x1(eigenstate_0, x2_idx=100, ax=axs[1][0])
    ax.set_title("(-dkx/2, -dky/2) at z=delta_x")
    (_, ax, _) = plot_eigenstate_x0x1(eigenstate_1, x2_idx=100, ax=axs[1][1])
    ax.set_title("(0,0) at z=delta_x")
    (_, ax, _) = plot_eigenstate_x0x1(eigenstate_2, x2_idx=100, ax=axs[1][2])
    ax.set_title("(-dkx/2, 0) at z=delta_x")

    fig.tight_layout()
    fig.suptitle("Plot of absolute value of the Bloch wavefunctions")
    save_figure(fig, "Center and middle wavefunctions 2D")
    fig.show()

    fig, axs = plt.subplots(1, 2)
    (_, ax, _) = plot_eigenstate_difference_2d_x(eigenstate_1, eigenstate_0, 2, axs[0])
    ax.set_title("(-dkx/2, -dky/2) vs (0,0)")
    (_, ax, _) = plot_eigenstate_difference_2d_x(eigenstate_2, eigenstate_0, 2, axs[1])
    ax.set_title("(-dkx/2, 0) vs (0,0)")

    fig.suptitle("Plot of difference in the absolute value of the Bloch wavefunctions")
    fig.show()
    fig.tight_layout()
    save_figure(fig, "Center wavefunction diff 2D")
    input()


# How different are the bloch wavefunctions
def calculate_eigenstate_cross_product() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_wavepacket(path)
    normalized = normalize_wavepacket(eigenstates)

    (ns0, ns1) = normalized["energies"].shape
    eigenstate_0 = get_eigenstate(normalized, (ns0 // 2, ns1 // 2))
    eigenstate_1 = get_eigenstate(normalized, (0, 0))

    prod = np.multiply(eigenstate_0["vector"], np.conjugate(eigenstate_1["vector"]))
    print(prod)  # noqa: T201
    norm: np.float_ = np.sum(prod)
    print(norm)  # 0.95548 # noqa: T201
