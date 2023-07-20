from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from surface_potential_analysis.state_vector.plot import (
    plot_eigenstate_2d_x,
    plot_state_vector_difference_2d_x,
)
from surface_potential_analysis.wavepacket.get_eigenstate import get_eigenstate
from surface_potential_analysis.wavepacket.localization import (
    localize_tightly_bound_wavepacket_idx,
)
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_sample_frequencies,
    plot_wavepacket_x0x1,
)

from .s4_wavepacket import get_wavepacket_hydrogen
from .surface_data import save_figure


def plot_wavepacket_points() -> None:
    wavepacket = get_wavepacket_hydrogen(0)
    fig, _, _ = plot_wavepacket_sample_frequencies(wavepacket)

    fig.show()

    input()


def plot_wavepacket_at_z_origin() -> None:
    wavepacket = get_wavepacket_hydrogen(0)
    normalized = localize_tightly_bound_wavepacket_idx(wavepacket)

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
    wavepacket = get_wavepacket_hydrogen(0)
    normalized = localize_tightly_bound_wavepacket_idx(wavepacket)

    fig, _, _ = animate_wavepacket_x0x1(normalized)
    fig.show()
    input()
    fig, _, _ = animate_wavepacket_x0x1(normalized)
    fig.show()
    input()


def compare_wavefunction_eigenstate_2d() -> None:
    wavepacket = get_wavepacket_hydrogen(0)
    normalized = localize_tightly_bound_wavepacket_idx(wavepacket)

    (ns0, ns1, _) = wavepacket["eigenvalues"].shape
    eigenstate_0 = get_eigenstate(normalized, (ns0 // 2, ns1 // 2, 0))
    eigenstate_1 = get_eigenstate(normalized, (0, 0, 0))
    eigenstate_2 = get_eigenstate(normalized, (ns0 // 2, 0, 0))

    fig, axs = plt.subplots(2, 3)
    (_, ax, _) = plot_eigenstate_2d_x(eigenstate_0, (0, 1), (0,), ax=axs[0][0])
    ax.set_title("(-dkx/2, -dky/2) at z=0")
    (_, ax, _) = plot_eigenstate_2d_x(eigenstate_1, (0, 1), (0,), ax=axs[0][1])
    ax.set_title("(0,0) at z=0")
    (_, ax, _) = plot_eigenstate_2d_x(eigenstate_2, (0, 1), (0,), ax=axs[0][2])
    ax.set_title("(-dkx/2, 0) at z=0")

    (_, ax, _) = plot_eigenstate_2d_x(eigenstate_0, (0, 1), (100,), ax=axs[1][0])
    ax.set_title("(-dkx/2, -dky/2) at z=delta_x")
    (_, ax, _) = plot_eigenstate_2d_x(eigenstate_1, (0, 1), (100,), ax=axs[1][1])
    ax.set_title("(0,0) at z=delta_x")
    (_, ax, _) = plot_eigenstate_2d_x(eigenstate_2, (0, 1), (100,), ax=axs[1][2])
    ax.set_title("(-dkx/2, 0) at z=delta_x")

    fig.tight_layout()
    fig.suptitle("Plot of absolute value of the Bloch wavefunctions")
    save_figure(fig, "Center and middle wavefunctions 2D")
    fig.show()

    fig, axs = plt.subplots(1, 2)
    (_, ax, _) = plot_state_vector_difference_2d_x(
        eigenstate_1, eigenstate_0, (0, 1), axs[0]
    )
    ax.set_title("(-dkx/2, -dky/2) vs (0,0)")
    (_, ax, _) = plot_state_vector_difference_2d_x(
        eigenstate_2, eigenstate_0, (0, 1), axs[1]
    )
    ax.set_title("(-dkx/2, 0) vs (0,0)")

    fig.suptitle("Plot of difference in the absolute value of the Bloch wavefunctions")
    fig.show()
    fig.tight_layout()
    save_figure(fig, "Center wavefunction diff 2D")
    input()


# How different are the bloch wavefunctions
def calculate_eigenstate_cross_product() -> None:
    eigenstates = get_wavepacket_hydrogen(0)
    normalized = localize_tightly_bound_wavepacket_idx(eigenstates)

    (ns0, ns1) = normalized["eigenvalues"].shape
    eigenstate_0 = get_eigenstate(normalized, (ns0 // 2, ns1 // 2))
    eigenstate_1 = get_eigenstate(normalized, (0, 0))

    prod = np.multiply(eigenstate_0["vector"], np.conjugate(eigenstate_1["vector"]))
    print(prod)  # noqa: T201
    norm: np.float_ = np.sum(prod)
    print(norm)  # 0.95548 # noqa: T201
