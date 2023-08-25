from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from surface_potential_analysis.state_vector.plot import (
    plot_state_2d_x,
    plot_state_difference_2d_x,
)
from surface_potential_analysis.wavepacket.get_eigenstate import get_state_vector
from surface_potential_analysis.wavepacket.localization import (
    localize_tightly_bound_wavepacket_idx,
)
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_2d_x_max,
    plot_wavepacket_sample_frequencies,
)

from .s4_wavepacket import (
    get_wannier90_localized_wavepacket_hydrogen,
    get_wavepacket_hydrogen,
)
from .surface_data import save_figure


def plot_wavepacket_points() -> None:
    wavepacket = get_wavepacket_hydrogen(0)
    fig, _, _ = plot_wavepacket_sample_frequencies(wavepacket)

    fig.show()

    input()


def plot_wavepacket_hydrogen() -> None:
    for band in [0, 2]:
        wavepacket = get_wavepacket_hydrogen(band)
        fig, _, _ = plot_wavepacket_2d_x_max(wavepacket, (0, 1), scale="symlog")
        fig.show()
        input()


def plot_wannier90_localized_wavepacket_hydrogen() -> None:
    for band in [0, 2]:
        wavepacket = get_wannier90_localized_wavepacket_hydrogen(band)
        fig, _, _ = plot_wavepacket_2d_x_max(wavepacket, (0, 1), scale="symlog")
        fig.show()

        fig, _, _ = plot_wavepacket_2d_x_max(wavepacket, (1, 2), scale="symlog")
        fig.show()
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
    state_0 = get_state_vector(normalized, (ns0 // 2, ns1 // 2, 0))
    state_1 = get_state_vector(normalized, (0, 0, 0))
    state_2 = get_state_vector(normalized, (ns0 // 2, 0, 0))

    fig, axs = plt.subplots(2, 3)
    (_, ax, _) = plot_state_2d_x(state_0, (0, 1), (0,), ax=axs[0][0])
    ax.set_title("(-dkx/2, -dky/2) at z=0")
    (_, ax, _) = plot_state_2d_x(state_1, (0, 1), (0,), ax=axs[0][1])
    ax.set_title("(0,0) at z=0")
    (_, ax, _) = plot_state_2d_x(state_2, (0, 1), (0,), ax=axs[0][2])
    ax.set_title("(-dkx/2, 0) at z=0")

    (_, ax, _) = plot_state_2d_x(state_0, (0, 1), (100,), ax=axs[1][0])
    ax.set_title("(-dkx/2, -dky/2) at z=delta_x")
    (_, ax, _) = plot_state_2d_x(state_1, (0, 1), (100,), ax=axs[1][1])
    ax.set_title("(0,0) at z=delta_x")
    (_, ax, _) = plot_state_2d_x(state_2, (0, 1), (100,), ax=axs[1][2])
    ax.set_title("(-dkx/2, 0) at z=delta_x")

    fig.tight_layout()
    fig.suptitle("Plot of absolute value of the Bloch wavefunctions")
    save_figure(fig, "Center and middle wavefunctions 2D")
    fig.show()

    fig, axs = plt.subplots(1, 2)
    (_, ax, _) = plot_state_difference_2d_x(state_1, state_0, (0, 1), axs[0])
    ax.set_title("(-dkx/2, -dky/2) vs (0,0)")
    (_, ax, _) = plot_state_difference_2d_x(state_2, state_0, (0, 1), axs[1])
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

    (ns0, ns1, _) = normalized["shape"]
    state_0 = get_state_vector(normalized, (ns0 // 2, ns1 // 2))
    state_1 = get_state_vector(normalized, (0, 0))

    prod = np.multiply(state_0["vector"], np.conjugate(state_1["vector"]))
    print(prod)  # noqa: T201
    norm: np.float_ = np.sum(prod)
    print(norm)  # 0.95548 # noqa: T201
