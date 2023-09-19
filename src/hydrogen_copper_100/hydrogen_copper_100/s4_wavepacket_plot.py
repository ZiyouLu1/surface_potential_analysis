from __future__ import annotations

import itertools

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.axis.util import BasisUtil
from surface_potential_analysis.state_vector.plot import (
    plot_state_2d_k,
    plot_state_2d_x,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_bloch_state_vector,
    get_wavepacket_state_vector,
)
from surface_potential_analysis.wavepacket.localization import (
    localize_tightly_bound_wavepacket_idx,
)
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_3d_x,
    plot_all_wavepacket_states_2d_k,
    plot_all_wavepacket_states_2d_x,
    plot_eigenvalues_1d_x,
    plot_wavepacket_2d_k,
    plot_wavepacket_2d_x,
    plot_wavepacket_sample_frequencies,
)

from .s4_wavepacket import (
    get_all_wavepackets_hydrogen,
    get_single_point_projection_localized_wavepacket_hydrogen,
    get_tight_binding_projection_localized_wavepacket_hydrogen,
    get_wannier90_localized_wavepacket_hydrogen,
    get_wavepacket_hydrogen,
)


def plot_wavepacket_points() -> None:
    wavepacket = get_wavepacket_hydrogen(0)
    fig, _, _ = plot_wavepacket_sample_frequencies(wavepacket)

    fig.show()

    input()


def plot_wavepacket_hydrogen() -> None:
    for band in [0, 2]:
        wavepacket = get_wavepacket_hydrogen(band)
        fig, _, _ = plot_wavepacket_2d_x(wavepacket, (0, 1), scale="symlog")
        fig.show()
        input()


def plot_tight_binding_projection_localized_wavepacket_hydrogen() -> None:
    for band in [4]:
        wavepacket = get_tight_binding_projection_localized_wavepacket_hydrogen(band)

        fig, _, _ = plot_wavepacket_2d_x(
            wavepacket, (0, 1), scale="symlog", measure="real"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_x(
            wavepacket, (1, 2), scale="symlog", measure="real"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(
            wavepacket, (1, 0), (0,), scale="symlog", measure="real"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(
            wavepacket, (2, 1), (0,), scale="symlog", measure="real"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(
            wavepacket, (0, 2), (0,), scale="symlog", measure="real"
        )
        fig.show()
        input()


def plot_energies() -> None:
    fig, ax = plt.subplots()
    for wavepacket in get_all_wavepackets_hydrogen()[4:6]:
        plot_eigenvalues_1d_x(wavepacket, ax=ax)
    fig.show()
    input()


def plot_single_point_projection_localized_wavepacket_hydrogen() -> None:
    for band in [4]:
        wavepacket = get_single_point_projection_localized_wavepacket_hydrogen(band)
        state = get_bloch_state_vector(wavepacket, 0)
        fig, _, _ = plot_state_2d_k(state, (0, 1), measure="abs", scale="symlog")
        fig.show()
        fig, _, _ = plot_wavepacket_2d_x(
            wavepacket, (0, 1), scale="symlog", measure="abs"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_x(
            wavepacket, (1, 2), scale="symlog", measure="abs"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(
            wavepacket, (1, 0), (0,), scale="symlog", measure="abs"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(
            wavepacket, (1, 2), (0,), scale="symlog", measure="abs"
        )
        fig.show()

        fig, _, _ = plot_wavepacket_2d_k(
            wavepacket, (0, 2), (0,), scale="symlog", measure="abs"
        )
        fig.show()
        input()


def plot_all_states_hydrogen() -> None:
    for band in [4]:
        wavepacket = get_single_point_projection_localized_wavepacket_hydrogen(band)

        for i, ((fig0, _, _), (fig1, _, _), (fig2, _, _), (fig3, _, _)) in enumerate(
            zip(
                itertools.islice(
                    plot_all_wavepacket_states_2d_x(
                        wavepacket, (0, 2), scale="symlog", measure="real"
                    ),
                    24,
                ),
                itertools.islice(
                    plot_all_wavepacket_states_2d_x(
                        wavepacket, (0, 1), scale="symlog", measure="real"
                    ),
                    24,
                ),
                plot_all_wavepacket_states_2d_k(
                    wavepacket, (0, 1), scale="symlog", measure="real"
                ),
                plot_all_wavepacket_states_2d_k(
                    wavepacket, (0, 2), scale="symlog", measure="real"
                ),
            )
        ):
            fig0.show()
            fig1.show()
            fig2.show()
            fig3.show()
            print(i)
            input()


def plot_wannier90_localized_wavepacket_hydrogen() -> None:
    for band in [0, 2]:
        wavepacket = get_wannier90_localized_wavepacket_hydrogen(band)
        fig, _, _ = plot_wavepacket_2d_x(wavepacket, (0, 1), scale="symlog")
        fig.show()

        fig, _, _ = plot_wavepacket_2d_x(wavepacket, (1, 2), scale="symlog")
        fig.show()
        input()


def plot_wavepacket_3d_x() -> None:
    wavepacket = get_wavepacket_hydrogen(0)
    normalized = localize_tightly_bound_wavepacket_idx(wavepacket)

    fig, _, _ = animate_wavepacket_3d_x(normalized)
    fig.show()
    input()
    fig, _, _ = animate_wavepacket_3d_x(normalized)
    fig.show()
    input()


def plot_wavepacket_state_hydrogen() -> None:
    wavepacket = get_wavepacket_hydrogen(1)
    state = get_bloch_state_vector(wavepacket, 1)
    fig, _, _ = plot_state_2d_k(state, (0, 1), measure="abs", scale="symlog")
    fig.show()
    fig, _, _ = plot_state_2d_k(state, (0, 1), measure="real", scale="symlog")
    fig.show()
    fig, _, _ = plot_state_2d_k(state, (0, 1), measure="imag", scale="symlog")
    fig.show()
    fig, _, _ = plot_state_2d_x(state, (0, 1), measure="abs", scale="symlog")
    fig.show()
    fig, _, _ = plot_state_2d_x(state, (0, 1), measure="real", scale="symlog")
    fig.show()
    fig, _, _ = plot_state_2d_x(state, (0, 1), measure="imag", scale="symlog")
    fig.show()
    input()


# How different are the bloch wavefunctions
def calculate_eigenstate_cross_product() -> None:
    eigenstates = get_wavepacket_hydrogen(0)
    normalized = localize_tightly_bound_wavepacket_idx(eigenstates)

    (ns0, ns1, _) = BasisUtil(normalized["basis"][0]).shape
    state_0 = get_wavepacket_state_vector(normalized, (ns0 // 2, ns1 // 2))
    state_1 = get_wavepacket_state_vector(normalized, (0, 0))

    prod = np.multiply(state_0["data"], np.conjugate(state_1["data"]))
    print(prod)  # noqa: T201
    norm: np.float_ = np.sum(prod)
    print(norm)  # 0.95548 # noqa: T201
