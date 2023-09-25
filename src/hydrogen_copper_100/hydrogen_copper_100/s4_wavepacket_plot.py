from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.probability_vector.plot import (
    plot_probability_1d_k,
    plot_probability_2d_k,
)
from surface_potential_analysis.probability_vector.probability_vector import (
    average_probabilities,
    from_state_vector_list,
)
from surface_potential_analysis.state_vector.plot import (
    plot_state_1d_k,
    plot_state_2d_k,
    plot_state_2d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import get_state_vector
from surface_potential_analysis.state_vector.util import (
    get_most_localized_free_state_vectors,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket_list,
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
    plot_eigenvalues_1d_x,
    plot_wavepacket_1d_k,
    plot_wavepacket_2d_k,
    plot_wavepacket_2d_x,
    plot_wavepacket_sample_frequencies,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_wavepacket_basis,
    get_wavepackets,
    wavepacket_list_into_iter,
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


def plot_localized_wavepackets_hydrogen() -> None:
    wavepackets = get_wannier90_localized_wavepacket_hydrogen(8)
    for wavepacket in wavepacket_list_into_iter(wavepackets):
        fig, _, _ = plot_wavepacket_1d_k(wavepacket, scale="symlog", measure="abs")
        fig.show()
        fig, _, _ = plot_wavepacket_2d_x(wavepacket, scale="symlog", measure="abs")
        fig.show()
    input()


def plot_wavepacket_average_occupation_probability() -> None:
    wavepackets = get_all_wavepackets_hydrogen()

    fig, ax = plt.subplots()
    for n_bands in [1, 4, 8, 16]:
        unfurled = unfurl_wavepacket_list(get_wavepackets(wavepackets, slice(n_bands)))
        probabilities = from_state_vector_list(unfurled)
        averaged = average_probabilities(probabilities)

        _, _, line = plot_probability_1d_k(
            averaged, idx=(0, 0), ax=ax, measure="abs", scale="symlog"
        )
        line.set_label(f"{n_bands} bands")

    projections = get_most_localized_free_state_vectors(
        get_wavepacket_basis(wavepackets), (4, 4, 1)
    )
    _, _, line = plot_state_1d_k(get_state_vector(projections, 0), ax=ax.twinx())
    line.set_label("Projections")
    ax.legend()
    fig.show()

    unfurled = unfurl_wavepacket_list(get_wavepackets(wavepackets, slice(8)))
    probabilities = from_state_vector_list(unfurled)
    averaged = average_probabilities(probabilities)
    fig, _, _ = plot_probability_2d_k(averaged, idx=(0,), measure="abs")
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
    for band in range(4, 6):
        wavepacket = get_wavepacket_hydrogen(band)

        plot_eigenvalues_1d_x(wavepacket, ax=ax)
    fig.show()
    input()


def plot_single_point_projection_localized_wavepacket_hydrogen() -> None:
    for band in [4]:
        wavepacket = get_single_point_projection_localized_wavepacket_hydrogen(band)
        state = get_bloch_state_vector(wavepacket, 0)
        fig, _, _ = plot_state_2d_k(state, (0, 1), measure="abs", scale="symlog")
        fig.show()

        fig, _, _ = plot_state_2d_x(state, (0, 1), measure="abs", scale="symlog")
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
