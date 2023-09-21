from __future__ import annotations

from matplotlib import pyplot as plt
from surface_potential_analysis.probability_vector.plot import plot_probability_1d_k
from surface_potential_analysis.probability_vector.probability_vector import (
    average_probabilities,
    from_state_vector_list,
)
from surface_potential_analysis.state_vector.plot import plot_state_1d_k
from surface_potential_analysis.state_vector.state_vector_list import get_state_vector
from surface_potential_analysis.state_vector.util import (
    get_most_localized_free_state_vectors,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket_list,
)
from surface_potential_analysis.wavepacket.localization import (
    localize_position_operator,
)
from surface_potential_analysis.wavepacket.plot import (
    plot_wavepacket_1d_k,
    plot_wavepacket_1d_x,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_wavepacket_basis,
    get_wavepackets,
    wavepacket_list_into_iter,
)

from sodium_copper_111.s4_wavepacket import (
    get_all_wavepackets,
    get_localized_wavepackets_projection,
    get_localized_wavepackets_wannier_90,
    get_wavepacket,
)


def plot_wavepacket_average_occupation_probability() -> None:
    wavepackets = get_all_wavepackets((21,), (60,))

    fig, ax = plt.subplots()
    for n_bands in [5, 10, 15, 20, 25]:
        unfurled = unfurl_wavepacket_list(get_wavepackets(wavepackets, slice(n_bands)))
        probabilities = from_state_vector_list(unfurled)
        averaged = average_probabilities(probabilities)

        _, _, line = plot_probability_1d_k(averaged, ax=ax, measure="abs")
        line.set_label(f"{n_bands} bands")

    projections = get_most_localized_free_state_vectors(
        get_wavepacket_basis(wavepackets), (30,)
    )
    _, _, line = plot_state_1d_k(get_state_vector(projections, 0), ax=ax.twinx())
    line.set_label("Projections")
    fig.show()
    input()


def plot_projection_localized_wavepacket() -> None:
    wavepackets = get_localized_wavepackets_projection((21,), (60,), 25)

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    for i, wavepacket in enumerate(wavepacket_list_into_iter(wavepackets)):
        _, _, ln = plot_wavepacket_1d_x(wavepacket, ax=ax0)
        ln.set_label(f"n={i}")

        _, _, ln = plot_wavepacket_1d_k(wavepacket, ax=ax1, measure="abs")
        ln.set_label(f"n={i}")
    fig0.show()
    fig1.show()
    input()


def plot_wannier90_localized_wavepacket() -> None:
    # only converges when n_bands is even
    wavepackets = get_localized_wavepackets_wannier_90((10,), (61,), 26)

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    for i, wavepacket in enumerate(wavepacket_list_into_iter(wavepackets)):
        _, _, ln = plot_wavepacket_1d_x(wavepacket, ax=ax0)
        ln.set_label(f"n={i}")

        _, _, ln = plot_wavepacket_1d_k(wavepacket, ax=ax1, measure="real")
        ln.set_label(f"n={i}")
    fig0.show()
    fig1.show()
    input()


def plot_operator_localized_states_large_band_increasing_resolution() -> None:
    wavepacket = get_wavepacket((12,), (600,), 16)
    wavepackets = localize_position_operator(wavepacket)

    fig, ax = plt.subplots()
    for i, wavepacket in enumerate(wavepackets):
        _, _, ln = plot_wavepacket_1d_x(wavepacket, ax=ax)
        ln.set_label(f"n={i}")

    ax.legend()
    ax.set_title("Plot of the six lowest energy wavepackets")
    fig.show()
    input()
