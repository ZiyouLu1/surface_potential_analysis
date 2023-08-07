from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.state_vector.plot import plot_state_1d_x
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.localization import (
    localize_position_operator,
    localize_position_operator_many_band_individual,
    localize_tightly_bound_wavepacket_idx,
)
from surface_potential_analysis.wavepacket.plot import plot_wavepacket_1d_x

from sodium_copper_111.s4_wavepacket import (
    get_all_wavepackets,
    get_localized_wavepackets,
    get_wavepacket,
)
from sodium_copper_111.surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.state_vector import StateVector


def plot_first_six_wavepackets() -> None:
    fig, ax = plt.subplots()

    for i in range(0, 13):
        wavepacket = get_wavepacket((12,), (600,), i)
        wavepacket = localize_tightly_bound_wavepacket_idx(wavepacket, idx=(0,))
        _, _, ln = plot_wavepacket_1d_x(wavepacket, ax=ax)
        ln.set_label(f"n={i}")

    ax.legend()
    ax.set_title("Plot of the six lowest energy wavepackets")
    fig.show()
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


def plot_operator_localized_states_single_band() -> None:
    wavepackets = get_localized_wavepackets((8,), (100,), 3)

    fig, ax = plt.subplots()
    for i, wavepacket in enumerate(wavepackets):
        _, _, ln = plot_wavepacket_1d_x(wavepacket, ax=ax)
        ln.set_label(f"n={i}")
    fig.show()

    wavepackets = get_localized_wavepackets((10,), (100,), 15)
    fig, ax = plt.subplots()
    for i, wavepacket in enumerate(wavepackets):
        _, _, ln = plot_wavepacket_1d_x(wavepacket, ax=ax)
        ln.set_label(f"n={i}")

    ax.legend()
    fig.show()
    input()


@npy_cached(
    get_data_path("wavepacket/localized_states_14_band_2.npy"), load_pickle=True
)
def get_many_band_localized_states() -> list[StateVector[Any]]:
    wavepackets = get_all_wavepackets((40,), (100,))
    return localize_position_operator_many_band_individual(wavepackets[0:16])


def plot_operator_localized_states_many_band() -> None:
    eigenstates = get_many_band_localized_states()

    fig, ax = plt.subplots()
    for i, eigenstate in enumerate(eigenstates):
        _, _, ln = plot_state_1d_x(eigenstate, ax=ax)
        ln.set_label(f"n={i}")

    ax.set_xlim(3.2e-9, 3.7e-9)
    fig.show()
    input()


def test_wavepacket_normalization() -> None:
    # Does the wavepacket remain normalized no matter which index we choose
    # to normalize onto. The answer is yes, as long as we dont choose
    # to sit on a node exactly!
    fig, ax = plt.subplots()

    for idx in [0, 250, 500, 750, 1000]:
        wavepacket = get_wavepacket((12,), (600,), 0)
        normalized = localize_tightly_bound_wavepacket_idx(wavepacket, idx=(idx,))
        _, _, ln = plot_wavepacket_1d_x(normalized, ax=ax)
        ln.set_label(f"{idx}")

    ax.legend()
    ax.set_title("Plot of wavepackets of Na, showing incorrect localization")
    fig.show()
    input()


def test_wavepacket_zero_at_next_unit_cell() -> None:
    wavepacket = get_wavepacket((12,), (600,), 0)
    offset = 50
    size = AxisWithLengthBasisUtil(wavepacket["basis"]).size

    normalized = localize_tightly_bound_wavepacket_idx(wavepacket, idx=(offset,))
    unfurled = unfurl_wavepacket(normalized)
    unfurled_position = convert_state_vector_to_position_basis(unfurled)  # type: ignore[arg-type]
    np.testing.assert_array_almost_equal(
        unfurled_position["vector"][offset + size :: size],
        np.zeros_like(np.prod(wavepacket["shape"]) - 1),
    )
