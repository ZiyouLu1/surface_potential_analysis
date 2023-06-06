from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.eigenstate.conversion import (
    convert_eigenstate_to_position_basis,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.normalization import normalize_wavepacket
from surface_potential_analysis.wavepacket.plot import plot_wavepacket_1d_x

from sodium_copper_111.s4_wavepacket import get_wavepacket


def plot_first_six_wavepackets() -> None:
    fig, ax = plt.subplots()

    for i in range(6):
        wavepacket = get_wavepacket(i)
        wavepacket = normalize_wavepacket(wavepacket, idx=(2,))
        _, _, ln = plot_wavepacket_1d_x(wavepacket, ax=ax)
        ln.set_label(f"n={i}")

    ax.legend()
    fig.show()
    input()


def test_wavepacket_normalization() -> None:
    # Does the wavepacket remain normalized no matter which index we choose
    # to normalize onto. The answer is yes, as long as we dont choose
    # to sit on a node exactly!
    fig, ax = plt.subplots()

    for idx in [0, 250, 500, 750, 1000]:
        wavepacket = get_wavepacket(0)
        normalized = normalize_wavepacket(wavepacket, idx=(idx,))
        _, _, ln = plot_wavepacket_1d_x(normalized, ax=ax)
        ln.set_label(f"{idx}")

    ax.legend()
    ax.set_title("Plot of wavepackets of Na, showing incorrect localization")
    fig.show()
    input()


def test_wavepacket_zero_at_next_unit_cell() -> None:
    wavepacket = get_wavepacket(0)
    offset = 50
    size = BasisUtil(wavepacket["basis"]).size

    normalized = normalize_wavepacket(wavepacket, idx=(offset,))
    unfurled = unfurl_wavepacket(normalized)
    unfurled_position = convert_eigenstate_to_position_basis(unfurled)  # type: ignore[arg-type]
    np.testing.assert_array_almost_equal(
        unfurled_position["vector"][offset + size :: size],
        np.zeros_like(np.prod(wavepacket["shape"]) - 1),
    )
