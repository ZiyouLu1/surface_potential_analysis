from __future__ import annotations

from matplotlib import pyplot as plt
from surface_potential_analysis.eigenstate.conversion import (
    convert_eigenstate_to_position_basis,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.normalization import normalize_wavepacket
from surface_potential_analysis.wavepacket.plot import plot_wavepacket_1d_x

from sodium_copper_111.s4_wavepacket import get_wavepacket


def plot_wavepacket() -> None:
    fig, ax = plt.subplots()

    wavepacket = get_wavepacket(0)
    wavepacket = normalize_wavepacket(wavepacket, idx=(0,))
    plot_wavepacket_1d_x(wavepacket, ax=ax)

    wavepacket = get_wavepacket(1)
    wavepacket = normalize_wavepacket(wavepacket, idx=(0,))
    plot_wavepacket_1d_x(wavepacket, ax=ax)

    wavepacket = get_wavepacket(2)
    wavepacket = normalize_wavepacket(wavepacket, idx=(0,))
    plot_wavepacket_1d_x(wavepacket, ax=ax)

    wavepacket = get_wavepacket(3)
    wavepacket = normalize_wavepacket(wavepacket, idx=(0,))
    plot_wavepacket_1d_x(wavepacket, ax=ax)

    fig.show()
    input()


def test_wavepacket_normalization() -> None:
    # Does the wavepacket remain normalized no matter which index we choose
    # to normalize onto.
    fig, ax = plt.subplots()

    wavepacket = get_wavepacket(0)
    normalized = normalize_wavepacket(wavepacket, idx=(0,))
    _, _, ln_0 = plot_wavepacket_1d_x(normalized, ax=ax)
    ln_0.set_label("0")

    normalized = normalize_wavepacket(wavepacket, idx=(500,))
    _, _, ln_0 = plot_wavepacket_1d_x(normalized, ax=ax)
    ln_0.set_label("50")

    normalized = normalize_wavepacket(wavepacket, idx=(1500,))
    _, _, ln_0 = plot_wavepacket_1d_x(normalized, ax=ax)
    ln_0.set_label("150")

    normalized = normalize_wavepacket(wavepacket, idx=(2000,))
    _, _, ln_0 = plot_wavepacket_1d_x(normalized, ax=ax)
    ln_0.set_label("200")

    ax.legend()
    ax.set_title("Plot of wavepackets of Na, showing proper localization")
    fig.show()
    input()


def test_wavepacket_zero_at_next_unit_cell() -> None:
    wavepacket = get_wavepacket(0)
    normalized = normalize_wavepacket(wavepacket, idx=(0,))
    unfurled = unfurl_wavepacket(normalized)
    unfurled_position = convert_eigenstate_to_position_basis(unfurled)  # type: ignore[arg-type]
    print(unfurled_position["vector"][::2000])  # noqa: T201
