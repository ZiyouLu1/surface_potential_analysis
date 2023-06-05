from __future__ import annotations

from matplotlib import pyplot as plt
from surface_potential_analysis.wavepacket.normalization import normalize_wavepacket
from surface_potential_analysis.wavepacket.plot import plot_wavepacket_1d_x

from sodium_copper_111.s4_wavepacket import get_wavepacket


def plot_wavepacket() -> None:
    fig, ax = plt.subplots()

    wavepacket = get_wavepacket(0)
    normalized = normalize_wavepacket(wavepacket, idx=(0,))
    plot_wavepacket_1d_x(normalized, ax=ax)

    wavepacket = get_wavepacket(1)
    normalized = normalize_wavepacket(wavepacket, idx=(0,))
    plot_wavepacket_1d_x(normalized, ax=ax)

    wavepacket = get_wavepacket(2)
    normalized = normalize_wavepacket(wavepacket, idx=(0,))
    plot_wavepacket_1d_x(normalized, ax=ax)

    wavepacket = get_wavepacket(3)
    normalized = normalize_wavepacket(wavepacket, idx=(0,))
    plot_wavepacket_1d_x(normalized, ax=ax)

    fig.show()
    input()


def test_wavepacket_normalization() -> None:
    # Does the wavepacket remain normalized no matter which index we choose
    # to normalize onto.
    fig, ax = plt.subplots()
    ax.twinx()

    wavepacket = get_wavepacket(0)
    normalized = normalize_wavepacket(wavepacket, idx=(0,))
    _, _, ln_0 = plot_wavepacket_1d_x(normalized, ax=ax)
    _, _, ln_1 = plot_wavepacket_1d_x(normalized, measure="real", ax=ax)
    _, _, ln_2 = plot_wavepacket_1d_x(normalized, measure="imag", ax=ax)
    ln_0.set_label("0")
    ln_1.set_color(ln_0.get_color())
    ln_1.set_linestyle("--")
    ln_2.set_color(ln_0.get_color())
    ln_2.set_linestyle("--")

    normalized = normalize_wavepacket(wavepacket, idx=(50,))
    _, _, ln_0 = plot_wavepacket_1d_x(normalized, ax=ax)
    _, _, ln_1 = plot_wavepacket_1d_x(normalized, measure="real", ax=ax)
    _, _, ln_2 = plot_wavepacket_1d_x(normalized, measure="imag", ax=ax)
    ln_0.set_label("50")
    ln_1.set_color(ln_0.get_color())
    ln_1.set_linestyle("--")
    ln_2.set_color(ln_0.get_color())
    ln_2.set_linestyle("--")

    normalized = normalize_wavepacket(wavepacket, idx=(150,))
    _, _, ln_0 = plot_wavepacket_1d_x(normalized, ax=ax)
    _, _, ln_1 = plot_wavepacket_1d_x(normalized, measure="real", ax=ax)
    _, _, ln_2 = plot_wavepacket_1d_x(normalized, measure="imag", ax=ax)
    ln_0.set_label("150")
    ln_1.set_color(ln_0.get_color())
    ln_1.set_linestyle("--")
    ln_2.set_color(ln_0.get_color())
    ln_2.set_linestyle("--")

    fig.show()
    ax.legend()
    input()
