from __future__ import annotations

from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_3d_x,
    plot_wavepacket_2d_x,
    plot_wavepacket_sample_frequencies,
)
from surface_potential_analysis.wavepacket.wavepacket import get_wavepacket

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


def animate_copper_111_wavepacket() -> None:
    wavepackets = get_wannier90_localized_wavepacket_hydrogen(8)
    wavepacket = get_wavepacket(wavepackets, 0)
    fig, _, _anim0 = animate_wavepacket_3d_x(wavepacket)
    fig.show()

    wavepacket = get_wavepacket(wavepackets, 1)
    fig, _, _anim1 = animate_wavepacket_3d_x(wavepacket)
    fig.show()
    input()


def plot_wavepacket_at_maximum() -> None:
    wavepackets = get_wannier90_localized_wavepacket_hydrogen(16)

    for band in range(16):
        localized = get_wavepacket(wavepackets, band)

        fig, ax, _ = plot_wavepacket_2d_x(localized, measure="abs")
        fig.show()
        ax.set_title("Plot of abs(wavefunction) for z=z max")
        save_figure(fig, f"wavepacket_grid_{band}.png")
    input()
