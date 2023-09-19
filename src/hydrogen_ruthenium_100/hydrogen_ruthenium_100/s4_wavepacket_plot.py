from __future__ import annotations

from surface_potential_analysis.wavepacket.localization import (
    localize_single_point_projection,
)
from surface_potential_analysis.wavepacket.plot import plot_wavepacket_2d_x

from .s4_wavepacket import (
    get_two_point_normalized_wavepacket_hydrogen,
    get_wavepacket_hydrogen,
)
from .surface_data import save_figure


def plot_hydrogen_wavepacket() -> None:
    for band in range(6):
        wavepacket = get_two_point_normalized_wavepacket_hydrogen(band)
        fig, _, _ = plot_wavepacket_2d_x(wavepacket, (0, 1), scale="symlog")
        fig.show()
        save_figure(fig, f"./wavepacket/hydrogen_wavepacket_{band}.png")
    input()


def plot_projection_localized_wavepacket_hydrogen() -> None:
    for band in [2]:
        wavepacket = get_wavepacket_hydrogen(band)
        localized = localize_single_point_projection(wavepacket)
        fig, _, _ = plot_wavepacket_2d_x(localized, (0, 1), scale="symlog")
        fig.show()
        input()
