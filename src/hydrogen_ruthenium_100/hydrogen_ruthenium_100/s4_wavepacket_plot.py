from __future__ import annotations

import numpy as np
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.wavepacket.plot import (
    plot_wavepacket_x0x1,
)

from .s4_wavepacket import get_two_point_normalized_wavepacket_hydrogen
from .surface_data import save_figure


def plot_hydrogen_wavepacket_at_x2_max() -> None:
    for band in range(0, 6):
        normalized = get_two_point_normalized_wavepacket_hydrogen(band)
        _, _, x2_max = BasisUtil(normalized["basis"]).get_stacked_index(
            np.argmax(np.abs(normalized["vectors"][0]))
        )
        fig, ax, _ = plot_wavepacket_x0x1(normalized, x2_max, scale="symlog")
        fig.show()
        ax.set_title(f"Plot of abs(wavefunction) for ix2={x2_max}")
        save_figure(fig, f"./wavepacket/hydrogen_wavepacket_{band}.png")
    input()
