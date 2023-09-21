from __future__ import annotations

from sodium_copper_111.s4_wavepacket_plot import (
    plot_projection_localized_wavepacket,
    plot_wannier90_localized_wavepacket,
)
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    plot_projection_localized_wavepacket()
    plot_wannier90_localized_wavepacket()


if __name__ == "__main__":
    main()
