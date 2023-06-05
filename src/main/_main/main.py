from __future__ import annotations

import sodium_copper_111
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    sodium_copper_111.s4_wavepacket_plot.plot_wavepacket()
    sodium_copper_111.s4_wavepacket_plot.test_wavepacket_normalization()


if __name__ == "__main__":
    main()
