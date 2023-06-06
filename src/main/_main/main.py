from __future__ import annotations

import sodium_copper_111
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    sodium_copper_111.s1_potential_plot.plot_sodium_potential()
    sodium_copper_111.s4_wavepacket_plot.plot_first_six_wavepackets()
    sodium_copper_111.s4_wavepacket_plot.test_wavepacket_normalization()
    sodium_copper_111.s4_wavepacket_plot.test_wavepacket_zero_at_next_unit_cell()


if __name__ == "__main__":
    main()
