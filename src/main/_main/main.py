from __future__ import annotations

import hydrogen_nickel_111.s6_schrodinger_isf
from hydrogen_copper_100.s4_wavepacket import get_all_wavepackets_hydrogen
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    get_all_wavepackets_hydrogen()
    hydrogen_nickel_111.s6_schrodinger_isf.plot_average_isf_all_temperatures()


if __name__ == "__main__":
    main()
