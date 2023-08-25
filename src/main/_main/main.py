from __future__ import annotations

import hydrogen_nickel_111.s6_schrodinger_dynamics
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    hydrogen_nickel_111.s4_wavepacket.get_all_wavepackets_hydrogen_extrapolated()


if __name__ == "__main__":
    main()
