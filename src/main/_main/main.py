from __future__ import annotations

import hydrogen_nickel_111.s6_schrodinger_dynamics
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    hydrogen_nickel_111.s4_wavepacket_plot.plot_wannier90_many_band_localized_wavepacket_hydrogen()


if __name__ == "__main__":
    main()
