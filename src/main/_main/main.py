from __future__ import annotations

import sodium_copper_111
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    sodium_copper_111.s3_eigenstates_plot.plot_first_six_band_energies()
    sodium_copper_111.s3_eigenstates_plot.plot_first_six_band_boltzmann_occupation()
    sodium_copper_111.s4_wavepacket_plot.plot_first_six_wavepackets()


if __name__ == "__main__":
    main()
