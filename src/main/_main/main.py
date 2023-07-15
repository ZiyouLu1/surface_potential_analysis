from __future__ import annotations

import hydrogen_nickel_111.s6_a_calculation
import hydrogen_nickel_111.s6_dynamics
import hydrogen_nickel_111.s6_isf_analysis
import hydrogen_ruthenium_100.s4_wavepacket
from surface_potential_analysis.util.decorators import timed


def generate_a_matrix() -> None:
    temperatures = [125, 150, 175, 200, 225]

    for t in temperatures:
        hydrogen_nickel_111.s6_a_calculation.get_tunnelling_a_matrix_deuterium(
            (5, 5), 6, t
        )
    hydrogen_nickel_111.s6_a_calculation._calculate_gamma_potential_integral_deuterium_diagonal.cache_clear()
    for t in temperatures:
        hydrogen_nickel_111.s6_a_calculation.get_tunnelling_a_matrix_hydrogen(
            (5, 5), 6, t
        )
    hydrogen_nickel_111.s6_a_calculation._calculate_gamma_potential_integral_hydrogen_diagonal.cache_clear()


@timed
def main() -> None:
    hydrogen_ruthenium_100.s4_wavepacket.get_all_wavepackets_deuterium()
    generate_a_matrix()
    hydrogen_nickel_111.s3_eigenstates_plot.plot_hydrogen_lowest_band_energy()
    hydrogen_nickel_111.s3_eigenstates_plot.plot_deuterium_lowest_band_energy()


if __name__ == "__main__":
    main()
