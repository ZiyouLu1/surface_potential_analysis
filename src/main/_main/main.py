from __future__ import annotations

import hydrogen_copper_111
import hydrogen_nickel_111.s6_a_calculation
import hydrogen_nickel_111.s6_dynamics
import hydrogen_nickel_111.s6_isf_analysis
from surface_potential_analysis.util.decorators import timed


def generate_a_matrix() -> None:
    temperatures = [100, 125, 150, 175, 200, 225, 250]

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
    hydrogen_copper_111.s3_eigenstates_plot.plot_lowest_band_energy_deuterium()

    hydrogen_nickel_111.s6_isf_analysis.plot_fast_slow_rate_ratios()

    hydrogen_copper_111.s3_eigenstates_plot.plot_lowest_band_energy_hydrogen()


if __name__ == "__main__":
    main()
