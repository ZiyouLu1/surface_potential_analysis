from __future__ import annotations

import hydrogen_nickel_111.s6_dynamics
from surface_potential_analysis.util.decorators import timed


def generate_a_matrix() -> None:
    temperatures = [125, 150, 175, 200, 225]

    for t in temperatures:
        hydrogen_nickel_111.s6_dynamics.get_tunnelling_a_matrix_deuterium((5, 5), 6, t)
    for t in temperatures:
        hydrogen_nickel_111.s6_dynamics.get_tunnelling_a_matrix_hydrogen((5, 5), 6, t)


@timed
def main() -> None:
    generate_a_matrix()

    hydrogen_nickel_111.s3_eigenstates_plot.plot_hydrogen_lowest_band_energy()
    hydrogen_nickel_111.s3_eigenstates_plot.plot_deuterium_lowest_band_energy()


if __name__ == "__main__":
    main()
