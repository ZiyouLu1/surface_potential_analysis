from __future__ import annotations

import hydrogen_copper_100
import hydrogen_nickel_111
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    hydrogen_nickel_111.s3_eigenstates_plot.plot_hydrogen_lowest_band_energy()
    hydrogen_nickel_111.s3_eigenstates_plot.plot_deuterium_lowest_band_energy()
    hydrogen_copper_100.s3_eigenstates_plot.plot_lowest_bands_relaxed()


if __name__ == "__main__":
    main()
