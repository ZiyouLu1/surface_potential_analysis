from __future__ import annotations

import hydrogen_platinum_111
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    hydrogen_platinum_111.s3_eigenstates_plot.plot_lowest_bands_hydrogen()
    hydrogen_platinum_111.s3_eigenstates_plot.plot_lowest_bands_deuterium()
    hydrogen_platinum_111.s3_eigenstates_plot.plot_lowest_band_energy_deuterium()


if __name__ == "__main__":
    main()
