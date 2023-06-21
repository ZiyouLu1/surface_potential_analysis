from __future__ import annotations

import hydrogen_nickel_111
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    hydrogen_nickel_111.s3_eigenstates_plot.plot_lowest_band_energies()


if __name__ == "__main__":
    main()
