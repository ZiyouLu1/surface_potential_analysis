from __future__ import annotations

import hydrogen_ruthenium_100
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    hydrogen_ruthenium_100.s4_wavepacket_plot.plot_wannier90_localized_wavepacket_hydrogen()


if __name__ == "__main__":
    main()
