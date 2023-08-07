from __future__ import annotations

import hydrogen_nickel_111
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    hydrogen_nickel_111.s4_wavepacket_plot.plot_tight_binding_projection_localized_wavepacket_hydrogen()


if __name__ == "__main__":
    main()
