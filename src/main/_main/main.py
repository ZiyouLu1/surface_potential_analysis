from __future__ import annotations

from hydrogen_copper_100.s4_wavepacket_plot import (
    plot_single_point_projection_localized_wavepacket_hydrogen,
)
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    plot_single_point_projection_localized_wavepacket_hydrogen()


if __name__ == "__main__":
    main()
