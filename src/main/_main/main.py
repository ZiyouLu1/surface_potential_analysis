from __future__ import annotations

from hydrogen_nickel_111.s6_schrodinger_isf import (
    plot_average_isf_all_temperatures,
)
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    plot_average_isf_all_temperatures()


if __name__ == "__main__":
    main()
