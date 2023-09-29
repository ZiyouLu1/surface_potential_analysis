from __future__ import annotations

from hydrogen_nickel_111.s6_schrodinger_dynamics import (
    plot_incoherent_occupation_comparison_hydrogen,
)
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    plot_incoherent_occupation_comparison_hydrogen()


if __name__ == "__main__":
    main()
