from __future__ import annotations

import hydrogen_nickel_111
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    hydrogen_nickel_111.s5_overlap_analysis.calculate_potential_integral()


if __name__ == "__main__":
    main()
