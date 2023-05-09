from __future__ import annotations

import nickel_111
from surface_potential_analysis.util import timed


@timed
def main() -> None:
    nickel_111.s5_overlap_analysis.simulate_hydrogen_system()


if __name__ == "__main__":
    main()
