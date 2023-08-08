from __future__ import annotations

import hydrogen_nickel_111
import hydrogen_nickel_111.s6_isf_analysis
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    hydrogen_nickel_111.s6_isf_analysis.plot_tunnelling_rate_hydrogen()


if __name__ == "__main__":
    main()
