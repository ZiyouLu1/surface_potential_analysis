from __future__ import annotations

import hydrogen_nickel_111
import hydrogen_nickel_111.s5_overlap_report_plots
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    hydrogen_nickel_111.s5_overlap_report_plots.plot_rate_equation()


if __name__ == "__main__":
    main()
