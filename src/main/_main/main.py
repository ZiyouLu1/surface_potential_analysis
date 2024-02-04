from __future__ import annotations

from cProfile import Profile

from hydrogen_nickel_111.s6_schrodinger_isf import (
    get_simulation_at_temperature_double_collapse,
)
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    temperature = 250
    times = EvenlySpacedTimeBasis(2000, 20, 0, 6e-10)
    try:
        with Profile() as p:
            get_simulation_at_temperature_double_collapse(temperature, 0, times)
    finally:
        p.dump_stats("out1.prof")


if __name__ == "__main__":
    main()
