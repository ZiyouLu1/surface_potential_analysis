from __future__ import annotations

from cProfile import Profile

from hydrogen_nickel_111.s6_schrodinger_isf import (
    get_simulation_at_temperature_double_collapse,
)
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    temperature = 125
    times = EvenlySpacedTimeBasis(2, 2000, 0, 24e-10)
    p: Profile | None = None

    try:
        with Profile() as p:
            get_simulation_at_temperature_double_collapse(temperature, 0, times, _i=0)
    finally:
        if p is not None:
            p.dump_stats("out.prof")


if __name__ == "__main__":
    main()
