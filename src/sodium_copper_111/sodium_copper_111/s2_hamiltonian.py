from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)

from .s1_potential import get_interpolated_potential

if TYPE_CHECKING:
    import numpy as np
    from surface_potential_analysis.axis.axis import FundamentalMomentumAxis1d
    from surface_potential_analysis.operator.operator import Operator

_L0Inv = TypeVar("_L0Inv", bound=int)

SODIUM_MASS = 3.8175458e-26


def get_hamiltonian(
    shape: tuple[_L0Inv],
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float_]] | None = None,
) -> Operator[
    tuple[FundamentalMomentumAxis1d[_L0Inv]], tuple[FundamentalMomentumAxis1d[_L0Inv]]
]:
    potential = get_interpolated_potential(shape)
    return total_surface_hamiltonian(potential, SODIUM_MASS, bloch_fraction)
