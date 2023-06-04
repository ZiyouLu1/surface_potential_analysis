from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

from surface_potential_analysis.hamiltonian.hamiltonian import (
    Hamiltonian1d,
    add_hamiltonian,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    hamiltonian_from_mass,
    hamiltonian_from_potential,
)

from .s1_potential import get_interpolated_potential

if TYPE_CHECKING:
    import numpy as np
    from surface_potential_analysis.axis.axis import FundamentalMomentumAxis1d

_L0Inv = TypeVar("_L0Inv", bound=int)

SODIUM_MASS = 3.8175458e-26


def get_hamiltonian(
    shape: tuple[_L0Inv],
    bloch_phase: np.ndarray[tuple[Literal[1]], np.dtype[np.float_]] | None = None,
) -> Hamiltonian1d[tuple[FundamentalMomentumAxis1d[_L0Inv]]]:
    potential = hamiltonian_from_potential(get_interpolated_potential(shape))
    momentum = hamiltonian_from_mass(potential["basis"], SODIUM_MASS, bloch_phase)
    return add_hamiltonian(potential, momentum)
