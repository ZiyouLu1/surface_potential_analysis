from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from scipy.constants import electron_volt
from surface_potential_analysis.axis.axis import (
    FundamentalTransformedPositionAxis1d,
    TransformedPositionAxis1d,
)
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis

if TYPE_CHECKING:
    from surface_potential_analysis.potential.potential import Potential1d

_L0Inv = TypeVar("_L0Inv", bound=int)

LATTICE_CONSTANT = 3.615 * 10**-10
BARRIER_ENERGY = 55 * 10**-3 * electron_volt


def get_potential() -> (
    Potential1d[tuple[FundamentalTransformedPositionAxis1d[Literal[3]]]]
):
    delta_x = np.sqrt(3) * LATTICE_CONSTANT / 2
    axis = FundamentalTransformedPositionAxis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * BARRIER_ENERGY * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": (axis,), "vector": vector}


def get_interpolated_potential(
    shape: tuple[_L0Inv],
) -> Potential1d[tuple[FundamentalTransformedPositionAxis1d[_L0Inv]]]:
    potential = get_potential()
    old = potential["basis"][0]
    basis = (
        TransformedPositionAxis1d[_L0Inv, Literal[3]](old.delta_x, old.n, shape[0]),
    )
    scaled_potential = potential["vector"] * np.sqrt(shape[0] / old.n)
    return convert_potential_to_basis(
        {"basis": basis, "vector": scaled_potential},
        basis_as_fundamental_momentum_basis(basis),
    )
