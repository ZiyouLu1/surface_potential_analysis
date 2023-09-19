from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from scipy.constants import electron_volt
from surface_potential_analysis.axis.axis import (
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis1d,
)
from surface_potential_analysis.axis.stacked_axis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.potential.potential import Potential

_L0Inv = TypeVar("_L0Inv", bound=int)

LATTICE_CONSTANT = 3.615 * 10**-10
BARRIER_ENERGY = 55 * 10**-3 * electron_volt


def get_potential() -> (
    Potential[StackedBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]
):
    delta_x = np.sqrt(3) * LATTICE_CONSTANT / 2
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * BARRIER_ENERGY * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": StackedBasis(axis), "data": vector}


def get_interpolated_potential(
    shape: tuple[_L0Inv],
) -> Potential[
    StackedBasisLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]]
]:
    potential = get_potential()
    old = potential["basis"][0]
    basis = StackedBasis(
        TransformedPositionBasis1d[_L0Inv, Literal[3]](old.delta_x, old.n, shape[0]),
    )
    scaled_potential = potential["data"] * np.sqrt(shape[0] / old.n)
    return convert_potential_to_basis(
        {"basis": basis, "data": scaled_potential},
        stacked_basis_as_fundamental_momentum_basis(basis),
    )
