from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import MomentumAxis
from surface_potential_analysis.axis.conversion import axis_as_fundamental_momentum_axis
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
    basis_as_fundamental_position_basis,
    convert_vector,
)
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        FundamentalMomentumAxis,
        FundamentalPositionAxis,
    )
    from surface_potential_analysis.axis.axis_like import AxisWithLengthLike
    from surface_potential_analysis.basis.basis import (
        AxisWithLengthBasis,
    )
    from surface_potential_analysis.state_vector.state_vector import (
        StateVector,
    )

    _B0Inv = TypeVar("_B0Inv", bound=AxisWithLengthBasis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=AxisWithLengthBasis[Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _S1Inv = TypeVar("_S1Inv", bound=tuple[int, ...])


def convert_state_vector_to_basis(
    state_vector: StateVector[_B0Inv], basis: _B1Inv
) -> StateVector[_B1Inv]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]
    basis : _B1Inv

    Returns
    -------
    StateVector[_B1Inv]
    """
    converted = convert_vector(state_vector["vector"], state_vector["basis"], basis)
    return {"basis": basis, "vector": converted}  # type: ignore[typeddict-item]


def convert_state_vector_to_position_basis(
    state_vector: StateVector[_B0Inv],
) -> StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]]:
    """
    Given an state vector, calculate the vector in position basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]

    Returns
    -------
    StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]]
    """
    return convert_state_vector_to_basis(
        state_vector,
        basis_as_fundamental_position_basis(state_vector["basis"]),
    )


def convert_state_vector_to_momentum_basis(
    state_vector: StateVector[_B0Inv],
) -> StateVector[tuple[FundamentalMomentumAxis[Any, Any], ...]]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]

    Returns
    -------
    StateVector[tuple[FundamentalMomentumAxis[Any, Any], ...]]
    """
    return convert_state_vector_to_basis(
        state_vector,
        basis_as_fundamental_momentum_basis(state_vector["basis"]),
    )


def interpolate_state_vector_momentum(
    state_vector: StateVector[_B0Inv], shape: _S0Inv, axes: _S1Inv
) -> StateVector[tuple[AxisWithLengthLike[Any, Any, Any], ...]]:
    """
    Given a state vector, get the equivalent vector in as a truncated vector in a larger basis.

    Parameters
    ----------
    state_vector : StateVector[_B0Inv]
    shape : _S0Inv
        Final fundamental shape of the basis

    Returns
    -------
    StateVector[tuple[MomentumAxis[Any, Any, Any], ...]]
    """
    converted_basis = tuple(
        axis_as_fundamental_momentum_axis(ax) if iax in axes else ax
        for (iax, ax) in enumerate(state_vector["basis"])
    )
    converted = convert_state_vector_to_basis(state_vector, converted_basis)
    util = AxisWithLengthBasisUtil(converted["basis"])
    final_basis = tuple(
        MomentumAxis(ax.delta_x, ax.n, shape[idx])
        if (
            idx := next((i for i, jax in enumerate(axes) if jax == iax), None)
            is not None
        )
        else ax
        for iax, ax in enumerate(converted["basis"])
    )
    scaled = converted["vector"] * np.sqrt(np.prod(shape) / util.size)
    return {"basis": final_basis, "vector": scaled}
