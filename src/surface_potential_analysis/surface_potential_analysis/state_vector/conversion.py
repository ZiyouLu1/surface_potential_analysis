from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import TransformedPositionBasis
from surface_potential_analysis.axis.axis_like import (
    BasisLike,
    convert_dual_vector,
    convert_vector,
)
from surface_potential_analysis.axis.conversion import axis_as_fundamental_momentum_axis
from surface_potential_analysis.axis.stacked_axis import (
    StackedBasis,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.axis.axis_like import BasisWithLengthLike
    from surface_potential_analysis.axis.stacked_axis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.state_vector.state_vector import (
        StateDualVector,
        StateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])


def convert_state_vector_to_basis(
    state_vector: StateVector[_B0], basis: _B1
) -> StateVector[_B1]:
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
    converted = convert_vector(state_vector["data"], state_vector["basis"], basis)
    return {"basis": basis, "data": converted}  # type: ignore[typeddict-item]


def convert_state_vector_list_to_basis(
    state_vector: StateVectorList[_B0, _B1], basis: _B2
) -> StateVectorList[_B0, _B2]:
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
    stacked = state_vector["data"].reshape(state_vector["basis"].shape)
    converted = convert_vector(stacked, state_vector["basis"][1], basis).reshape(-1)
    return {"basis": StackedBasis(state_vector["basis"][0], basis), "data": converted}


def convert_state_dual_vector_to_basis(
    state_vector: StateDualVector[_B0], basis: _B1
) -> StateDualVector[_B1]:
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
    converted = convert_dual_vector(state_vector["data"], state_vector["basis"], basis)
    return {"basis": basis, "data": converted}  # type: ignore[typeddict-item]


def convert_state_vector_to_position_basis(
    state_vector: StateVector[StackedBasisLike[*tuple[_BL0, ...]]],
) -> StateVector[StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]]:
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
        stacked_basis_as_fundamental_position_basis(state_vector["basis"]),
    )


def convert_state_vector_to_momentum_basis(
    state_vector: StateVector[StackedBasisLike[*tuple[_BL0, ...]]],
) -> StateVector[
    StackedBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
]:
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
        stacked_basis_as_fundamental_momentum_basis(state_vector["basis"]),
    )


def convert_state_dual_vector_to_position_basis(
    state_vector: StateDualVector[StackedBasisLike[*tuple[_BL0, ...]]],
) -> StateDualVector[StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]]:
    """
    Given an state vector, calculate the vector in position basis.

    Parameters
    ----------
    state_vector : StateDualVector[_B0Inv]

    Returns
    -------
    StateDualVector[tuple[FundamentalPositionAxis[Any, Any], ...]]
    """
    return convert_state_dual_vector_to_basis(
        state_vector,
        stacked_basis_as_fundamental_position_basis(state_vector["basis"]),
    )


def convert_state_dual_vector_to_momentum_basis(
    state_vector: StateDualVector[StackedBasisLike[*tuple[_BL0, ...]]],
) -> StateDualVector[
    StackedBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
]:
    """
    Given a state vector, calculate the vector in the given basis.

    Parameters
    ----------
    state_vector : StateDualVector[_B0Inv]

    Returns
    -------
    StateDualVector[tuple[FundamentalMomentumAxis[Any, Any], ...]]
    """
    return convert_state_dual_vector_to_basis(
        state_vector,
        stacked_basis_as_fundamental_momentum_basis(state_vector["basis"]),
    )


def interpolate_state_vector_momentum(
    state_vector: StateVector[StackedBasisLike[*tuple[_BL0, ...]]],
    shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> StateVector[StackedBasisLike[*tuple[BasisWithLengthLike[Any, Any, Any], ...]]]:
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
    converted_basis = StackedBasis[Any](
        *tuple(
            axis_as_fundamental_momentum_axis(ax) if iax in axes else ax
            for (iax, ax) in enumerate(state_vector["basis"])
        )
    )
    converted = convert_state_vector_to_basis(state_vector, converted_basis)

    final_basis = StackedBasis[Any](
        *tuple(
            TransformedPositionBasis(ax.delta_x, ax.n, shape[idx])
            if (
                idx := next((i for i, jax in enumerate(axes) if jax == iax), None)
                is not None
            )
            else ax
            for iax, ax in enumerate(converted["basis"])
        )
    )
    scaled = converted["data"] * np.sqrt(np.prod(shape) / converted_basis.n)
    return {"basis": final_basis, "data": scaled}
