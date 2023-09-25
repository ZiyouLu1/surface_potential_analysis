from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import (
    BasisLike,
)
from surface_potential_analysis.basis.stacked_basis import StackedBasis

if TYPE_CHECKING:
    from collections.abc import Iterable

    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.operator.operator import Operator
    from surface_potential_analysis.state_vector.state_vector import (
        StateDualVector,
        StateVector,
    )
    from surface_potential_analysis.types import SingleFlatIndexLike

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])

    _SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])

_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike[Any, Any], covariant=True)


class StateVectorList(TypedDict, Generic[_B0_co, _B1_co]):
    """
    Represents a list of states.

    The first axis represents the basis of the list, and the second the basis of the states
    """

    basis: StackedBasisLike[_B0_co, _B1_co]
    data: np.ndarray[tuple[int], np.dtype[np.complex_]]
    """A list of state vectors"""


def get_state_vector(
    state_list: StateVectorList[_B0, _B1], idx: SingleFlatIndexLike
) -> StateVector[_B1]:
    """
    Get a single state vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return {
        "basis": state_list["basis"][1],
        "data": state_list["data"].reshape(state_list["basis"].shape)[idx],
    }


def get_state_dual_vector(
    state_list: StateVectorList[_B0, _B1], idx: SingleFlatIndexLike
) -> StateDualVector[_B1]:
    """
    Get a single state dual vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return {
        "basis": state_list["basis"][1],
        "data": np.conj(state_list["data"].reshape(state_list["basis"].shape)[idx]),
    }


def state_vector_list_into_iter(
    states: StateVectorList[_B0, _B1],
) -> Iterable[StateVector[_B1]]:
    """
    Select an eigenstate from an eigenstate collection.

    Parameters
    ----------
    collection : EigenstateColllection[_B0_co]
    bloch_idx : int
    band_idx : int

    Returns
    -------
    Eigenstate[_B0_co]
    """
    return [
        {
            "basis": states["basis"][1],
            "data": states["data"].reshape(states["basis"].shape)[idx],
        }
        for idx in range(states["basis"].shape[0])
    ]


def as_state_vector_list(
    states: Iterable[StateVector[_B1]],
) -> StateVectorList[FundamentalBasis[int], _B1]:
    states = list(states)
    return {
        "basis": StackedBasis(FundamentalBasis(len(states)), states[0]["basis"]),
        "data": np.array([w["data"] for w in states]).reshape(-1),
    }


_B2Inv = TypeVar("_B2Inv", bound=BasisLike[Any, Any])


def calculate_inner_product(
    state_0: StateVectorList[_B0, _B2Inv],
    state_1: StateVectorList[_B1, _B2Inv],
) -> Operator[_B0, _B1]:
    """
    Calculate the inner product of two states.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateDualVector[_B0Inv]

    Returns
    -------
    np.complex_
    """
    return {
        "basis": StackedBasis(state_0["basis"][0], state_1["basis"][0]),
        "data": np.tensordot(
            np.conj(state_0["data"]).reshape(state_0["basis"].shape),
            state_1["data"].reshape(state_1["basis"].shape),
            axes=(1, 1),
        ).reshape(-1),
    }


def average_state_vector(
    probabilities: StateVectorList[_SB0, _B1],
    axis: tuple[int, ...] | None = None,
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float_]] | None = None,
) -> StateVectorList[Any, _B1]:
    """
    Average probabilities over several repeats.

    Parameters
    ----------
    probabilities : list[ProbabilityVectorList[_B0Inv, _L0Inv]]

    Returns
    -------
    ProbabilityVectorList[_B0Inv, _L0Inv]
    """
    axis = tuple(range(probabilities["basis"][0].ndim)) if axis is None else axis
    basis = StackedBasis(
        *tuple(b for (i, b) in enumerate(probabilities["basis"][0]) if i not in axis)
    )
    return {
        "basis": StackedBasis(basis, probabilities["basis"][1]),
        "data": np.average(
            probabilities["data"].reshape(*probabilities["basis"][0].shape, -1),
            axis=tuple(ax for ax in axis),
            weights=weights,
        ).reshape(-1),
    }
