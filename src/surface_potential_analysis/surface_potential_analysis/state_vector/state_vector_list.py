from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    Basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleFlatIndexLike
    from surface_potential_analysis.state_vector.state_vector import (
        StateDualVector,
        StateVector,
    )

_B0Inv = TypeVar("_B0Inv", bound=Basis)
_B1Inv = TypeVar("_B1Inv", bound=Basis)


class StateVectorList(TypedDict, Generic[_B0Inv, _B1Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    list_basis: _B0Inv
    basis: _B1Inv
    vectors: np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    """A list of state vectors"""


def get_state_vector(
    state_list: StateVectorList[_B0Inv, _B1Inv], idx: SingleFlatIndexLike
) -> StateVector[_B1Inv]:
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
    return {"basis": state_list["basis"], "vector": state_list["vectors"][idx]}


def get_state_dual_vector(
    state_list: StateVectorList[_B0Inv, _B1Inv], idx: SingleFlatIndexLike
) -> StateDualVector[_B1Inv]:
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
        "basis": state_list["basis"],
        "vector": np.conj(state_list["vectors"][idx]),
    }


def get_all_states(
    states: StateVectorList[_B0Inv, _B1Inv],
) -> list[StateVector[_B1Inv]]:
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
        {"basis": states["basis"], "vector": states["vectors"][idx]}
        for idx in range(states["vectors"].shape[0])
    ]
