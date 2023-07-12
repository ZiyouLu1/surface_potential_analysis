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
_L0Inv = TypeVar("_L0Inv", bound=int)


class StateVectorList(TypedDict, Generic[_B0Inv, _L0Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: _B0Inv
    vectors: np.ndarray[tuple[_L0Inv, int], np.dtype[np.complex_]]
    """A list of state vectors"""


def get_state_vector(
    state_list: StateVectorList[_B0Inv, _L0Inv], idx: SingleFlatIndexLike
) -> StateVector[_B0Inv]:
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
    state_list: StateVectorList[_B0Inv, _L0Inv], idx: SingleFlatIndexLike
) -> StateDualVector[_B0Inv]:
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
