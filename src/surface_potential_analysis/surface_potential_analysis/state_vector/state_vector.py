from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    Basis,
    Basis1d,
    Basis2d,
    Basis3d,
    FundamentalPositionBasis3d,
)

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleFlatIndexLike

_B0Inv = TypeVar("_B0Inv", bound=Basis[Any])


class StateVector(TypedDict, Generic[_B0Inv]):
    """represents a state vector in a basis."""

    basis: _B0Inv
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


class StateDualVector(TypedDict, Generic[_B0Inv]):
    """represents a dual vector in a basis."""

    basis: _B0Inv
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


_B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])
_B2d0Inv = TypeVar("_B2d0Inv", bound=Basis2d[Any, Any])
_B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])


Vector1d = StateVector[_B1d0Inv]
Vector2d = StateVector[_B2d0Inv]
StateVector3d = StateVector[_B3d0Inv]


def as_vector(vector: StateDualVector[_B0Inv]) -> StateVector[_B0Inv]:
    """
    Convert a state dual vector into a state vector.

    Parameters
    ----------
    vector : StateDualVector[_B0Inv]

    Returns
    -------
    StateVector[_B0Inv]
    """
    return {"basis": vector["basis"], "vector": np.conj(vector["vector"])}


def as_dual_vector(vector: StateVector[_B0Inv]) -> StateDualVector[_B0Inv]:
    """
    Convert a state vector into a state dual vector.

    Parameters
    ----------
    vector : StateVector[_B0Inv]

    Returns
    -------
    StateDualVector[_B0Inv]
    """
    return {"basis": vector["basis"], "vector": np.conj(vector["vector"])}


_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)


FundamentalPositionBasisEigenstate3d = StateVector3d[
    FundamentalPositionBasis3d[_NF0Inv, _NF1Inv, _NF2Inv]
]


class StateVectorList(TypedDict, Generic[_B0Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: _B0Inv
    vectors: np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    """A list of state vectors"""
    energies: np.ndarray[tuple[int], np.dtype[np.float_]]


def get_state_vector(
    eigenstates: StateVectorList[_B0Inv], idx: SingleFlatIndexLike
) -> StateVector[_B0Inv]:
    """
    Get a single state vector from a list of states.

    Parameters
    ----------
    eigenstates : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return {"basis": eigenstates["basis"], "vector": eigenstates["vectors"][idx]}


def get_state_dual_vector(
    eigenstates: StateVectorList[_B0Inv], idx: SingleFlatIndexLike
) -> StateDualVector[_B0Inv]:
    """
    Get a single state dual vector from a list of states.

    Parameters
    ----------
    eigenstates : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return {
        "basis": eigenstates["basis"],
        "vector": np.conj(eigenstates["vectors"][idx]),
    }


def calculate_normalization(eigenstate: StateVector3d[Any]) -> float:
    """
    calculate the normalization of an eigenstate.

    This should always be 1

    Parameters
    ----------
    eigenstate: Eigenstate[Any]

    Returns
    -------
    float
    """
    return np.sum(np.conj(eigenstate["vector"]) * eigenstate["vector"])
