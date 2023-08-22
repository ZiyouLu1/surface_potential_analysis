from __future__ import annotations

from typing import Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    Basis,
    Basis1d,
    Basis2d,
    Basis3d,
    FundamentalPositionBasis3d,
)

_B0Inv = TypeVar("_B0Inv", bound=Basis)


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


def calculate_normalization(state: StateVector[Any] | StateDualVector[Any]) -> float:
    """
    calculate the normalization of a state.

    This should always be 1

    Parameters
    ----------
    state: StateVector[Any] | StateDualVector[Any]

    Returns
    -------
    float
    """
    return np.sum(np.abs(state["vector"]) ** 2)


def calculate_inner_product(
    state_0: StateVector[_B0Inv],
    state_1: StateDualVector[_B0Inv],
) -> np.complex_:
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
    return np.tensordot(state_1["vector"], state_0["vector"], axes=(0, 0))  # type: ignore[no-any-return]


_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)


FundamentalPositionBasisEigenstate3d = StateVector3d[
    FundamentalPositionBasis3d[_NF0Inv, _NF1Inv, _NF2Inv]
]
