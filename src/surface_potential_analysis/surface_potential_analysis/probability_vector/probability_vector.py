from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import Basis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.util import get_data_in_axes

if TYPE_CHECKING:
    from surface_potential_analysis._types import (
        SingleIndexLike,
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

_B0Inv = TypeVar("_B0Inv", bound=Basis)
_B1Inv = TypeVar("_B1Inv", bound=Basis)


class ProbabilityVector(TypedDict, Generic[_B0Inv]):
    """represents a state vector in a basis."""

    basis: _B0Inv
    vector: np.ndarray[tuple[int], np.dtype[np.float_]]


class ProbabilityVectorList(TypedDict, Generic[_B0Inv, _B1Inv]):
    """represents a list of probabilities in a basis."""

    list_basis: _B0Inv
    basis: _B1Inv
    vectors: np.ndarray[tuple[int, int], np.dtype[np.float_]]


def from_state_vector(state: StateVector[_B0Inv]) -> ProbabilityVector[_B0Inv]:
    """
    Get a probability vector for a given state vector.

    Parameters
    ----------
    state : StateVector[_B0Inv]

    Returns
    -------
    ProbabilityVector[_B0Inv]
    """
    return {"basis": state["basis"], "vector": np.abs(state["vector"]) ** 2}


def from_state_vector_list(
    states: StateVectorList[_B0Inv, _B1Inv],
) -> ProbabilityVectorList[_B0Inv, _B1Inv]:
    """
    Get a probability vector list for a given state vector list.

    Parameters
    ----------
    states : StateVectorList[_B0Inv, _L0Inv]

    Returns
    -------
    ProbabilityVectorList[_B0Inv, _L0Inv]
    """
    return {
        "list_basis": states["list_basis"],
        "basis": states["basis"],
        "vectors": np.abs(states["vectors"]) ** 2,
    }


def get_probability(
    probabilities: ProbabilityVectorList[_B0Inv, _B1Inv],
    idx: SingleIndexLike | None = None,
) -> ProbabilityVector[_B1Inv]:
    """
    Get Probability from the list.

    Parameters
    ----------
    probabilities : ProbabilityVectorList[_B0Inv, _L0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    ProbabilityVector[_B0Inv]
    """
    idx = 0 if idx is None else idx
    util = BasisUtil(probabilities["list_basis"])
    idx_flat = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    return {
        "basis": probabilities["basis"],
        "vector": probabilities["vectors"][idx_flat],
    }


def get_probability_along_axis(
    probabilities: ProbabilityVectorList[_B0Inv, _B1Inv],
    axes: tuple[int, ...] = (0,),
    idx: SingleStackedIndexLike | None = None,
) -> ProbabilityVectorList[Any, _B1Inv]:
    """
    Get Probability from the list.

    Parameters
    ----------
    probabilities : ProbabilityVectorList[_B0Inv, _L0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    ProbabilityVector[_B0Inv]
    """
    util = BasisUtil(probabilities["list_basis"])
    idx = tuple(0 for _ in range(util.ndim - 2)) if idx is None else idx
    final_basis = tuple(b for (i, b) in enumerate(probabilities["basis"]) if i in axes)

    vector = get_data_in_axes(  # type: ignore[call-overload]
        probabilities["vectors"].reshape(*util.shape, -1), (*axes, util.ndim), idx
    ).reshape(BasisUtil(final_basis).shape, -1)
    return {
        "list_basis": final_basis,
        "basis": probabilities["basis"],
        "vectors": vector,
    }


def sum_probability_over_axis(
    probability: ProbabilityVector[_B0Inv], axis: tuple[int, ...] | None
) -> ProbabilityVector[Any]:
    """
    Sum the probabilities over the given axis.

    Parameters
    ----------
    probability : ProbabilityVector[_B0Inv]
    axis : tuple[int, ...] | None

    Returns
    -------
    ProbabilityVector[Any]
    """
    axis = tuple(range(len(probability["basis"]))) if axis is None else axis
    util = BasisUtil(probability["basis"])
    basis = tuple(b for (i, b) in enumerate(probability["basis"]) if i not in axis)
    return {
        "basis": basis,
        "vector": (
            np.sum(probability["vector"].reshape(*util.shape), axis=axis).reshape(-1)
        ),
    }


def sum_probabilities_over_axis(
    probabilities: ProbabilityVectorList[_B0Inv, _B1Inv],
    axis: tuple[int, ...] | None = None,
) -> ProbabilityVectorList[_B0Inv, Any]:
    """
    Sum the probabilities over the given axis.

    Parameters
    ----------
    probabilities : ProbabilityVectorList[_B0Inv, _L0Inv]
    axis : tuple[int, ...] | None

    Returns
    -------
    ProbabilityVectorList[Any, _L0Inv]
    """
    axis = tuple(range(len(probabilities["basis"]))) if axis is None else axis
    util = BasisUtil(probabilities["basis"])
    basis = tuple(b for (i, b) in enumerate(probabilities["basis"]) if i not in axis)
    return {
        "list_basis": probabilities["list_basis"],
        "basis": basis,
        "vectors": np.sum(
            probabilities["vectors"].reshape(-1, *util.shape),
            axis=tuple(ax + 1 for ax in axis),
        ).reshape(probabilities["vectors"].shape[0], -1),
    }


def average_probabilities(
    probabilities: ProbabilityVectorList[_B0Inv, _B1Inv],
    axis: tuple[int, ...] | None = None,
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float_]] | None = None,
) -> ProbabilityVectorList[Any, _B1Inv]:
    """
    Average probabilities over several repeats.

    Parameters
    ----------
    probabilities : list[ProbabilityVectorList[_B0Inv, _L0Inv]]

    Returns
    -------
    ProbabilityVectorList[_B0Inv, _L0Inv]
    """
    axis = tuple(range(len(probabilities["list_basis"]))) if axis is None else axis
    util = BasisUtil(probabilities["list_basis"])
    basis = tuple(
        b for (i, b) in enumerate(probabilities["list_basis"]) if i not in axis
    )
    return {
        "list_basis": basis,
        "basis": probabilities["basis"],
        "vectors": np.average(
            probabilities["vectors"].reshape(*util.shape, -1),
            axis=tuple(ax for ax in axis),
            weights=weights,
        ).reshape(-1, probabilities["vectors"].shape[1]),
    }
