from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    Basis,
)
from surface_potential_analysis.basis.util import BasisUtil

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleFlatIndexLike

    from .operator import (
        DiagonalOperator,
        Operator,
    )
_B0Inv = TypeVar("_B0Inv", bound=Basis)
_B1Inv = TypeVar("_B1Inv", bound=Basis)
_B2Inv = TypeVar("_B2Inv", bound=Basis)


class OperatorList(TypedDict, Generic[_B0Inv, _B1Inv, _B2Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    list_basis: _B0Inv
    basis: _B1Inv
    dual_basis: _B2Inv
    arrays: np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]
    """A list of state vectors"""


def get_operator(
    operator_list: OperatorList[_B0Inv, _B1Inv, _B2Inv], idx: SingleFlatIndexLike
) -> Operator[_B1Inv, _B2Inv]:
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
        "basis": operator_list["basis"],
        "dual_basis": operator_list["dual_basis"],
        "array": operator_list["arrays"][idx],
    }


class DiagonalOperatorList(TypedDict, Generic[_B0Inv, _B1Inv, _B2Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    list_basis: _B0Inv
    basis: _B1Inv
    dual_basis: _B2Inv
    vectors: np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    """A list of state vectors"""


def get_diagonal_operator(
    operator_list: DiagonalOperatorList[_B0Inv, _B1Inv, _B2Inv],
    idx: SingleFlatIndexLike,
) -> DiagonalOperator[_B1Inv, _B2Inv]:
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
        "basis": operator_list["basis"],
        "dual_basis": operator_list["dual_basis"],
        "vector": operator_list["vectors"][idx],
    }


def sum_diagonal_operator_list_over_axes(
    states: DiagonalOperatorList[_B0Inv, Any, Any], axes: tuple[int, ...]
) -> DiagonalOperatorList[_B0Inv, Any, Any]:
    """
    given a diagonal operator list, sum the states over axes.

    Parameters
    ----------
    states : DiagonalOperatorList[Any, Any, _L0Inv]
    axes : tuple[int, ...]

    Returns
    -------
    DiagonalOperatorList[Any, Any, _L0Inv]
    """
    util = BasisUtil(states["basis"])
    traced_basis = tuple(b for (i, b) in enumerate(states["basis"]) if i not in axes)
    return {
        "list_basis": states["basis"],
        "basis": traced_basis,
        "dual_basis": traced_basis,
        "vectors": np.sum(
            states["vectors"].reshape(-1, *util.shape),
            axis=tuple(1 + np.array(axes)),
        ).reshape(states["vectors"].shape[0], -1),
    }
