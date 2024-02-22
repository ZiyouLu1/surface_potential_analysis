from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar, Unpack

import numpy as np

from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import StackedBasis

if TYPE_CHECKING:
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
    from surface_potential_analysis.types import SingleFlatIndexLike

    from .operator import (
        DiagonalOperator,
        Operator,
    )
_B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
_B1Inv = TypeVar("_B1Inv", bound=BasisLike[Any, Any])
_B2Inv = TypeVar("_B2Inv", bound=BasisLike[Any, Any])


class OperatorList(TypedDict, Generic[_B0Inv, _B1Inv, _B2Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: StackedBasisLike[_B0Inv, StackedBasisLike[_B1Inv, _B2Inv]]
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]
    """A list of state vectors"""


def as_operator(
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
        "basis": operator_list["basis"][1],
        "data": operator_list["data"]
        .reshape(operator_list["basis"].shape)[idx]
        .reshape(-1),
    }


class DiagonalOperatorList(TypedDict, Generic[_B0Inv, _B1Inv, _B2Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: StackedBasisLike[_B0Inv, StackedBasisLike[_B1Inv, _B2Inv]]
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]
    """A list of state vectors"""


SingleBasisDiagonalOperatorList = DiagonalOperatorList[_B0Inv, _B1Inv, _B1Inv]


def as_operator_list(
    diagonal_list: DiagonalOperatorList[_B0Inv, _B1Inv, _B2Inv],
) -> OperatorList[_B0Inv, _B1Inv, _B2Inv]:
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
    shape = diagonal_list["basis"][1].shape
    data = np.zeros((diagonal_list["basis"][0].n, *shape), dtype=np.complex128)
    data[:, np.diag_indices_from(data[0])] = diagonal_list["data"]
    return {
        "basis": diagonal_list["basis"],
        "data": np.diag(diagonal_list["data"]),
    }


def as_flat_operator(
    operator_list: SingleBasisDiagonalOperatorList[_B0Inv, _B1Inv],
) -> SingleBasisDiagonalOperator[StackedBasisLike[_B0Inv, Unpack[tuple[Any, ...]]]]:
    basis = StackedBasis[_B0Inv, Unpack[tuple[Any, ...]]](
        operator_list["basis"][0], *operator_list["basis"][1]
    )
    return {
        "basis": StackedBasis(basis, basis),
        "data": operator_list["data"],
    }


def as_diagonal_operator(
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
        "basis": operator_list["basis"][1],
        "data": operator_list["data"]
        .reshape(operator_list["basis"][0].n, -1)[idx]
        .reshape(-1),
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
    traced_basis = tuple(
        b for (i, b) in enumerate(states["basis"][1][0]) if i not in axes
    )
    # TODO: just wrong
    return {
        "basis": StackedBasis(
            states["basis"][0], StackedBasis(traced_basis, traced_basis)
        ),
        "data": np.sum(
            states["data"].reshape(-1, *states["basis"][1][0].shape),
            axis=tuple(1 + np.array(axes, dtype=np.int_)),
        ).reshape(states["data"].shape[0], -1),
    }
