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
_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])
_B3 = TypeVar("_B3", bound=BasisLike[Any, Any])

_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike[Any, Any], covariant=True)
_B2_co = TypeVar("_B2_co", bound=BasisLike[Any, Any], covariant=True)


class OperatorList(TypedDict, Generic[_B0_co, _B1_co, _B2_co]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: StackedBasisLike[_B0_co, StackedBasisLike[_B1_co, _B2_co]]
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]
    """A list of state vectors"""


def select_operator(
    operator_list: OperatorList[_B0, _B1, _B2], idx: SingleFlatIndexLike
) -> Operator[_B1, _B2]:
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


class DiagonalOperatorList(TypedDict, Generic[_B0_co, _B1_co, _B2_co]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: StackedBasisLike[_B0_co, StackedBasisLike[_B1_co, _B2_co]]
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]
    """A list of state vectors"""


SingleBasisDiagonalOperatorList = DiagonalOperatorList[_B0, _B1, _B1]


def select_operator_diagonal(
    operator_list: DiagonalOperatorList[_B0, _B1, _B2], idx: SingleFlatIndexLike
) -> DiagonalOperator[_B1, _B2]:
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


def as_operator_list(
    diagonal_list: DiagonalOperatorList[_B0, _B1, _B2],
) -> OperatorList[_B0, _B1, _B2]:
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
    data[:, np.diag_indices_from(data[0])] = diagonal_list["data"].reshape(
        diagonal_list["basis"][0].n, -1
    )
    return {
        "basis": diagonal_list["basis"],
        "data": data,
    }


def as_flat_operator(
    operator_list: SingleBasisDiagonalOperatorList[_B0, _B1],
) -> SingleBasisDiagonalOperator[StackedBasisLike[_B0, Unpack[tuple[Any, ...]]]]:
    basis = StackedBasis[_B0, Unpack[tuple[Any, ...]]](
        operator_list["basis"][0], *operator_list["basis"][1]
    )
    return {
        "basis": StackedBasis(basis, basis),
        "data": operator_list["data"],
    }


def as_diagonal_operator(
    operator_list: DiagonalOperatorList[_B0, _B1, _B2],
    idx: SingleFlatIndexLike,
) -> DiagonalOperator[_B1, _B2]:
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
    states: DiagonalOperatorList[_B0, Any, Any], axes: tuple[int, ...]
) -> DiagonalOperatorList[_B0, Any, Any]:
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


def matmul_operator_list(
    lhs: Operator[_B0, _B1], rhs: OperatorList[_B3, _B1, _B2]
) -> OperatorList[_B3, _B0, _B2]:
    data = np.einsum(
        "ik,mkj->mij",
        lhs["data"].reshape(lhs["basis"].shape),
        rhs["data"].reshape(-1, *rhs["basis"][1].shape),
    ).reshape(-1)
    return {
        "basis": StackedBasis(
            rhs["basis"][0], StackedBasis(lhs["basis"][0], rhs["basis"][1][1])
        ),
        "data": data,
    }


def matmul_list_operator(
    lhs: OperatorList[_B3, _B0, _B1], rhs: Operator[_B1, _B2]
) -> OperatorList[_B3, _B0, _B2]:
    data = np.tensordot(
        lhs["data"].reshape(-1, *lhs["basis"][1].shape),
        rhs["data"].reshape(rhs["basis"].shape),
        axes=(2, 0),
    ).reshape(-1)
    return {
        "basis": StackedBasis(
            lhs["basis"][0], StackedBasis(lhs["basis"][1][0], rhs["basis"][1])
        ),
        "data": data,
    }
