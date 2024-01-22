from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis_like import (
    BasisLike,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)

if TYPE_CHECKING:
    from surface_potential_analysis.types import SingleFlatIndexLike

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])

_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike[Any, Any], covariant=True)

_SB0Inv = TypeVar("_SB0Inv", bound=StackedBasisLike[*tuple[Any, ...]])
_SB1Inv = TypeVar("_SB1Inv", bound=StackedBasisLike[*tuple[Any, ...]])


class Operator(TypedDict, Generic[_B0_co, _B1_co]):
    """Represents an operator in the given basis."""

    basis: StackedBasisLike[_B0_co, _B1_co]
    # We need higher kinded types, and const generics to do this properly
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


SingleBasisOperator = Operator[_B0, _B0]
"""Represents an operator where both vector and dual vector uses the same basis"""


class DiagonalOperator(TypedDict, Generic[_B0_co, _B1_co]):
    """Represents an operator in the given basis."""

    basis: StackedBasisLike[_B0_co, _B1_co]
    """Basis of the lhs (first index in array)"""
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


class StatisticalDiagonalOperator(DiagonalOperator[_B0_co, _B1_co]):
    """Represents a statistical operator in the given basis."""

    standard_deviation: np.ndarray[tuple[int], np.dtype[np.float64]]


def as_operator(operator: DiagonalOperator[_B0, _B1]) -> Operator[_B0, _B1]:
    """
    Convert a diagonal operator into an operator.

    Parameters
    ----------
    operator : DiagonalOperator[_B0_co, _B1_co]

    Returns
    -------
    Operator[_B0_co, _B1_co]
    """
    return {"basis": operator["basis"], "data": np.diag(operator["data"])}


def as_diagonal_operator(operator: Operator[_B0, _B1]) -> DiagonalOperator[_B0, _B1]:
    """
    Convert an operator into a diagonal operator.

    Parameters
    ----------
    operator : DiagonalOperator[_B0_co, _B1_co]

    Returns
    -------
    Operator[_B0_co, _B1_co]
    """
    diagonal = np.diag(operator["data"])
    np.testing.assert_array_equal(0, operator["data"] - np.diag(diagonal))
    np.testing.assert_equal(operator["data"].shape[0], operator["data"].shape[1])
    return {"basis": operator["basis"], "data": diagonal.reshape(-1)}


def sum_diagonal_operator_over_axes(
    operator: DiagonalOperator[_SB0Inv, _SB1Inv], axes: tuple[int, ...]
) -> DiagonalOperator[Any, Any]:
    """
    given a diagonal operator, sum the states over axes.

    Parameters
    ----------
    states : DiagonalOperator[Any, Any]
    axes : tuple[int, ...]

    Returns
    -------
    DiagonalOperator[Any, Any]
    """
    BasisUtil(operator["basis"])
    traced_basis = tuple(
        b for (i, b) in enumerate(operator["basis"][0]) if i not in axes
    )
    return {
        "basis": StackedBasis(StackedBasis(*traced_basis), StackedBasis(*traced_basis)),
        "data": np.sum(
            operator["data"].reshape(operator["basis"][0].shape), axis=axes
        ).reshape(-1),
    }


SingleBasisDiagonalOperator = DiagonalOperator[_B0, _B0]


def get_eigenvalue(
    eigenvalue_list: SingleBasisDiagonalOperator[BasisLike[Any, Any]],
    idx: SingleFlatIndexLike,
) -> np.complex128:
    """
    Get a single eigenvalue from the list.

    Parameters
    ----------
    eigenvalue_list : EigenvalueList[_L0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    np.complex_
    """
    return eigenvalue_list["data"][idx]


def average_eigenvalues(
    eigenvalues: SingleBasisDiagonalOperator[StackedBasisLike[*tuple[Any, ...]]],
    axis: tuple[int, ...] | None = None,
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> SingleBasisDiagonalOperator[StackedBasis[*tuple[Any, ...]]]:
    """
    Average eigenvalues over the given axis.

    Parameters
    ----------
    eigenvalues : EigenvalueList[_B0Inv]
    axis : tuple[int, ...] | None, optional
        axis, by default None
    weights : np.ndarray[tuple[int], np.dtype[np.float_]] | None, optional
        weights, by default None

    Returns
    -------
    EigenvalueList[Any]
    """
    axis = tuple(range(eigenvalues["basis"].ndim)) if axis is None else axis
    basis = tuple(b for (i, b) in enumerate(eigenvalues["basis"][0]) if i not in axis)
    return {
        "basis": StackedBasis(StackedBasis(*basis), StackedBasis(*basis)),
        "data": np.average(
            eigenvalues["data"].reshape(*eigenvalues["basis"][0].shape),
            axis=tuple(ax for ax in axis),
            weights=weights,
        ).reshape(-1),
    }
