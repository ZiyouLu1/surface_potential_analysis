from __future__ import annotations

from typing import Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import (
    BasisLike,
)
from surface_potential_analysis.axis.stacked_axis import (
    StackedBasisLike,
)
from surface_potential_analysis.axis.util import (
    BasisUtil,
)

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])

_SB0Inv = TypeVar("_SB0Inv", bound=StackedBasisLike[*tuple[Any, ...]])
_SB1Inv = TypeVar("_SB1Inv", bound=StackedBasisLike[*tuple[Any, ...]])


class Operator(TypedDict, Generic[_B0, _B1]):
    """Represents an operator in the given basis."""

    basis: StackedBasisLike[_B0, _B1]
    # We need higher kinded types, and const generics to do this properly
    data: np.ndarray[tuple[int], np.dtype[np.complex_]]


SingleBasisOperator = Operator[_B0, _B0]
"""Represents an operator where both vector and dual vector uses the same basis"""


class DiagonalOperator(TypedDict, Generic[_B0, _B1]):
    """Represents an operator in the given basis."""

    basis: StackedBasisLike[_B0, _B1]
    """Basis of the lhs (first index in array)"""
    data: np.ndarray[tuple[int], np.dtype[np.complex_]]


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
    # TODO: this is just wrong
    return {
        "basis": traced_basis,
        "dual_basis": traced_basis,
        "data": np.sum(
            operator["data"].reshape(operator["basis"].shape), axis=axes
        ).reshape(-1),
    }
