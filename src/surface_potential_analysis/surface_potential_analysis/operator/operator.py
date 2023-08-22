from __future__ import annotations

from typing import Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    Basis,
    Basis1d,
    Basis2d,
    Basis3d,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)

_L0_co = TypeVar("_L0_co", bound=int, covariant=True)
_L1_co = TypeVar("_L1_co", bound=int, covariant=True)

_B0_co = TypeVar("_B0_co", bound=Basis, covariant=True)
_B0Inv = TypeVar("_B0Inv", bound=Basis)
_B1_co = TypeVar("_B1_co", bound=Basis, covariant=True)
_B1Inv = TypeVar("_B1Inv", bound=Basis)
_B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])
_B1d1Inv = TypeVar("_B1d1Inv", bound=Basis1d[Any])
_B2d0Inv = TypeVar("_B2d0Inv", bound=Basis2d[Any, Any])
_B2d1Inv = TypeVar("_B2d1Inv", bound=Basis2d[Any, Any])
_B3d0_co = TypeVar("_B3d0_co", bound=Basis3d[Any, Any, Any], covariant=True)
_B3d1Inv = TypeVar("_B3d1Inv", bound=Basis3d[Any, Any, Any])

OperatorPoints = np.ndarray[
    tuple[_L0_co, _L1_co], np.dtype[np.complex_] | np.dtype[np.float_]
]


class Operator(TypedDict, Generic[_B0_co, _B1_co]):
    """Represents an operator in the given basis."""

    basis: _B0_co
    """Basis of the lhs (first index in array)"""
    dual_basis: _B1_co
    """basis of the rhs (second index in array)"""
    # We need higher kinded types, and const generics to do this properly
    array: OperatorPoints[int, int]


SingleBasisOperator = Operator[_B0_co, _B0_co]
"""Represents an operator where both vector and dual vector uses the same basis"""


Operator1d = Operator[_B1d0Inv, _B1d1Inv]

Operator2d = Operator[_B2d0Inv, _B2d1Inv]

Operator3d = Operator[_B3d0_co, _B3d1Inv]


def add_operator(
    a: Operator[_B0Inv, _B1Inv], b: Operator[_B0Inv, _B1Inv]
) -> Operator[_B0Inv, _B1Inv]:
    """
    Add together two operators.

    Parameters
    ----------
    a : Operator[_B0Inv]
    b : Operator[_B0Inv]

    Returns
    -------
    Operator[_B0Inv]
    """
    return {
        "basis": a["basis"],
        "dual_basis": a["dual_basis"],
        "array": a["array"] + b["array"],
    }


class DiagonalOperator(TypedDict, Generic[_B0_co, _B1_co]):
    """Represents an operator in the given basis."""

    basis: _B0_co
    """Basis of the lhs (first index in array)"""
    dual_basis: _B1_co
    """basis of the rhs (second index in array)"""
    # We need higher kinded types, and const generics to do this properly
    vector: np.ndarray[tuple[int], np.dtype[np.complex_] | np.dtype[np.float_]]


def as_operator(operator: DiagonalOperator[_B0_co, _B1_co]) -> Operator[_B0_co, _B1_co]:
    """
    Convert a diagonal operator into an operator.

    Parameters
    ----------
    operator : DiagonalOperator[_B0_co, _B1_co]

    Returns
    -------
    Operator[_B0_co, _B1_co]
    """
    return {
        "basis": operator["basis"],
        "dual_basis": operator["dual_basis"],
        "array": np.diag(operator["vector"]),
    }


def as_diagonal_operator(
    operator: Operator[_B0_co, _B1_co]
) -> DiagonalOperator[_B0_co, _B1_co]:
    """
    Convert an operator into a diagonal operator.

    Parameters
    ----------
    operator : DiagonalOperator[_B0_co, _B1_co]

    Returns
    -------
    Operator[_B0_co, _B1_co]
    """
    diagonal = np.diag(operator["array"])
    np.testing.assert_array_equal(0, operator["array"] - np.diag(diagonal))
    np.testing.assert_equal(operator["array"].shape[0], operator["array"].shape[1])
    return {
        "basis": operator["basis"],
        "dual_basis": operator["dual_basis"],
        "vector": np.diag(operator["array"]),
    }


def sum_diagonal_operator_over_axes(
    operator: DiagonalOperator[Any, Any], axes: tuple[int, ...]
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
    util = BasisUtil(operator["basis"])
    traced_basis = tuple(b for (i, b) in enumerate(operator["basis"]) if i not in axes)
    return {
        "basis": traced_basis,
        "dual_basis": traced_basis,
        "vector": np.sum(operator["vector"].reshape(util.shape), axis=axes).reshape(-1),
    }
