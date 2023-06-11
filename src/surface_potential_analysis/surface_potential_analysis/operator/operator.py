from __future__ import annotations

from typing import Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.axis import AxisLike3d
from surface_potential_analysis.basis.basis import (
    Basis,
    Basis1d,
    Basis2d,
    Basis3d,
    FundamentalMomentumBasis3d,
    FundamentalPositionBasis3d,
)

_A3d0Cov = TypeVar("_A3d0Cov", bound=AxisLike3d[Any, Any], covariant=True)
_A3d1Cov = TypeVar("_A3d1Cov", bound=AxisLike3d[Any, Any], covariant=True)
_A3d2Cov = TypeVar("_A3d2Cov", bound=AxisLike3d[Any, Any], covariant=True)

_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)

_B0Cov = TypeVar("_B0Cov", bound=Basis[Any], covariant=True)
_B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
_B1Cov = TypeVar("_B1Cov", bound=Basis[Any], covariant=True)
_B1Inv = TypeVar("_B1Inv", bound=Basis[Any])
_B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])
_B1d1Inv = TypeVar("_B1d1Inv", bound=Basis1d[Any])
_B2d0Inv = TypeVar("_B2d0Inv", bound=Basis2d[Any, Any])
_B2d1Inv = TypeVar("_B2d1Inv", bound=Basis2d[Any, Any])
_B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
_B3d1Inv = TypeVar("_B3d1Inv", bound=Basis3d[Any, Any, Any])

HamiltonianPoints = np.ndarray[
    tuple[_L0Cov, _L1Cov], np.dtype[np.complex_] | np.dtype[np.float_]
]


class Operator(TypedDict, Generic[_B0Cov, _B1Cov]):
    """Represents an operator in the given basis."""

    basis: _B0Cov
    """Basis of the lhs (first index in array)"""
    dual_basis: _B1Cov
    """basis of the rhs (second index in array)"""
    # We need higher kinded types, and const generics to do this properly
    array: HamiltonianPoints[int, int]


SingleBasisOperator = Operator[_B0Cov, _B0Cov]
"""Represents an operator where both vector and dual vector uses the same basis"""


Operator1d = Operator[_B1d0Inv, _B1d1Inv]

Operator2d = Operator[_B2d0Inv, _B2d1Inv]

Operator3d = Operator[_B3d0Inv, _B3d1Inv]

SingleBasisOperator3d = SingleBasisOperator[_B3d0Inv]


HamiltonianWith3dBasis = SingleBasisOperator3d[Basis3d[_A3d0Cov, _A3d1Cov, _A3d2Cov]]

FundamentalMomentumBasisHamiltonian3d = SingleBasisOperator3d[
    FundamentalMomentumBasis3d[_L0Cov, _L1Cov, _L2Cov]
]
FundamentalPositionBasisHamiltonian3d = SingleBasisOperator3d[
    FundamentalPositionBasis3d[_L0Cov, _L1Cov, _L2Cov]
]


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
