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
_B1d0Cov = TypeVar("_B1d0Cov", bound=Basis1d[Any], covariant=True)
_B2d0Cov = TypeVar("_B2d0Cov", bound=Basis2d[Any, Any], covariant=True)
_B3d0Cov = TypeVar("_B3d0Cov", bound=Basis3d[Any, Any, Any], covariant=True)
_B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])


HamiltonianPoints = np.ndarray[
    tuple[_L0Cov, _L1Cov], np.dtype[np.complex_] | np.dtype[np.float_]
]


class Hamiltonian(TypedDict, Generic[_B0Cov]):
    """Represents an operator in the given basis."""

    basis: _B0Cov
    # We need higher kinded types, and const generics to do this properly
    array: HamiltonianPoints[int, int]


Hamiltonian1d = Hamiltonian[_B1d0Cov]

Hamiltonian2d = Hamiltonian[_B2d0Cov]

Hamiltonian3d = Hamiltonian[_B3d0Cov]


HamiltonianWith3dBasis = Hamiltonian3d[Basis3d[_A3d0Cov, _A3d1Cov, _A3d2Cov]]

FundamentalMomentumBasisHamiltonian3d = Hamiltonian3d[
    FundamentalMomentumBasis3d[_L0Cov, _L1Cov, _L2Cov]
]
FundamentalPositionBasisHamiltonian3d = Hamiltonian3d[
    FundamentalPositionBasis3d[_L0Cov, _L1Cov, _L2Cov]
]


def add_hamiltonian(
    a: Hamiltonian[_B0Inv], b: Hamiltonian[_B0Inv]
) -> Hamiltonian[_B0Inv]:
    """
    Add together two operators.

    Parameters
    ----------
    a : Hamiltonian[_B0Inv]
    b : Hamiltonian[_B0Inv]

    Returns
    -------
    Hamiltonian[_B0Inv]
    """
    return {"basis": a["basis"], "array": a["array"] + b["array"]}
