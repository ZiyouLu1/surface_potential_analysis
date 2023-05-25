from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    FundamentalMomentumBasisConfig,
    FundamentalPositionBasisConfig,
)
from surface_potential_analysis.basis_config.util import BasisConfigUtil

if TYPE_CHECKING:
    from surface_potential_analysis.basis import BasisLike

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)

    _LInv = TypeVar("_LInv", bound=int)

    _BX0Cov = TypeVar("_BX0Cov", bound=BasisLike[Any, Any], covariant=True)
    _BX1Cov = TypeVar("_BX1Cov", bound=BasisLike[Any, Any], covariant=True)
    _BX2Cov = TypeVar("_BX2Cov", bound=BasisLike[Any, Any], covariant=True)


_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)

_BC0Cov = TypeVar("_BC0Cov", bound=BasisConfig[Any, Any, Any], covariant=True)
_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])


HamiltonianPoints = np.ndarray[
    tuple[_L0Cov, _L1Cov], np.dtype[np.complex_] | np.dtype[np.float_]
]


class Hamiltonian(TypedDict, Generic[_BC0Cov]):
    """Represents an operator in the given basis."""

    basis: _BC0Cov
    # We need higher kinded types, and const generics to do this properly
    array: HamiltonianPoints[int, int]


HamiltonianWithBasis = Hamiltonian[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]

FundamentalMomentumBasisHamiltonian = Hamiltonian[
    FundamentalMomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]
]
FundamentalPositionBasisHamiltonian = Hamiltonian[
    FundamentalPositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]
]

_StackedHamiltonianPoints = np.ndarray[
    tuple[_L0Cov, _L1Cov, _L2Cov, _L0Cov, _L1Cov, _L2Cov],
    np.dtype[np.complex_] | np.dtype[np.float_],
]


class StackedHamiltonian(TypedDict, Generic[_BC0Cov]):
    """Represents an operator with it's array of points 'stacked'."""

    basis: _BC0Cov
    # We need higher kinded types to do this properly
    array: _StackedHamiltonianPoints[int, int, int]


StackedHamiltonianWithBasis = StackedHamiltonian[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]
FundamentalMomentumBasisStackedHamiltonian = StackedHamiltonian[
    FundamentalMomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]
]
FundamentalPositionBasisStackedHamiltonian = StackedHamiltonian[
    FundamentalPositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]
]


def flatten_hamiltonian(
    hamiltonian: StackedHamiltonian[_BC0Inv],
) -> Hamiltonian[_BC0Inv]:
    """
    Convert a stacked hamiltonian to a hamiltonian.

    Parameters
    ----------
    hamiltonian : StackedHamiltonian[_BC0Inv]

    Returns
    -------
    Hamiltonian[_BC0Inv]
    """
    n_states = np.prod(hamiltonian["array"].shape[:3])
    return {
        "basis": hamiltonian["basis"],
        "array": hamiltonian["array"].reshape(n_states, n_states),
    }


def stack_hamiltonian(hamiltonian: Hamiltonian[_BC0Inv]) -> StackedHamiltonian[_BC0Inv]:
    """
    Convert a hamiltonian to a stacked hamiltonian.

    Parameters
    ----------
    hamiltonian : Hamiltonian[_BC0Inv]

    Returns
    -------
    StackedHamiltonian[_BC0Inv]
    """
    basis = BasisConfigUtil(hamiltonian["basis"])
    return {
        "basis": hamiltonian["basis"],
        "array": hamiltonian["array"].reshape(*basis.shape, *basis.shape),
    }


def add_hamiltonian(
    a: Hamiltonian[_BC0Inv], b: Hamiltonian[_BC0Inv]
) -> Hamiltonian[_BC0Inv]:
    """
    Add together two operators.

    Parameters
    ----------
    a : Hamiltonian[_BC0Inv]
    b : Hamiltonian[_BC0Inv]

    Returns
    -------
    Hamiltonian[_BC0Inv]
    """
    return {"basis": a["basis"], "array": a["array"] + b["array"]}
