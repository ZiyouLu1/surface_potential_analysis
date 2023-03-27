from pathlib import Path
from typing import Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis import Basis, MomentumBasis, PositionBasis
from surface_potential_analysis.basis_config import BasisConfig

_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)


class Eigenstate(TypedDict, Generic[_BX0Cov, _BX1Cov, _BX2Cov]):
    basis: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


def save_eigenstate(path: Path, eigenstates: Eigenstate[Any, Any, Any]) -> None:
    state = np.array(eigenstates, dtype=Eigenstate)
    np.save(path, state)


def load_eigenstate(path: Path) -> Eigenstate[Any, Any, Any]:
    return np.load(path)[()]  # type:ignore


_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)

PositionBasisEigenstate = Eigenstate[
    PositionBasis[_L0Cov], PositionBasis[_L1Cov], PositionBasis[_L2Cov]
]

MomentumBasisEigenstate = Eigenstate[
    MomentumBasis[_L0Cov], MomentumBasis[_L1Cov], MomentumBasis[_L2Cov]
]


class EigenstateList(TypedDict, Generic[_BX0Cov, _BX1Cov, _BX2Cov]):
    """
    Represents a list of eigenstates, each with the same basis
    and bloch wavevector
    """

    basis: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]
    vectors: np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    energies: np.ndarray[tuple[int], np.dtype[np.float_]]


def save_eigenstate_list(
    path: Path, eigenstates: EigenstateList[Any, Any, Any]
) -> None:
    state = np.array(eigenstates, dtype=EigenstateList)
    np.save(path, state)


def load_eigenstate_list(path: Path) -> EigenstateList[Any, Any, Any]:
    return np.load(path)[()]  # type:ignore


_L0Inv = TypeVar("_L0Inv", bound=int, covariant=True)
_L1Inv = TypeVar("_L1Inv", bound=int, covariant=True)
_L2Inv = TypeVar("_L2Inv", bound=int, covariant=True)


def convert_eigenstate_to_momentum_basis(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
) -> MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    raise NotImplementedError()


def convert_eigenstate_to_position_basis(
    eigenstate: MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
) -> PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    raise NotImplementedError()
