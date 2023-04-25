from pathlib import Path
from typing import Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis import (
    Basis,
)
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    MomentumBasisConfig,
    PositionBasisConfig,
)

_BC0Cov = TypeVar("_BC0Cov", bound=BasisConfig[Any, Any, Any], covariant=True)
_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)


class Eigenstate(TypedDict, Generic[_BC0Cov]):
    """represents an eigenstate in an explicit basis."""

    basis: _BC0Cov
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


EigenstateWithBasis = Eigenstate[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]


def save_eigenstate(path: Path, eigenstates: Eigenstate[Any]) -> None:
    """
    Save an eigenstate in an npy file.

    Parameters
    ----------
    path : Path
    eigenstates : Eigenstate[Any]
    """
    np.save(path, eigenstates)


def load_eigenstate(path: Path) -> Eigenstate[Any]:
    """
    Load an eigenstate from an npy file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Eigenstate[Any]
    """
    return np.load(path, allow_pickle=True)[()]  # type:ignore[no-any-return]


_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)


PositionBasisEigenstate = Eigenstate[PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]]

MomentumBasisEigenstate = Eigenstate[MomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]]


class EigenstateList(TypedDict, Generic[_BC0Cov]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: _BC0Cov
    vectors: np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    energies: np.ndarray[tuple[int], np.dtype[np.float_]]


def save_eigenstate_list(path: Path, eigenstates: EigenstateList[Any]) -> None:
    """
    Save an eigenstate list as an npy file.

    Parameters
    ----------
    path : Path
    eigenstates : EigenstateList[Any]
    """
    np.save(path, eigenstates)


def load_eigenstate_list(path: Path) -> EigenstateList[Any]:
    """
    load an eigenstate list form an npy file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    EigenstateList[Any]
    """
    return np.load(path, allow_pickle=True)[()]  # type:ignore[no-any-return]


def calculate_normalisation(eigenstate: Eigenstate[Any]) -> float:
    """
    calculate the normalization of an eigenstate.

    This should always be 1

    Parameters
    ----------
    eigenstate: Eigenstate[Any]

    Returns
    -------
    float
    """
    return np.sum(np.conj(eigenstate["vector"]) * eigenstate["vector"])
