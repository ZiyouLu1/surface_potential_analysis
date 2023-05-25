from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    FundamentalMomentumBasisConfig,
    FundamentalPositionBasisConfig,
)

if TYPE_CHECKING:
    from pathlib import Path


_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_BX0Inv = TypeVar("_BX0Inv", bound=BasisLike[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=BasisLike[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=BasisLike[Any, Any])


class Eigenstate(TypedDict, Generic[_BC0Inv]):
    """represents an eigenstate in an explicit basis."""

    basis: _BC0Inv
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


EigenstateWithBasis = Eigenstate[BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]]


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
    return np.load(path, allow_pickle=True)[()]  # type: ignore[no-any-return]


_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)


FundamentalPositionBasisEigenstate = Eigenstate[
    FundamentalPositionBasisConfig[_NF0Inv, _NF1Inv, _NF2Inv]
]

FundamentalMomentumBasisEigenstate = Eigenstate[
    FundamentalMomentumBasisConfig[_NF0Inv, _NF1Inv, _NF2Inv]
]


class EigenstateList(TypedDict, Generic[_BC0Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: _BC0Inv
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
    return np.load(path, allow_pickle=True)[()]  # type: ignore[no-any-return]


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
