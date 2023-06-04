from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import AxisLike3d
from surface_potential_analysis.basis.basis import (
    Basis,
    Basis1d,
    Basis2d,
    Basis3d,
    FundamentalMomentumBasis3d,
    FundamentalPositionBasis3d,
)

if TYPE_CHECKING:
    from pathlib import Path

_B0Inv = TypeVar("_B0Inv", bound=Basis[Any])


_A3d0Inv = TypeVar("_A3d0Inv", bound=AxisLike3d[Any, Any])
_A3d1Inv = TypeVar("_A3d1Inv", bound=AxisLike3d[Any, Any])
_A3d2Inv = TypeVar("_A3d2Inv", bound=AxisLike3d[Any, Any])


class Eigenstate(TypedDict, Generic[_B0Inv]):
    """represents an eigenstate in a basis."""

    basis: _B0Inv
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


_B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])
_B2d0Inv = TypeVar("_B2d0Inv", bound=Basis2d[Any, Any])
_B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])


class Eigenstate1d(Eigenstate[_B1d0Inv]):
    """represents an eigenstate in a 1d basis."""


class Eigenstate2d(Eigenstate[_B2d0Inv]):
    """represents an eigenstate in a 2d basis."""


class Eigenstate3d(Eigenstate[_B3d0Inv]):
    """represents an eigenstate in a 3d basis."""


Eigenstate3dWithBasis = Eigenstate3d[Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]]


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


FundamentalPositionBasisEigenstate3d = Eigenstate3d[
    FundamentalPositionBasis3d[_NF0Inv, _NF1Inv, _NF2Inv]
]

FundamentalMomentumBasisEigenstate3d = Eigenstate3d[
    FundamentalMomentumBasis3d[_NF0Inv, _NF1Inv, _NF2Inv]
]


class EigenstateList(TypedDict, Generic[_B0Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: _B0Inv
    vectors: np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    energies: np.ndarray[tuple[int], np.dtype[np.float_]]


class EigenstateList3d(EigenstateList[_B3d0Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""


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


def calculate_normalisation(eigenstate: Eigenstate3d[Any]) -> float:
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
