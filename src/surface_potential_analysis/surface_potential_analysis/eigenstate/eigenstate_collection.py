from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    Basis,
    Basis1d,
    Basis2d,
    Basis3d,
)
from surface_potential_analysis.basis.util import BasisUtil

from .eigenstate_calculation import calculate_eigenstates

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from surface_potential_analysis.eigenstate.eigenstate import Eigenstate
    from surface_potential_analysis.hamiltonian.hamiltonian import (
        Hamiltonian,
    )
_L0Inv = TypeVar("_L0Inv", bound=int)

_B0Cov = TypeVar("_B0Cov", bound=Basis[Any], covariant=True)
_B0Inv = TypeVar("_B0Inv", bound=Basis[Any])

_B1d0Cov = TypeVar("_B1d0Cov", bound=Basis1d[Any], covariant=True)
_B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])
_B2d0Cov = TypeVar("_B2d0Cov", bound=Basis2d[Any, Any], covariant=True)
_B2d0Inv = TypeVar("_B2d0Inv", bound=Basis2d[Any, Any])
_B3d0Cov = TypeVar("_B3d0Cov", bound=Basis3d[Any, Any, Any], covariant=True)
_B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])


class EigenstateColllection(TypedDict, Generic[_B0Cov, _L0Inv]):
    """Represents a collection of eigenstates, each with the same basis but with _L0Inv different bloch phases."""

    basis: _B0Cov
    bloch_phases: np.ndarray[tuple[_L0Inv, int], np.dtype[np.float_]]
    vectors: np.ndarray[tuple[_L0Inv, int, int], np.dtype[np.complex_]]
    energies: np.ndarray[tuple[_L0Inv, int], np.dtype[np.float_]]


class EigenstateColllection1d(EigenstateColllection[_B1d0Cov, _L0Inv]):
    """
    Represents a collection of eigenstates, each with the same basis but a variety of different bloch phases.

    NOTE: bloch_phases: np.ndarray[tuple[_L0Inv, Literal[1]], np.dtype[np.float_]].
    """


class EigenstateColllection2d(EigenstateColllection[_B2d0Cov, _L0Inv]):
    """
    Represents a collection of eigenstates, each with the same basis but a variety of different bloch phases.

    NOTE: bloch_phases: np.ndarray[tuple[_L0Inv, Literal[2]], np.dtype[np.float_]]
    """


class EigenstateColllection3d(EigenstateColllection[_B3d0Cov, _L0Inv]):
    """
    Represents a collection of eigenstates, each with the same basis but a variety of different bloch phases.

    NOTE: bloch_phases: np.ndarray[tuple[_L0Inv, Literal[3]], np.dtype[np.float_]]
    """


def save_eigenstate_collection(
    path: Path, eigenstates: EigenstateColllection[Any, Any]
) -> None:
    """Save an eigenstate collection to an npy file."""
    np.save(path, eigenstates)


def load_eigenstate_collection(path: Path) -> EigenstateColllection[Any, Any]:
    """Load an eigenstate collection from an npy file."""
    return np.load(path, allow_pickle=True)[()]  # type: ignore[no-any-return]


def calculate_eigenstate_collection(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[int], np.dtype[np.float_]]],
        Hamiltonian[_B0Inv],
    ],
    bloch_phases: np.ndarray[tuple[_L0Inv, int], np.dtype[np.float_]],
    *,
    subset_by_index: tuple[int, int] | None = None,
) -> EigenstateColllection[_B0Inv, _L0Inv]:
    """
    Calculate an eigenstate collection with the given bloch phases.

    Parameters
    ----------
    hamiltonian_generator : Callable[[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]], Hamiltonian[_B3d0Inv]]
        Function used to generate the hamiltonian
    bloch_phases : np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]]
        List of bloch phases
    subset_by_index : tuple[int, int] | None, optional
        subset_by_index, by default (0,0)

    Returns
    -------
    EigenstateColllection[_B3d0Inv]
    """
    subset_by_index = (0, 0) if subset_by_index is None else subset_by_index
    n_states = 1 + subset_by_index[1] - subset_by_index[0]

    basis = hamiltonian_generator(bloch_phases[0])["basis"]
    util = BasisUtil(basis)
    out: EigenstateColllection[_B0Inv, _L0Inv] = {
        "basis": basis,
        "vectors": np.zeros(
            (bloch_phases.shape[0], n_states, util.size), dtype=np.complex_
        ),
        "energies": np.zeros((bloch_phases.shape[0], n_states), dtype=np.float_),
        "bloch_phases": bloch_phases,
    }

    for idx, bloch_phase in enumerate(bloch_phases):
        h = hamiltonian_generator(bloch_phase)
        eigenstates = calculate_eigenstates(h, subset_by_index=subset_by_index)

        out["vectors"][idx] = eigenstates["vectors"]
        out["energies"][idx] = eigenstates["energies"]

    return out


def select_eigenstate(
    collection: EigenstateColllection[_B0Cov, _L0Inv],
    bloch_idx: int,
    band_idx: int,
) -> Eigenstate[_B0Cov]:
    """
    Select an eigenstate from an eigenstate collection.

    Parameters
    ----------
    collection : EigenstateColllection[_B0Cov]
    bloch_idx : int
    band_idx : int

    Returns
    -------
    Eigenstate[_B0Cov]
    """
    return {
        "basis": collection["basis"],
        "vector": collection["vectors"][bloch_idx, band_idx],
    }
