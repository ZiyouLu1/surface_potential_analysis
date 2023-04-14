from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, Literal, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
)
from surface_potential_analysis.hamiltonian import Hamiltonian

from .eigenstate import Eigenstate
from .eigenstate_calculation import calculate_eigenstates

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])


class EigenstateColllection(TypedDict, Generic[_BC0Inv]):
    """Represents a collection of eigenstates, each with the same basis but a variety of different bloch phases."""

    basis: _BC0Inv
    bloch_phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]]
    vectors: np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]
    energies: np.ndarray[tuple[int, int], np.dtype[np.float_]]


def save_eigenstate_collection(
    path: Path, eigenstates: EigenstateColllection[Any]
) -> None:
    """Save an eigenstate collection to an npy file."""
    np.save(path, eigenstates)


def load_eigenstate_collection(path: Path) -> EigenstateColllection[Any]:
    """Load an eigenstate collection from an npy file."""
    return np.load(path, allow_pickle=True)[()]  # type:ignore[no-any-return]


def calculate_eigenstate_collection(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]],
        Hamiltonian[_BC0Inv],
    ],
    bloch_phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    *,
    subset_by_index: tuple[int, int] | None = None,
) -> EigenstateColllection[_BC0Inv]:
    """
    Calculate an eigenstate collection with the given bloch phases.

    Parameters
    ----------
    hamiltonian_generator : Callable[[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]], Hamiltonian[_BC0Inv]]
        Function used to generate the hamiltonian
    bloch_phases : np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]]
        List of bloch phases
    subset_by_index : tuple[int, int] | None, optional
        subset_by_index, by default (0,0)

    Returns
    -------
    EigenstateColllection[_BC0Inv]
    """
    subset_by_index = (0, 0) if subset_by_index is None else subset_by_index
    n_states = 1 + subset_by_index[1] - subset_by_index[0]

    basis = hamiltonian_generator(bloch_phases[0])["basis"]
    util = BasisConfigUtil(basis)
    out: EigenstateColllection[_BC0Inv] = {
        "basis": basis,
        "vectors": np.zeros(
            (bloch_phases.shape[0], n_states, len(util)), dtype=np.complex_
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
    collection: EigenstateColllection[_BC0Inv],
    bloch_idx: int,
    band_idx: int,
) -> Eigenstate[_BC0Inv]:
    """
    Select an eigenstate from an eigenstate collection.

    Parameters
    ----------
    collection : EigenstateColllection[_BC0Inv]
    bloch_idx : int
    band_idx : int

    Returns
    -------
    Eigenstate[_BC0Inv]
    """
    return {
        "basis": collection["basis"],
        "vector": collection["vectors"][bloch_idx, band_idx],
    }
