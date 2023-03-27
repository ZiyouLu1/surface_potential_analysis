from pathlib import Path
from typing import Any, Generic, Literal, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis import Basis
from surface_potential_analysis.basis_config import BasisConfig, BasisConfigUtil
from surface_potential_analysis.hamiltonian import hamiltonian_in_basis
from surface_potential_analysis.hamiltonian_builder import total_surface_hamiltonian
from surface_potential_analysis.potential import Potential

from .eigenstate import Eigenstate
from .eigenstate_calculation import calculate_eigenstates

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)


class EigenstateColllection(TypedDict, Generic[_BX0Cov, _BX1Cov, _BX2Cov]):
    """
    Represents a collection of eigenstates, each with the same basis
    but a variety of different bloch phases
    """

    basis: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]
    bloch_phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]]
    vectors: np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]
    energies: np.ndarray[tuple[int, int], np.dtype[np.float_]]


def save_eigenstate_collection(
    path: Path, eigenstates: EigenstateColllection[Any, Any, Any]
) -> None:
    state = np.array(eigenstates, dtype=EigenstateColllection)
    np.save(path, state)


def load_eigenstate_collection(path: Path) -> EigenstateColllection[Any, Any, Any]:
    return np.load(path)[()]  # type:ignore


def calculate_eigenstate_collection(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    mass: float,
    bloch_phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    basis: BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv],
    *,
    include_bands: list[int] | None = None,
) -> EigenstateColllection[_BX0Inv, _BX1Inv, _BX2Inv]:
    include_bands = [0] if include_bands is None else include_bands

    util = BasisConfigUtil(basis)
    out: EigenstateColllection[_BX0Inv, _BX1Inv, _BX2Inv] = {
        "basis": basis,
        "states": np.zeros(
            (bloch_phases.shape[0], len(include_bands), len(util)), dtype=np.complex_
        ),
        "energies": np.zeros(
            (bloch_phases.shape[0], len(include_bands)), dtype=np.float_
        ),
        "bloch_phases": bloch_phases,
    }

    for idx, bloch_phase in enumerate(bloch_phases):
        h = total_surface_hamiltonian(potential, mass=mass, bloch_phase=bloch_phase)
        h_in_basis = hamiltonian_in_basis(h, basis)
        eigenstates = calculate_eigenstates(h_in_basis)

        out["states"][idx] = eigenstates["states"][eigenstates][include_bands]
        out["energies"][idx] = eigenstates["energies"][eigenstates][include_bands]

    return out


def select_eigenstate(
    collection: EigenstateColllection,  bloch_idx: int,band_idx: int
) -> Eigenstate:
    return {
        "basis": collection["basis"],
        "vector": collection["vectors"][bloch_idx, band_idx],
    }
