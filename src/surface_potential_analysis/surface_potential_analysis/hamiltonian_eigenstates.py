from typing import Any, Literal, TypeVar

import numpy as np

from .basis import Basis
from .basis_config import BasisConfig, BasisConfigUtil
from .eigenstate.eigenstate import Eigenstate, EigenstateColllection, EigenstateList
from .hamiltonian import Hamiltonian, hamiltonian_in_basis
from .hamiltonian_builder import total_surface_hamiltonian
from .potential import Potential

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def calculate_eigenstates(
    hamiltonian: Hamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]
) -> EigenstateList[_BX0Inv, _BX1Inv, _BX2Inv]:
    energies, states = np.linalg.eigh(hamiltonian["array"])
    return {"basis": hamiltonian["basis"], "states": states, "energies": energies}


def calculate_energy(
    hamiltonian: Hamiltonian[_BX0Inv, _BX1Inv, _BX2Inv],
    eigenstate: Eigenstate[_BX0Inv, _BX1Inv, _BX2Inv],
) -> complex:
    """
    Calculate the energy of the given eigenvector

    Parameters
    ----------
    hamiltonian : Hamiltonian[BX0, BX1, BX2]
    eigenstate : Eigenstate[BX0, BX1, BX2]

    Returns
    -------
    complex
        The energy of the Eigenvector given Hamiltonian
    """
    return np.linalg.multi_dot(  # type:ignore
        [np.conj(eigenstate["vector"]), hamiltonian["array"], eigenstate["vector"]]
    )


def calculate_energy_eigenstates(
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
