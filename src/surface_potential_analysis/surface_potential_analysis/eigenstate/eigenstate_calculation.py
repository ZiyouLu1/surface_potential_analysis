from typing import Any, TypeVar

import numpy as np

from surface_potential_analysis.basis import Basis
from surface_potential_analysis.hamiltonian import Hamiltonian

from .eigenstate import Eigenstate, EigenstateList

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])


def calculate_eigenstates(
    hamiltonian: Hamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]
) -> EigenstateList[_BX0Inv, _BX1Inv, _BX2Inv]:
    energies, vectors = np.linalg.eigh(hamiltonian["array"])
    return {"basis": hamiltonian["basis"], "vectors": vectors, "energies": energies}


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
