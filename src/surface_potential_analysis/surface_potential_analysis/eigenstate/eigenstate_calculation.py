from typing import Any, TypeVar

import numpy as np

from surface_potential_analysis.basis_config import BasisConfig
from surface_potential_analysis.hamiltonian import Hamiltonian
from surface_potential_analysis.util import timed

from .eigenstate import Eigenstate, EigenstateList

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])


@timed
def calculate_eigenstates(hamiltonian: Hamiltonian[_BC0Inv]) -> EigenstateList[_BC0Inv]:
    energies, vectors = np.linalg.eigh(hamiltonian["array"])
    return {"basis": hamiltonian["basis"], "vectors": vectors.T, "energies": energies}


def calculate_energy(
    hamiltonian: Hamiltonian[_BC0Inv], eigenstate: Eigenstate[_BC0Inv]
) -> complex:
    """
    Calculate the energy of the given eigenvector

    Parameters
    ----------
    hamiltonian : Hamiltonian[_BC0Inv]
    eigenstate : Eigenstate[_BC0Inv]

    Returns
    -------
    complex
        The energy of the Eigenvector given Hamiltonian
    """
    return np.linalg.multi_dot(  # type:ignore
        [np.conj(eigenstate["vector"]), hamiltonian["array"], eigenstate["vector"]]
    )
