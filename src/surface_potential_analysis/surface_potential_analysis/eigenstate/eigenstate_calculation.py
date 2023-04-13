from typing import Any, TypeVar

import numpy as np
import scipy.linalg

from surface_potential_analysis.basis_config import BasisConfig
from surface_potential_analysis.hamiltonian import Hamiltonian
from surface_potential_analysis.util import timed

from .eigenstate import Eigenstate, EigenstateList

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])


@timed
def calculate_eigenstates(
    hamiltonian: Hamiltonian[_BC0Inv], subset_by_index: tuple[int, int] | None = None
) -> EigenstateList[_BC0Inv]:
    """Get a list of eigenstates for a given hamiltonain."""
    energies, vectors = scipy.linalg.eigh(
        hamiltonian["array"], subset_by_index=subset_by_index
    )
    return {"basis": hamiltonian["basis"], "vectors": vectors.T, "energies": energies}


def calculate_energy(
    hamiltonian: Hamiltonian[_BC0Inv], eigenstate: Eigenstate[_BC0Inv]
) -> complex:
    """
    Calculate the energy of the given eigenvector.

    Parameters
    ----------
    hamiltonian : Hamiltonian[_BC0Inv]
    eigenstate : Eigenstate[_BC0Inv]

    Returns
    -------
    complex
        The energy of the Eigenvector given Hamiltonian
    """
    return np.linalg.multi_dot(  # type:ignore[no-any-return]
        [np.conj(eigenstate["vector"]), hamiltonian["array"], eigenstate["vector"]]
    )
