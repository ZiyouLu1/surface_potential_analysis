from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import scipy.linalg

from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.eigenstate.conversion import (
    convert_eigenstate_to_position_basis,
)
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis._types import IndexLike
    from surface_potential_analysis.basis.basis import Basis, Basis3d
    from surface_potential_analysis.eigenstate.eigenstate import (
        Eigenstate,
        EigenstateList,
    )
    from surface_potential_analysis.hamiltonian.hamiltonian import Hamiltonian

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])


@timed
def calculate_eigenstates(
    hamiltonian: Hamiltonian[_B0Inv], subset_by_index: tuple[int, int] | None = None
) -> EigenstateList[_B0Inv]:
    """Get a list of eigenstates for a given hamiltonain."""
    energies, vectors = scipy.linalg.eigh(
        hamiltonian["array"], subset_by_index=subset_by_index
    )
    return {"basis": hamiltonian["basis"], "vectors": vectors.T, "energies": energies}


def calculate_energy(
    hamiltonian: Hamiltonian[_B0Inv], eigenstate: Eigenstate[_B0Inv]
) -> complex:
    """
    Calculate the energy of the given eigenvector.

    Parameters
    ----------
    hamiltonian : Hamiltonian[_B3d0Inv]
    eigenstate : Eigenstate[_B3d0Inv]

    Returns
    -------
    complex
        The energy of the Eigenvector given Hamiltonian
    """
    return np.linalg.multi_dot(  # type: ignore[no-any-return]
        [np.conj(eigenstate["vector"]), hamiltonian["array"], eigenstate["vector"]]
    )


def calculate_eigenstate_at_index_position(
    eigenstate: Eigenstate[_B0Inv], idx: IndexLike
) -> np.ndarray[tuple[int], np.dtype[np.complex_]]:
    """
    Given an eigenstate, get the value at idx in position basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B0Inv]
    idx : IndexLike

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    converted = convert_eigenstate_to_position_basis(eigenstate)
    util = BasisUtil(converted["basis"])
    idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    phases = np.tensordot(eigenstate["bloch_fraction"], idx, axes=(0, 0))
    idx_flat = util.get_flat_index(idx, mode="wrap")
    return converted["vector"][idx_flat] * np.exp(  # type:ignore[no-any-return]
        1j * phases
    )
