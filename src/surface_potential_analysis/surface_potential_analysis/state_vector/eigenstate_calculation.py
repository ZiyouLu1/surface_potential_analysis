from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
import scipy.linalg

from surface_potential_analysis.basis.basis import Basis
from surface_potential_analysis.util.decorators import timed

from .eigenvalue_list import (
    EigenvalueList,
)
from .state_vector_list import StateVectorList

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import (
        Operator,
        SingleBasisOperator,
    )

    from .state_vector import (
        StateDualVector,
        StateVector,
    )

_B0Inv = TypeVar("_B0Inv", bound=Basis)
_B1Inv = TypeVar("_B1Inv", bound=Basis)
_L0Inv = TypeVar("_L0Inv", bound=int)


class EigenvectorList(StateVectorList[_B0Inv, _L0Inv], EigenvalueList[_L0Inv]):
    """Represents a list of eigenvectors, each with the same basis."""


@timed
def calculate_eigenvectors_hermitian(
    hamiltonian: SingleBasisOperator[_B0Inv],
    subset_by_index: tuple[int, int] | None = None,
) -> EigenvectorList[_B0Inv, int]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    eigenvalues, vectors = scipy.linalg.eigh(
        hamiltonian["array"], subset_by_index=subset_by_index
    )
    return {
        "basis": hamiltonian["basis"],
        "vectors": vectors.T,
        "eigenvalues": eigenvalues,
    }


def calculate_expectation(
    hamiltonian: SingleBasisOperator[_B0Inv], eigenstate: StateVector[_B0Inv]
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


def calculate_operator_inner_product(
    dual_vector: StateDualVector[_B0Inv],
    operator: Operator[_B0Inv, _B1Inv],
    vector: StateVector[_B1Inv],
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
    return np.linalg.multi_dot(  # type:ignore[no-any-return]
        [dual_vector["vector"], operator["array"], vector["vector"]]
    )
