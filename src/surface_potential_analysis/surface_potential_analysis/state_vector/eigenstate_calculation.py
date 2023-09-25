from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import scipy.linalg

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import (
        Operator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        EigenstateList,
    )
    from surface_potential_analysis.types import IntLike_co

    from .state_vector import (
        StateDualVector,
        StateVector,
    )

_B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
_B1Inv = TypeVar("_B1Inv", bound=BasisLike[Any, Any])


@timed
def calculate_eigenvectors_hermitian(
    hamiltonian: SingleBasisOperator[_B0Inv],
    subset_by_index: tuple[IntLike_co, IntLike_co] | None = None,
) -> EigenstateList[FundamentalBasis[int], _B0Inv]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    eigenvalues, vectors = scipy.linalg.eigh(
        hamiltonian["data"].reshape(hamiltonian["basis"].shape),
        subset_by_index=subset_by_index,
    )
    return {
        "basis": StackedBasis(
            FundamentalBasis(np.size(eigenvalues)), hamiltonian["basis"][0]
        ),
        "data": np.transpose(vectors).reshape(-1),
        "eigenvalue": np.array(eigenvalues),
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
        [
            np.conj(eigenstate["data"]),
            hamiltonian["data"].reshape(hamiltonian["basis"].shape),
            eigenstate["data"],
        ]
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
        [
            dual_vector["data"],
            operator["data"].reshape(operator["basis"].shape),
            vector["data"],
        ]
    )
