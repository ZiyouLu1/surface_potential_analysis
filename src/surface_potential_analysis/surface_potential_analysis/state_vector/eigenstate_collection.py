from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalAxis
from surface_potential_analysis.axis.block_fraction_axis import (
    AxisWithBlockFractionLike,
    ExplicitBlockFractionAxis,
)
from surface_potential_analysis.basis.basis import Basis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.eigenvalue_list import EigenvalueList
from surface_potential_analysis.state_vector.state_vector import StateVector
from surface_potential_analysis.state_vector.state_vector_list import StateVectorList

from .eigenstate_calculation import calculate_eigenvectors_hermitian

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.operator.operator import (
        SingleBasisOperator,
    )
_L0Inv = TypeVar("_L0Inv", bound=int)

_B0_co = TypeVar("_B0_co", bound=Basis, covariant=True)
_B0Inv = TypeVar("_B0Inv", bound=Basis)
_B1Inv = TypeVar("_B1Inv", bound=Basis)


class Eigenstate(StateVector[_B0_co], TypedDict):
    """A State vector which is the eigenvector of some operator."""

    eigenvalue: complex | np.complex_


class EigenstateList(  # type: ignore[misc]
    StateVectorList[_B0Inv, _B1Inv],
    EigenvalueList[_B0Inv],
    TypedDict,
):
    """Represents a collection of eigenstates, each with the same basis."""


_B1_co = TypeVar(
    "_B1_co",
    bound=tuple[AxisWithBlockFractionLike[Any, Any], FundamentalAxis[Any]],
    covariant=True,
)


class EigenstateColllection(EigenstateList[_B1_co, _B0_co]):
    """
    Represents a collection of eigenstates, each with the same basis but with _L0Inv different bloch phases.

    NOTE: bloch_fractions: np.ndarray[tuple[_L0Inv, Literal[_NdInv]], np.dtype[np.float_]].
    """


def calculate_eigenstate_collection(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[int], np.dtype[np.float_]]],
        SingleBasisOperator[_B0Inv],
    ],
    bloch_fractions: np.ndarray[tuple[_L0Inv, int], np.dtype[np.float_]],
    *,
    subset_by_index: tuple[int, int] | None = None,
) -> EigenstateColllection[
    tuple[ExplicitBlockFractionAxis[_L0Inv], FundamentalAxis[int]], _B0Inv
]:
    """
    Calculate an eigenstate collection with the given bloch phases.

    Parameters
    ----------
    hamiltonian_generator : Callable[[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]], Hamiltonian[_B3d0Inv]]
        Function used to generate the hamiltonian
    bloch_fractions : np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]]
        List of bloch phases
    subset_by_index : tuple[int, int] | None, optional
        subset_by_index, by default (0,0)

    Returns
    -------
    EigenstateColllection[_B3d0Inv]
    """
    subset_by_index = (0, 0) if subset_by_index is None else subset_by_index
    n_states = 1 + subset_by_index[1] - subset_by_index[0]

    basis = hamiltonian_generator(bloch_fractions[0])["basis"]
    util = BasisUtil(basis)

    vectors = np.zeros(
        (bloch_fractions.shape[0], n_states, util.size), dtype=np.complex_
    )
    eigenvalues = np.zeros((bloch_fractions.shape[0], n_states), dtype=np.float_)

    for idx, bloch_fraction in enumerate(bloch_fractions):
        h = hamiltonian_generator(bloch_fraction)
        eigenstates = calculate_eigenvectors_hermitian(
            h, subset_by_index=subset_by_index
        )

        vectors[idx] = eigenstates["vectors"]
        eigenvalues[idx] = eigenstates["eigenvalues"]

    return {
        "list_basis": (
            ExplicitBlockFractionAxis(bloch_fractions),
            FundamentalAxis(n_states),
        ),
        "basis": basis,
        "vectors": vectors.reshape(-1, util.size),
        "eigenvalues": eigenvalues.reshape(-1),
    }


def select_eigenstate(
    collection: EigenstateColllection[_B1_co, _B0_co],
    bloch_idx: int,
    band_idx: int,
) -> Eigenstate[_B0_co]:
    """
    Select an eigenstate from an eigenstate collection.

    Parameters
    ----------
    collection : EigenstateColllection[_B0_co]
    bloch_idx : int
    band_idx : int

    Returns
    -------
    Eigenstate[_B0_co]
    """
    util = BasisUtil(collection["list_basis"])
    return {
        "basis": collection["basis"],
        "vector": collection["vectors"].reshape(*util.shape, -1)[bloch_idx, band_idx],
        "eigenvalue": collection["eigenvalues"].reshape(util.shape)[bloch_idx, band_idx],  # type: ignore[typeddict-item]
    }
