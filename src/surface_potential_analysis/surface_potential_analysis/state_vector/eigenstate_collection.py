from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalBasis
from surface_potential_analysis.axis.axis_like import BasisLike
from surface_potential_analysis.axis.block_fraction_axis import (
    AxisWithBlockFractionLike,
    ExplicitBlockFractionAxis,
)
from surface_potential_analysis.axis.stacked_axis import StackedBasis, StackedBasisLike
from surface_potential_analysis.state_vector.state_vector import StateVector
from surface_potential_analysis.state_vector.state_vector_list import StateVectorList

from .eigenstate_calculation import calculate_eigenvectors_hermitian

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.operator.operator import (
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.eigenvalue_list import EigenvalueList
_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike[Any, Any], covariant=True)
_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])


class Eigenstate(StateVector[_B0_co], TypedDict):
    """A State vector which is the eigenvector of some operator."""

    eigenvalue: complex | np.complex_


class EigenstateList(
    StateVectorList[_B0_co, _B1_co],
    TypedDict,
):
    """Represents a collection of eigenstates, each with the same basis."""

    eigenvalue: np.ndarray[tuple[int], np.dtype[np.complex_]]


_SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])
_BF0 = TypeVar("_BF0", bound=AxisWithBlockFractionLike[Any, Any])
EigenstateColllection = EigenstateList[_SB0, _B0]


def calculate_eigenstate_collection(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[_L1], np.dtype[np.float_]]],
        SingleBasisOperator[_B0],
    ],
    bloch_fractions: np.ndarray[tuple[_L1, _L0], np.dtype[np.float_]],
    *,
    subset_by_index: tuple[int, int] | None = None,
) -> EigenstateColllection[
    StackedBasisLike[ExplicitBlockFractionAxis[_L0], FundamentalBasis[int]], _B0
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

    basis = hamiltonian_generator(bloch_fractions[:, 0])["basis"][0]

    vectors = np.zeros(
        (bloch_fractions.shape[1], n_states * basis.n), dtype=np.complex_
    )
    eigenvalues = np.zeros((bloch_fractions.shape[1], n_states), dtype=np.complex_)

    for idx, bloch_fraction in enumerate(bloch_fractions.T):
        h = hamiltonian_generator(bloch_fraction)
        eigenstates = calculate_eigenvectors_hermitian(
            h, subset_by_index=subset_by_index
        )

        vectors[idx] = eigenstates["data"]
        eigenvalues[idx] = eigenstates["eigenvalue"]

    return {
        "basis": StackedBasis(
            StackedBasis(
                ExplicitBlockFractionAxis[_L0](bloch_fractions),
                FundamentalBasis(n_states),
            ),
            basis,
        ),
        "data": vectors.reshape(-1),
        "eigenvalue": eigenvalues.reshape(-1),
    }


def select_eigenstate(
    collection: EigenstateColllection[
        StackedBasisLike[_BF0, _B0],
        _B0_co,
    ],
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
    return {
        "basis": collection["basis"][1],
        "data": collection["data"].reshape(*collection["basis"][0].shape, -1)[
            bloch_idx, band_idx
        ],
        "eigenvalue": collection["eigenvalue"].reshape(
            *collection["basis"][0].shape, -1
        )[bloch_idx, band_idx],
    }


def get_eigenvalues_list(states: EigenstateList[_B0, Any]) -> EigenvalueList[_B0]:
    """
    Extract eigenvalues from an eigenstate list.

    Parameters
    ----------
    states : EigenstateList[_B0, Any]

    Returns
    -------
    EigenvalueList[_B0]
    """
    return {"basis": states["basis"][0], "data": states["eigenvalue"]}
