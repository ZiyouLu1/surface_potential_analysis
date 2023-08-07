from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import Basis, Basis1d, Basis2d, Basis3d
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.state_vector import StateVector

from .eigenstate_calculation import calculate_eigenvectors_hermitian

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.operator.operator import (
        SingleBasisOperator,
    )
_L0Inv = TypeVar("_L0Inv", bound=int)

_B0Cov = TypeVar("_B0Cov", bound=Basis, covariant=True)
_B0Inv = TypeVar("_B0Inv", bound=Basis)

_B1d0Cov = TypeVar("_B1d0Cov", bound=Basis1d[Any], covariant=True)
_B2d0Cov = TypeVar("_B2d0Cov", bound=Basis2d[Any, Any], covariant=True)
_B3d0Cov = TypeVar("_B3d0Cov", bound=Basis3d[Any, Any, Any], covariant=True)


class Eigenstate(StateVector[_B0Cov], TypedDict):
    """A State vector which is the eigenvector of some operator."""

    eigenvalue: complex | np.complex_


class EigenstateColllection(TypedDict, Generic[_B0Cov, _L0Inv]):
    """Represents a collection of eigenstates, each with the same basis but with _L0Inv different bloch phases."""

    basis: _B0Cov
    bloch_fractions: np.ndarray[tuple[_L0Inv, int], np.dtype[np.float_]]
    vectors: np.ndarray[tuple[_L0Inv, int, int], np.dtype[np.complex_]]
    eigenvalues: np.ndarray[tuple[_L0Inv, int], np.dtype[np.complex_]]


EigenstateColllection1d = EigenstateColllection[_B1d0Cov, _L0Inv]
"""
Represents a collection of eigenstates, each with the same basis but a variety of different bloch phases.

NOTE: bloch_fractions: np.ndarray[tuple[_L0Inv, Literal[1]], np.dtype[np.float_]].
"""


EigenstateColllection2d = EigenstateColllection[_B2d0Cov, _L0Inv]
"""
Represents a collection of eigenstates, each with the same basis but a variety of different bloch phases.

NOTE: bloch_fractions: np.ndarray[tuple[_L0Inv, Literal[2]], np.dtype[np.float_]]
"""


EigenstateColllection3d = EigenstateColllection[_B3d0Cov, _L0Inv]
"""
Represents a collection of eigenstates, each with the same basis but a variety of different bloch phases.

NOTE: bloch_fractions: np.ndarray[tuple[_L0Inv, Literal[3]], np.dtype[np.float_]]
"""


def calculate_eigenstate_collection(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[int], np.dtype[np.float_]]],
        SingleBasisOperator[_B0Inv],
    ],
    bloch_fractions: np.ndarray[tuple[_L0Inv, int], np.dtype[np.float_]],
    *,
    subset_by_index: tuple[int, int] | None = None,
) -> EigenstateColllection[_B0Inv, _L0Inv]:
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
    out: EigenstateColllection[_B0Inv, _L0Inv] = {
        "basis": basis,
        "vectors": np.zeros(
            (bloch_fractions.shape[0], n_states, util.size), dtype=np.complex_
        ),
        "eigenvalues": np.zeros((bloch_fractions.shape[0], n_states), dtype=np.float_),
        "bloch_fractions": bloch_fractions,
    }

    for idx, bloch_fraction in enumerate(bloch_fractions):
        h = hamiltonian_generator(bloch_fraction)
        eigenstates = calculate_eigenvectors_hermitian(
            h, subset_by_index=subset_by_index
        )

        out["vectors"][idx] = eigenstates["vectors"]
        out["eigenvalues"][idx] = eigenstates["eigenvalues"]

    return out


def select_eigenstate(
    collection: EigenstateColllection[_B0Cov, _L0Inv],
    bloch_idx: int,
    band_idx: int,
) -> Eigenstate[_B0Cov]:
    """
    Select an eigenstate from an eigenstate collection.

    Parameters
    ----------
    collection : EigenstateColllection[_B0Cov]
    bloch_idx : int
    band_idx : int

    Returns
    -------
    Eigenstate[_B0Cov]
    """
    return {
        "basis": collection["basis"],
        "vector": collection["vectors"][bloch_idx, band_idx],
        "eigenvalue": collection["eigenvalues"][bloch_idx, band_idx],  # type: ignore[typeddict-item]
    }
