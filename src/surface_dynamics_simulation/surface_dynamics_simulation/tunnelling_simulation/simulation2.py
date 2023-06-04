from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np
import scipy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from surface_potential_analysis.basis.basis import Basis3d

    from surface_dynamics_simulation.tunnelling_matrix.tunnelling_matrix import (
        TunnellingMatrix2,
        TunnellingState2,
    )
    from surface_dynamics_simulation.tunnelling_simulation.tunnelling_simulation_state import (
        TunnellingSimulationState2,
    )

    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])

_L1Inv = TypeVar("_L1Inv", bound=int)


class TunnellingEigenstates(TypedDict, Generic[_B3d0Inv]):
    """Represents the eigenstates of a given tunnelling matrix."""

    energies: np.ndarray[tuple[int], np.dtype[np.float_]]
    vectors: np.ndarray[tuple[int, int], np.dtype[np.float_]]
    """Eigenvectors, indexed such that the ith state has a vector vectors[:,i]"""
    basis: _B3d0Inv


def calculate_tunnelling_eigenstates(
    matrix: TunnellingMatrix2[_B3d0Inv],
) -> TunnellingEigenstates[_B3d0Inv]:
    """
    Given a tunnelling matrix, find the eigenstates.

    Parameters
    ----------
    matrix : TunnellingMatrix[_B3d0Inv]

    Returns
    -------
    TunnellingEigenstates[_B3d0Inv]
    """
    energies, vectors = scipy.linalg.eig(matrix["array"])
    return {"basis": matrix["basis"], "energies": energies, "vectors": vectors}


def get_equilibrium_state(
    eigenstates: TunnellingEigenstates[_B3d0Inv],
) -> TunnellingState2[_B3d0Inv]:
    """
    Select the equilibrium tunnelling state from a list of eigenstates.

    Since all of the eigenstates have E < 0 except for the equilibrium
    this corresponds to the single "zero energy" state

    Parameters
    ----------
    eigenstates : TunnellingEigenstates[_B3d0Inv]

    Returns
    -------
    TunnellingVector[_B3d0Inv]
    """
    vector = eigenstates["vectors"][:, np.argmax(eigenstates["energies"])]
    return {"basis": eigenstates["basis"], "vector": vector}


def calculate_equilibrium_state(
    matrix: TunnellingMatrix2[_B3d0Inv],
) -> TunnellingState2[_B3d0Inv]:
    """
    Calculate the equilibrium tunnelling state for a given matrix.

    Since all of the eigenstates have E < 0 except for the equilibrium
    this corresponds to the single "zero energy" state

    Parameters
    ----------
    matrix : TunnellingMatrix[_B3d0Inv]

    Returns
    -------
    TunnellingVector[_B3d0Inv]
    """
    eigenstates = calculate_tunnelling_eigenstates(matrix)
    return get_equilibrium_state(eigenstates)


def get_vector_eigenstate_decomposition(
    state: TunnellingState2[_B3d0Inv], eigenstates: TunnellingEigenstates[_B3d0Inv]
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    """
    Given a state and a set of TunnellingEigenstates decompose the state into the eigenstates.

    Parameters
    ----------
    state : TunnellingVector[_B3d0Inv]
        state to decompose
    eigenstates : TunnellingEigenstates[_B3d0Inv]
        set of eigenstates to decompose into

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
        A list of coefficients for each vector such that a[i] eigenstates["vectors"][:, i] = vector[:]
    """
    # eigenstates["vectors"] is the matrix such that the ith vector is
    # eigenstates["vectors"][:, i].
    # linalg.solve(a, b) = x where np.dot(a, x) == b, which is the sum
    # of the product over the last axis of x, so a[i] x[:, i] = b[:]
    # ie solved is the decomposition of b into the eigenvectors
    return scipy.linalg.solve(eigenstates["vectors"], state["vector"])  # type: ignore[no-any-return]


def get_tunnelling_simulation_state(
    eigenstates: TunnellingEigenstates[_B3d0Inv],
    initial_state: TunnellingState2[_B3d0Inv],
    times: np.ndarray[tuple[_L1Inv], np.dtype[np.float_]],
) -> TunnellingSimulationState2[_L1Inv, _B3d0Inv]:
    """
    Get the TunnellingSimulationState given TunnellingEigenstates and initial TunnellingVector.

    Parameters
    ----------
    eigenstates : TunnellingEigenstates[_B3d0Inv]
        The eigenstates of the system
    initial_state : TunnellingVector[_B3d0Inv]
        The initial tunnelling state
    times : np.ndarray[tuple[_L1Inv], np.dtype[np.float_]]
        Times to calculate the occupation

    Returns
    -------
    TunnellingSimulationState[_L1Inv, _B3d0Inv]
    """
    coefficients = get_vector_eigenstate_decomposition(initial_state, eigenstates)

    constants = coefficients[:, np.newaxis] * np.exp(
        eigenstates["energies"][:, np.newaxis] * times[np.newaxis, :]
    )

    vectors = np.sum(
        eigenstates["vectors"][:, :, np.newaxis] * constants[np.newaxis], axis=1
    )
    return {"basis": initial_state["basis"], "times": times, "vectors": vectors}


def simulate_tunnelling_from_matrix(
    matrix: TunnellingMatrix2[_B3d0Inv],
    initial_state: TunnellingState2[_B3d0Inv],
    times: np.ndarray[tuple[_L1Inv], np.dtype[np.float_]],
) -> TunnellingSimulationState2[_L1Inv, _B3d0Inv]:
    """
    Get the TunnellingSimulationState given a tunnelling matrix and initial TunnellingVector.

    Parameters
    ----------
    matrix : TunnellingMatrix[_S0Inv]
        The matrix of tunnelling coefficients
    initial_state : TunnellingVector[_S0Inv]
        The initial tunnelling state
    times : np.ndarray[tuple[_L1Inv], np.dtype[np.float_]]
        Times to calculate the occupation

    Returns
    -------
    TunnellingSimulationState[_L1Inv, _S0Inv]
    """
    eigenstates = calculate_tunnelling_eigenstates(matrix)
    return get_tunnelling_simulation_state(eigenstates, initial_state, times)


def calculate_hopping_rate(
    matrix: TunnellingMatrix2[_B3d0Inv], internal_sites: Sequence[int]
) -> float:
    """
    Given the tunnelling matrix, calculate the hopping rate between sites at equilibrium.

    Parameters
    ----------
    matrix : TunnellingMatrix[_B3d0Inv]

    Returns
    -------
    float
    """
    equilibrium_state = calculate_equilibrium_state(matrix)
    stacked_vector = equilibrium_state["vector"].reshape(matrix["shape"])
    occupations = np.sqrt(np.sum(np.square(stacked_vector), axis=(0, 1)))

    external_hopping = matrix["array"][internal_sites].copy()
    external_hopping[:, internal_sites] = 0

    summed_hopping = np.sum(external_hopping, axis=(1, 2, 3))
    return float(np.sum(summed_hopping * occupations))
