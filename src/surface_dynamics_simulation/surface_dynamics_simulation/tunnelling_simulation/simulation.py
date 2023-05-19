from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np
import scipy

if TYPE_CHECKING:
    from surface_dynamics_simulation.tunnelling_matrix.tunnelling_matrix import (
        TunnellingMatrix,
    )
    from surface_dynamics_simulation.tunnelling_simulation.tunnelling_simulation_state import (
        TunnellingSimulationState,
        TunnellingVector,
    )

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, int, int])

_L1Inv = TypeVar("_L1Inv", bound=int)


class TunnellingEigenstates(TypedDict, Generic[_S0Inv]):
    """Represents the eigenstates of a given tunnelling matrix."""

    energies: np.ndarray[tuple[int], np.dtype[np.float_]]
    vectors: np.ndarray[tuple[int, int], np.dtype[np.float_]]
    """Eigenvectors, indexed such that the ith state has a vector vectors[:,i]"""
    shape: _S0Inv


def calculate_tunnelling_eigenstates(
    matrix: TunnellingMatrix[_S0Inv],
) -> TunnellingEigenstates[_S0Inv]:
    """
    Given a tunnelling matrix, find the eigenstates.

    Parameters
    ----------
    matrix : TunnellingMatrix[_S0Inv]

    Returns
    -------
    TunnellingEigenstates[_S0Inv]
    """
    energies, vectors = scipy.linalg.eig(matrix["array"])
    return {"energies": energies, "vectors": vectors, "shape": matrix["shape"]}


def calculate_equilibrium_state(
    matrix: TunnellingMatrix[_S0Inv],
) -> TunnellingVector[_S0Inv]:
    """
    Get the vector corresponding to the equilibrium tunnellign state.

    Since all of the eigenstates have E < 0 except for the equilibrium
    this corresponds to the single "zero energy" state

    Parameters
    ----------
    matrix : TunnellingMatrix[_S0Inv]

    Returns
    -------
    TunnellingVector[_S0Inv]
    """
    eigenstates = calculate_tunnelling_eigenstates(matrix)
    vector = eigenstates["vectors"][:, np.argmax(eigenstates["energies"])]
    return {"shape": eigenstates["shape"], "vector": vector}


def simulate_tunnelling(
    matrix: TunnellingMatrix[_S0Inv],
    initial_state: TunnellingVector[_S0Inv],
    times: np.ndarray[tuple[_L1Inv], np.dtype[np.float_]],
) -> TunnellingSimulationState[_L1Inv, _S0Inv]:
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
    eigenvalues = calculate_tunnelling_eigenstates(matrix)

    # eigenvalues["vectors"] is the matrix such that the ith vector is
    # eigenvalues["vectors"][:, i].
    # linalg.solve(a, b) = x where np.dot(a, x) == b, which is the sum
    # of the product over the last axis of x, so a[i] x[:, i] = b[:]
    # ie solved is the decomposition of b into the eigenvectors
    solved = scipy.linalg.solve(eigenvalues["vectors"], initial_state["vector"])

    constants = solved[:, np.newaxis] * np.exp(
        eigenvalues["energies"][:, np.newaxis] * times[np.newaxis, :]
    )

    vectors = np.sum(
        eigenvalues["vectors"][:, :, np.newaxis] * constants[np.newaxis], axis=1
    )
    return {"shape": initial_state["shape"], "times": times, "vectors": vectors}


def calculate_flux_from_site(matrix: TunnellingMatrix[_S0Inv]) -> None:
    eigenvalues = calculate_tunnelling_eigenstates(matrix)
    a_max = np.argmax(np.abs(eigenvalues["energies"]))
    print(eigenvalues["vectors"][:, a_max])
    print(
        np.sum(
            np.square(eigenvalues["vectors"][:, a_max].reshape(*matrix["shape"])),
            axis=(0, 1),
        )
    )

    a_min = np.argmin(np.abs(eigenvalues["energies"]))
    print(eigenvalues["vectors"][:, a_min])
    print(
        np.sum(
            np.square(eigenvalues["vectors"][:, a_min].reshape(*matrix["shape"])),
            axis=(0, 1),
        )
    )

    start: TunnellingVector[Any] = {
        "vector": np.zeros(np.prod(matrix["shape"])),
        "shape": matrix["shape"],
    }
    start["vector"][0] = 1
    solved = scipy.linalg.solve(eigenvalues["vectors"], start["vector"])
    flux = np.sum(solved * eigenvalues["energies"])
    print(flux)
    print(1 / flux)


def calculate_hopping_rate(matrix: TunnellingMatrix[_S0Inv]) -> float:
    """
    Given the tunnelling matrix, calculate the hopping rate between sites at equilibrium.

    Parameters
    ----------
    matrix : TunnellingMatrix[_S0Inv]

    Returns
    -------
    float
    """
    equilibrium_state = calculate_equilibrium_state(matrix)
    stacked_vector = equilibrium_state["vector"].reshape(matrix["shape"])
    occupations = np.sqrt(np.sum(np.square(stacked_vector), axis=(0, 1)))

    external_hopping = (
        matrix["array"].reshape(*matrix["shape"], *matrix["shape"])[0, 0].copy()
    )
    external_hopping[:, 0, 0] = 0

    summed_hopping = np.sum(external_hopping, axis=(1, 2, 3))
    return float(np.sum(summed_hopping * occupations))
