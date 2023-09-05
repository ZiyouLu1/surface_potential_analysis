from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalAxis
from surface_potential_analysis.axis.time_axis_like import (
    ExplicitTimeAxis,
    FundamentalTimeAxis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.incoherent_propagation.eigenstates import (
    calculate_tunnelling_eigenstates,
    calculate_tunnelling_simulation_state,
    get_equilibrium_state,
    get_operator_state_vector_decomposition,
    get_tunnelling_simulation_state,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    density_matrix_as_probability,
    density_matrix_list_as_probabilities,
    get_initial_pure_density_matrix_for_basis,
)
from surface_potential_analysis.dynamics.isf import calculate_isf_approximate_locations
from surface_potential_analysis.probability_vector.probability_vector import (
    ProbabilityVector,
    ProbabilityVectorList,
    average_probabilities,
    sum_probability_over_axis,
)
from surface_potential_analysis.state_vector.eigenvalue_list import average_eigenvalues

if TYPE_CHECKING:
    from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
        TunnellingMMatrix,
    )
    from surface_potential_analysis.dynamics.tunnelling_basis import (
        TunnellingSimulationBasis,
    )
    from surface_potential_analysis.operator.operator import DiagonalOperator
    from surface_potential_analysis.state_vector.eigenvalue_list import EigenvalueList

    _B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis[Any, Any, Any])


_L0Inv = TypeVar("_L0Inv", bound=int)


def calculate_isf_at_times(
    matrix: TunnellingMMatrix[_B0Inv],
    initial: DiagonalOperator[_B0Inv, _B0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
    dk: np.ndarray[tuple[Literal[2]], np.dtype[np.float_]],
) -> EigenvalueList[tuple[ExplicitTimeAxis[_L0Inv]]]:
    """
    Calculate the ISF, assuming all states are approximately eigenstates of position.

    Parameters
    ----------
    initial_matrix : DiagonalOperator[_B0Inv, _B0Inv]
        Initial density matrix
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    dk : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        direction along which to measure the ISF

    Returns
    -------
    EigenvalueList[_L0Inv]
    """
    final = calculate_tunnelling_simulation_state(matrix, initial, times)
    initial_occupation = density_matrix_as_probability(initial)
    final_occupation = density_matrix_list_as_probabilities(final)
    return calculate_isf_approximate_locations(initial_occupation, final_occupation, dk)


def calculate_equilibrium_state_averaged_isf(
    matrix: TunnellingMMatrix[_B0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
    dk: np.ndarray[tuple[Literal[2]], np.dtype[np.float_]],
) -> EigenvalueList[tuple[FundamentalTimeAxis[_L0Inv]]]:
    """
    Calculate the ISF, averaging over the equilibrium occupation of each band.

    Parameters
    ----------
    matrix : TunnellingMMatrix[_B0Inv]
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    dk : np.ndarray[tuple[Literal[2]], np.dtype[np.float_]]

    Returns
    -------
    EigenvalueList[_L0Inv]
    """
    util = BasisUtil(matrix["basis"])
    eigenstates = calculate_tunnelling_eigenstates(matrix)
    equilibrium = get_equilibrium_state(eigenstates)

    occupation_probabilities = sum_probability_over_axis(
        density_matrix_as_probability(equilibrium), (0, 1)
    )
    eigenvalues = np.zeros((util.shape[2], times.size))
    for band in range(util.shape[2]):
        initial = get_initial_pure_density_matrix_for_basis(
            matrix["basis"], (0, 0, band)
        )
        initial_probability = density_matrix_as_probability(initial)
        final = get_tunnelling_simulation_state(eigenstates, initial, times)
        final_probabilities = density_matrix_list_as_probabilities(final)
        isf = calculate_isf_approximate_locations(
            initial_probability, final_probabilities, dk
        )
        eigenvalues[band] = isf
    isf_per_band: EigenvalueList[
        tuple[FundamentalAxis[int], ExplicitTimeAxis[_L0Inv]]
    ] = {
        "list_basis": (FundamentalAxis(util.shape[2]), ExplicitTimeAxis(times)),
        "eigenvalues": eigenvalues.reshape(-1),
    }
    return average_eigenvalues(
        isf_per_band, (0,), weights=occupation_probabilities["vector"]
    )


def calculate_equilibrium_initial_state_isf(
    matrix: TunnellingMMatrix[_B0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
    dk: np.ndarray[tuple[Literal[2]], np.dtype[np.float_]],
) -> EigenvalueList[tuple[FundamentalTimeAxis[_L0Inv]]]:
    """
    Calculate the ISF, averaging over the equilibrium occupation of each band.

    Parameters
    ----------
    matrix : TunnellingMMatrix[_B0Inv]
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    dk : np.ndarray[tuple[Literal[2]], np.dtype[np.float_]]

    Returns
    -------
    EigenvalueList[_L0Inv]
    """
    util = BasisUtil(matrix["basis"])
    eigenstates = calculate_tunnelling_eigenstates(matrix)

    vectors = np.zeros((util.shape[2], times.size, util.size))
    for band in range(util.shape[2]):
        initial_state = get_initial_pure_density_matrix_for_basis(
            matrix["basis"], (0, 0, band)
        )
        final_state = get_tunnelling_simulation_state(eigenstates, initial_state, times)
        final_probabilities = density_matrix_list_as_probabilities(final_state)
        vectors[band] = final_probabilities["vectors"]
    probability_per_band: ProbabilityVectorList[
        tuple[FundamentalAxis[int], ExplicitTimeAxis[_L0Inv]], _B0Inv
    ] = {
        "basis": matrix["basis"],
        "list_basis": (FundamentalAxis(util.shape[2]), ExplicitTimeAxis(times)),
        "vectors": vectors.reshape(-1, util.size),
    }

    equilibrium = get_equilibrium_state(eigenstates)
    occupation_probabilities = sum_probability_over_axis(
        density_matrix_as_probability(equilibrium), (0, 1)
    )
    vector = np.zeros(util.shape)
    vector[0, 0, :] = occupation_probabilities["vector"]
    initial: ProbabilityVector[_B0Inv] = {
        "basis": matrix["basis"],
        "vector": vector.reshape(-1),
    }
    average_probability = average_probabilities(
        probability_per_band, weights=occupation_probabilities["vector"]
    )
    return calculate_isf_approximate_locations(initial, average_probability, dk)


@dataclass
class RateDecomposition(Generic[_L0Inv]):
    """Result of fitting a double exponential to an ISF."""

    eigenvalues: np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    coefficients: np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]


def get_rate_decomposition(
    matrix: TunnellingMMatrix[_B0Inv], initial: DiagonalOperator[_B0Inv, _B0Inv]
) -> RateDecomposition[int]:
    """
    Get the eigenvalues and relevant contribution of the rates in the simulation.

    Parameters
    ----------
    matrix : TunnellingMMatrix[_B0Inv]
    initial : DiagonalOperator[_B0Inv, _B0Inv]

    Returns
    -------
    RateDecomposition[int]
    """
    eigenstates = calculate_tunnelling_eigenstates(matrix)
    coefficients = get_operator_state_vector_decomposition(initial, eigenstates)
    return RateDecomposition(eigenstates["eigenvalues"], coefficients)
