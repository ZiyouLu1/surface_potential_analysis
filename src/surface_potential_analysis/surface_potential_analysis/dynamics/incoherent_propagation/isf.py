from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import scipy.optimize

from surface_potential_analysis.axis.axis import FundamentalPositionAxis
from surface_potential_analysis.basis.util import BasisUtil, wrap_x_point_around_origin

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import DiagonalOperator
    from surface_potential_analysis.operator.operator_list import DiagonalOperatorList
    from surface_potential_analysis.state_vector.eigenvalue_list import EigenvalueList

    from .tunnelling_basis import (
        TunnellingSimulationBandsAxis,
        TunnellingSimulationBasis,
    )

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis[Any, Any, Any])
    _TSX0Inv = TypeVar("_TSX0Inv", bound=TunnellingSimulationBandsAxis[Any])


def _get_location_offsets_per_band(
    axis: TunnellingSimulationBandsAxis[_L0Inv],
) -> np.ndarray[tuple[Literal[2], _L0Inv], np.dtype[np.float_]]:
    return np.tensordot(axis.unit_cell, axis.locations, axes=(0, 0))  # type: ignore[no-any-return]


def _calculate_approximate_locations(
    basis: TunnellingSimulationBasis[Any, Any, _TSX0Inv],
) -> np.ndarray[tuple[Literal[2], Any], np.dtype[np.float_]]:
    nx_points = BasisUtil(basis).nx_points
    central_locations = np.tensordot(
        basis[2].unit_cell, (nx_points[0], nx_points[1]), axes=(0, 0)
    )
    band_offsets = _get_location_offsets_per_band(basis[2])
    offsets = band_offsets[:, nx_points[2]]
    return central_locations + offsets  # type: ignore[no-any-return]


def calculate_isf_approximate_locations(
    initial_matrix: DiagonalOperator[_B0Inv, _B0Inv],
    final_matrices: DiagonalOperatorList[_B0Inv, _B0Inv, _L0Inv],
    dk: np.ndarray[tuple[Literal[2]], np.dtype[np.float_]],
) -> EigenvalueList[_L0Inv]:
    """
    Calculate the ISF, assuming all states are approximately eigenstates of position.

    Parameters
    ----------
    initial_matrix : DiagonalOperator[_B0Inv, _B0Inv]
        Initial density matrix
    final_matrices : DiagonalOperatorList[_B0Inv, _B0Inv, _L0Inv]
        Final density matrices
    dk : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        direction along which to measure the ISF

    Returns
    -------
    EigenvalueList[_L0Inv]
        _description_
    """
    locations = _calculate_approximate_locations(initial_matrix["basis"])
    initial_location = np.average(locations, axis=1, weights=initial_matrix["vector"])
    distances = locations - initial_location[:, np.newaxis]
    distances_wrapped = wrap_x_point_around_origin(
        (
            FundamentalPositionAxis(
                initial_matrix["basis"][2].unit_cell[0]
                * initial_matrix["basis"][0].fundamental_n,
                1,
            ),
            FundamentalPositionAxis(
                initial_matrix["basis"][2].unit_cell[1]
                * initial_matrix["basis"][1].fundamental_n,
                1,
            ),
        ),
        distances,
    )

    mean_phi = np.tensordot(dk, distances_wrapped, axes=(0, 0))
    eigenvalues = np.tensordot(
        np.exp(-1j * mean_phi), final_matrices["vectors"], axes=(0, 1)
    )
    return {"eigenvalues": eigenvalues}


@dataclass
class ISFFit:
    """Result of fitting a double exponential to an ISF."""

    fast_rate: float
    fast_amplitude: float
    slow_rate: float
    slow_amplitude: float
    baseline: float


def get_isf_from_fit(
    fit: ISFFit, times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
) -> EigenvalueList[_L0Inv]:
    """
    Given an ISF Fit calculate the ISF.

    Parameters
    ----------
    fit : ISFFit
    times : np.ndarray[tuple[int], np.dtype[np.float_]]

    Returns
    -------
    EigenvalueList[_L0Inv]
    """
    return {
        "eigenvalues": fit.fast_amplitude * np.exp(-fit.fast_rate * times)
        + fit.slow_amplitude * np.exp(-fit.slow_rate * times)
        + fit.baseline
    }


def fit_isf_to_double_exponential(
    isf: EigenvalueList[_L0Inv], times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
) -> ISFFit:
    """
    Fit the ISF to a double exponential, and calculate the fast and slow rates.

    Parameters
    ----------
    isf : EigenvalueList[_L0Inv]
    times : np.ndarray[tuple[int], np.dtype[np.float_]]

    Returns
    -------
    ISFFit
    """
    popt, pcov = scipy.optimize.curve_fit(
        lambda t, a, b, c, d: (
            a * np.exp(-(b) * t) + (1 - a - d) * np.exp(-(c) * t) + d
        ),
        times,
        isf["eigenvalues"],
        p0=(0.5, 1e10, 2e10, 0),
        bounds=([0, 0, 0, 0], [1, np.inf, np.inf, 1]),
    )
    return ISFFit(popt[1], popt[0], popt[2], 1 - popt[3] - popt[0], popt[3])
