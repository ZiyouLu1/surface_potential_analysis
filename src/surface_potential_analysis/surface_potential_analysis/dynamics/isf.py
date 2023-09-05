from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import scipy.optimize

from surface_potential_analysis.axis.axis import FundamentalPositionAxis
from surface_potential_analysis.axis.time_axis_like import (
    AxisWithTimeLike,
    ExplicitTimeAxis,
)
from surface_potential_analysis.basis.util import BasisUtil, wrap_x_point_around_origin
from surface_potential_analysis.util.util import Measure, get_measured_data

if TYPE_CHECKING:
    from surface_potential_analysis.dynamics.tunnelling_basis import (
        TunnellingSimulationBandsAxis,
        TunnellingSimulationBasis,
    )
    from surface_potential_analysis.probability_vector.probability_vector import (
        ProbabilityVector,
        ProbabilityVectorList,
    )
    from surface_potential_analysis.state_vector.eigenvalue_list import EigenvalueList

    _B1Inv = TypeVar("_B1Inv", bound=TunnellingSimulationBasis[Any, Any, Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

    _B0Inv = TypeVar("_B0Inv", bound=tuple[AxisWithTimeLike[Any, Any]])
    _L0Inv = TypeVar("_L0Inv", bound=int)


def _get_location_offsets_per_band(
    axis: TunnellingSimulationBandsAxis[_L0Inv],
) -> np.ndarray[tuple[Literal[2], _L0Inv], np.dtype[np.float_]]:
    return np.tensordot(axis.unit_cell, axis.locations, axes=(0, 0))  # type: ignore[no-any-return]


def _calculate_approximate_locations(
    basis: _B1Inv,
) -> np.ndarray[tuple[Literal[2], Any], np.dtype[np.float_]]:
    nx_points = BasisUtil(basis).nx_points
    central_locations = np.tensordot(
        basis[2].unit_cell, (nx_points[0], nx_points[1]), axes=(0, 0)
    )
    band_offsets = _get_location_offsets_per_band(basis[2])
    offsets = band_offsets[:, nx_points[2]]
    return central_locations + offsets  # type: ignore[no-any-return]


def calculate_isf_approximate_locations(
    initial_occupation: ProbabilityVector[_B1Inv],
    final_occupation: ProbabilityVectorList[_B0Inv, _B1Inv],
    dk: np.ndarray[tuple[Literal[2]], np.dtype[np.float_]],
) -> EigenvalueList[_B0Inv]:
    """
    Calculate the ISF, assuming all states are approximately eigenstates of position.

    Parameters
    ----------
    initial_matrix : ProbabilityVector[_B0Inv]
        Initial occupation
    final_matrices : ProbabilityVectorList[_B0Inv, _L0Inv]
        Final occupation
    dk : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        direction along which to measure the ISF

    Returns
    -------
    EigenvalueList[_L0Inv]
    """
    locations = _calculate_approximate_locations(initial_occupation["basis"])
    initial_location = np.average(
        locations, axis=1, weights=initial_occupation["vector"]
    )
    distances = locations - initial_location[:, np.newaxis]
    distances_wrapped = wrap_x_point_around_origin(
        (
            FundamentalPositionAxis(
                initial_occupation["basis"][2].unit_cell[0]
                * initial_occupation["basis"][0].fundamental_n,
                1,
            ),
            FundamentalPositionAxis(
                initial_occupation["basis"][2].unit_cell[1]
                * initial_occupation["basis"][1].fundamental_n,
                1,
            ),
        ),
        distances,
    )

    mean_phi = np.tensordot(dk, distances_wrapped, axes=(0, 0))
    eigenvalues = np.tensordot(
        np.exp(1j * mean_phi), final_occupation["vectors"], axes=(0, 1)
    )
    return {"eigenvalues": eigenvalues, "list_basis": final_occupation["list_basis"]}


@dataclass
class ISF4VariableFit:
    """Result of fitting a double exponential to an ISF."""

    fast_rate: float
    fast_amplitude: float
    slow_rate: float
    slow_amplitude: float
    baseline: float


def get_isf_from_4_variable_fit(
    fit: ISF4VariableFit, times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
) -> EigenvalueList[tuple[ExplicitTimeAxis[_L0Inv]]]:
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
        "list_basis": (ExplicitTimeAxis(times),),
        "eigenvalues": fit.fast_amplitude * np.exp(-fit.fast_rate * times)
        + fit.slow_amplitude * np.exp(-fit.slow_rate * times)
        + fit.baseline,
    }


def fit_isf_to_double_exponential(
    isf: EigenvalueList[_B0Inv],
    *,
    measure: Measure = "abs",
) -> ISF4VariableFit:
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
    data = get_measured_data(isf["eigenvalues"], measure)
    params, _ = scipy.optimize.curve_fit(
        lambda t, a, b, c, d: (
            a * np.exp(-(b) * t) + (1 - a - d) * np.exp(-(c) * t) + d
        ),
        isf["list_basis"][0].times,
        data,
        p0=(0.5, 2e10, 1e10, 0),
        bounds=([0, 0, 0, 0], [1, np.inf, np.inf, 1]),
    )
    return ISF4VariableFit(
        params[1], params[0], params[2], 1 - params[3] - params[0], params[3]
    )


@dataclass
class ISFFeyModelFit:
    """Result of fitting a double exponential to an ISF."""

    fast_rate: float
    slow_rate: float
    a_dk: float = 2


def calculate_isf_fey_model_110(
    t: np.ndarray[_S0Inv, np.dtype[np.float_]],
    fast_rate: float,
    slow_rate: float,
    *,
    a_dk: float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    """
    Use the fey model calculate the ISF as measured in the 112bar direction given dk = 2/a.

    Parameters
    ----------
    t : np.ndarray[_S0Inv, np.dtype[np.float_]]
    fast_rate : float
    slow_rate : float

    Returns
    -------
    np.ndarray[_S0Inv, np.dtype[np.float_]]
    """
    lam = slow_rate / fast_rate
    z = np.sqrt(
        9 * lam**2
        + 16 * lam * np.cos(a_dk / 2) ** 2
        + 16 * lam * np.cos(a_dk / 2)
        - 14 * lam
        + 9
    )
    top_factor = 4 * np.cos(a_dk / 2) + 2
    n_0 = 1 + lam * np.square(top_factor / (3 * lam - 3 + z))
    n_1 = 1 + lam * np.square(top_factor / (3 * lam - 3 - z))
    norm_0 = np.square(np.abs(1 - lam * top_factor / (3 * lam - 3 + z)))
    norm_1 = np.square(np.abs(1 - lam * top_factor / (3 * lam - 3 - z)))
    c_0 = 1 / (1 + lam)
    return c_0 * (  # type: ignore[no-any-return]
        ((norm_0 / n_0) * np.exp(-slow_rate * (3 * lam + 3 + z) * t / (6 * lam)))
        + ((norm_1 / n_1) * np.exp(-slow_rate * (3 * lam + 3 - z) * t / (6 * lam)))
    )


def calculate_isf_fey_model_112bar(
    t: np.ndarray[_S0Inv, np.dtype[np.float_]],
    fast_rate: float,
    slow_rate: float,
    *,
    a_dk: float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    """
    Use the fey model calculate the ISF as measured in the 112bar direction given dk = 2/a.

    Parameters
    ----------
    t : np.ndarray[_S0Inv, np.dtype[np.float_]]
    fast_rate : float
    slow_rate : float

    Returns
    -------
    np.ndarray[_S0Inv, np.dtype[np.float_]]
    """
    lam = slow_rate / fast_rate
    y = np.sqrt(lam**2 + 2 * lam * (8 * np.cos(a_dk * np.sqrt(3) / 2) + 1) / 9 + 1)
    top_factor = np.exp(1j * a_dk / np.sqrt(3)) + 2 * np.exp(-1j * a_dk / (np.sqrt(12)))
    m_0 = 1 + 4 * lam * np.square(np.abs(top_factor / (3 * lam - 3 + 3 * y)))
    m_1 = 1 + 4 * lam * np.square(np.abs(top_factor / (3 * lam - 3 - 3 * y)))
    norm_0 = np.square(np.abs(1 - 2 * lam * top_factor / (3 * lam - 3 + 3 * y)))
    norm_1 = np.square(np.abs(1 - 2 * lam * top_factor / (3 * lam - 3 - 3 * y)))
    c_0 = 1 / (1 + lam)
    return c_0 * (  # type: ignore[no-any-return]
        ((norm_0 / m_0) * np.exp(-slow_rate * (lam + 1 + y) * t / (2 * lam)))
        + ((norm_1 / m_1) * np.exp(-slow_rate * (lam + 1 - y) * t / (2 * lam)))
    )


def get_isf_from_fey_model_fit_110(
    fit: ISFFeyModelFit, times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
) -> EigenvalueList[tuple[ExplicitTimeAxis[_L0Inv]]]:
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
        "list_basis": (ExplicitTimeAxis(times),),
        "eigenvalues": calculate_isf_fey_model_110(
            times, fit.fast_rate, fit.slow_rate, a_dk=fit.a_dk
        ).astype(np.complex_),
    }


def get_isf_from_fey_model_fit_112bar(
    fit: ISFFeyModelFit, times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
) -> EigenvalueList[tuple[ExplicitTimeAxis[_L0Inv]]]:
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
        "list_basis": (ExplicitTimeAxis(times),),
        "eigenvalues": calculate_isf_fey_model_112bar(
            times, fit.fast_rate, fit.slow_rate, a_dk=fit.a_dk
        ).astype(np.complex_),
    }


def fit_isf_to_fey_model_110(
    isf: EigenvalueList[_B0Inv],
    *,
    measure: Measure = "abs",
    a_dk: float = 2,
) -> ISFFeyModelFit:
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
    data = get_measured_data(isf["eigenvalues"], measure)
    params, _ = scipy.optimize.curve_fit(
        lambda t, f, s: calculate_isf_fey_model_110(t, f, s, a_dk=a_dk),
        isf["list_basis"][0].times,
        data,
        p0=(1.4e9, 3e8),
        bounds=([0, 0], [np.inf, np.inf]),
    )
    return ISFFeyModelFit(params[0], params[1], a_dk=a_dk)


def fit_isf_to_fey_model_112bar(
    isf: EigenvalueList[_B0Inv],
    *,
    measure: Measure = "abs",
    a_dk: float = 2,
) -> ISFFeyModelFit:
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
    data = get_measured_data(isf["eigenvalues"], measure)
    params, _ = scipy.optimize.curve_fit(
        lambda t, f, s: calculate_isf_fey_model_112bar(t, f, s, a_dk=a_dk),
        isf["list_basis"][0].times,
        data,
        p0=(2e10, 1e10),
        bounds=([0, 0], [np.inf, np.inf]),
    )
    return ISFFeyModelFit(params[0], params[1], a_dk=a_dk)
