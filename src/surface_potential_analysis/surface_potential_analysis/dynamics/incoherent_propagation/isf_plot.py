from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.dynamics.incoherent_propagation.isf import (
    ISF4VariableFit,
    ISFFeyModelFit,
    RateDecomposition,
    get_isf_from_4_variable_fit,
    get_isf_from_fey_model_fit,
)
from surface_potential_analysis.state_vector.eigenvalue_list_plot import (
    plot_eigenvalue_against_x,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.state_vector.eigenvalue_list import EigenvalueList
    from surface_potential_analysis.util.plot import Scale
    from surface_potential_analysis.util.util import Measure

_N0Inv = TypeVar("_N0Inv", bound=int)
_L0Inv = TypeVar("_L0Inv", bound=int)


def plot_isf_against_time(
    eigenvalues: EigenvalueList[_N0Inv],
    times: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the isf against time.

    Parameters
    ----------
    eigenvalues : EigenvalueList[_N0Inv]
        list of eigenvalues to plot
    times : np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
        Times for which to plot the eigenvalues
    ax : Axes | None, optional
        Plot axis, by default None
    measure : Measure, optional
        plot measure, by default "abs"
    scale : Scale, optional
        plot y scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    (fig, ax, line) = plot_eigenvalue_against_x(
        eigenvalues, times, ax=ax, measure=measure, scale=scale
    )
    ax.set_ylabel("ISF")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time /s")
    ax.set_xlim(times[0], times[-1])
    return fig, ax, line


def plot_isf_4_variable_fit_against_time(
    fit: ISF4VariableFit,
    times: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the ISF fit against time.

    Parameters
    ----------
    fit : ISFFit
        The fit to the ISF
    times : np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
        times to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    isf = get_isf_from_4_variable_fit(fit, times)
    return plot_isf_against_time(isf, times, ax=ax, measure=measure, scale=scale)


def plot_isf_fey_model_fit_against_time(
    fit: ISFFeyModelFit,
    times: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the ISF fit against time.

    Parameters
    ----------
    fit : ISFFit
        The fit to the ISF
    times : np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
        times to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    isf = get_isf_from_fey_model_fit(fit, times)
    return plot_isf_against_time(isf, times, ax=ax, measure=measure, scale=scale)


def plot_rate_decomposition_against_temperature(
    rates: list[RateDecomposition[_L0Inv]],
    temperatures: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the individual rates in the simulation against temperature.

    Parameters
    ----------
    rates : list[RateDecomposition[_L0Inv]]
    temperatures : np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    rate_constants = -np.real([rate.eigenvalues for rate in rates])
    coefficients = np.abs([rate.coefficients for rate in rates])
    relevant_rates = np.argsort(coefficients, axis=-1)
    sorted_rates = np.take_along_axis(rate_constants, relevant_rates, axis=-1)[:, ::-1]

    coefficients = np.abs([rate.coefficients for rate in rates])
    for i in range(50):
        ax.plot(temperatures, sorted_rates[:, i])
    return fig, ax
