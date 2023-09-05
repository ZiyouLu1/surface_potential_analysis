from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.dynamics.isf import (
    ISF4VariableFit,
    ISFFeyModelFit,
    get_isf_from_4_variable_fit,
    get_isf_from_fey_model_fit_110,
    get_isf_from_fey_model_fit_112bar,
)
from surface_potential_analysis.state_vector.eigenvalue_list import average_eigenvalues
from surface_potential_analysis.state_vector.eigenvalue_list_plot import (
    plot_eigenvalue_against_time,
)

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.axis.axis_like import AxisLike
    from surface_potential_analysis.axis.time_axis_like import AxisWithTimeLike
    from surface_potential_analysis.state_vector.eigenvalue_list import EigenvalueList
    from surface_potential_analysis.util.plot import Scale
    from surface_potential_analysis.util.util import Measure

    _B0Inv = TypeVar("_B0Inv", bound=tuple[AxisWithTimeLike[Any, Any]])
    _N0Inv = TypeVar("_N0Inv", bound=int)
    _B0StackedInv = TypeVar(
        "_B0StackedInv", bound=tuple[AxisLike[Any, Any], AxisWithTimeLike[Any, Any]]
    )


def plot_isf_against_time(
    eigenvalues: EigenvalueList[_B0Inv],
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
    (fig, ax, line) = plot_eigenvalue_against_time(
        eigenvalues, ax=ax, measure=measure, scale=scale
    )
    ax.set_ylabel("ISF")
    ax.set_ylim(0, 1)
    return fig, ax, line


def plot_average_isf_against_time(
    eigenvalues: EigenvalueList[_B0StackedInv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the average ISF against time.

    Parameters
    ----------
    eigenvalues : EigenvalueList[_B0StackedInv]
        _description_
    ax : Axes | None, optional
        _description_, by default None
    measure : Measure, optional
        _description_, by default "abs"
    scale : Scale, optional
        _description_, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
        _description_
    """
    averaged = average_eigenvalues(eigenvalues, (0,))
    return plot_isf_against_time(averaged, ax=ax, measure=measure, scale=scale)


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
    return plot_isf_against_time(isf, ax=ax, measure=measure, scale=scale)


def plot_isf_fey_model_fit_112bar_against_time(
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
    isf = get_isf_from_fey_model_fit_112bar(fit, times)
    return plot_isf_against_time(isf, ax=ax, measure=measure, scale=scale)


def plot_isf_fey_model_fit_110_against_time(
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
    isf = get_isf_from_fey_model_fit_110(fit, times)
    return plot_isf_against_time(isf, ax=ax, measure=measure, scale=scale)