from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from surface_potential_analysis.state_vector.eigenvalue_list_plot import (
    plot_eigenvalue_against_x,
)

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.state_vector.eigenvalue_list import EigenvalueList
    from surface_potential_analysis.util.plot import Scale
    from surface_potential_analysis.util.util import Measure

_N0Inv = TypeVar("_N0Inv", bound=int)


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
