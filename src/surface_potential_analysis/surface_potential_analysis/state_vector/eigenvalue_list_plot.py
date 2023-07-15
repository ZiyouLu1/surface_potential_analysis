from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import matplotlib.pyplot as plt

from surface_potential_analysis.util.util import Measure, get_measured_data

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.util.plot import Scale

    from .eigenvalue_list import EigenvalueList

_N0Inv = TypeVar("_N0Inv", bound=int)


def plot_eigenvalue_against_x(
    eigenvalues: EigenvalueList[_N0Inv],
    x_values: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the eigenvalues against time.

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
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    data = get_measured_data(eigenvalues["eigenvalues"], measure)
    (line,) = ax.plot(x_values, data)
    ax.set_ylabel("Eigenvalue")
    ax.set_yscale(scale)
    return fig, ax, line
