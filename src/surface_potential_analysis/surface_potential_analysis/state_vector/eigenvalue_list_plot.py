from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import matplotlib.pyplot as plt

from surface_potential_analysis.util.util import Measure, get_measured_data

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.axis.time_axis_like import BasisWithTimeLike
    from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
    from surface_potential_analysis.util.plot import Scale

    _B0_co = TypeVar(
        "_B0_co",
        bound=BasisWithTimeLike[Any, Any],
        covariant=True,
    )


def plot_eigenvalue_against_time(
    eigenvalues: SingleBasisDiagonalOperator[_B0_co],
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

    data = get_measured_data(eigenvalues["data"], measure)
    times = eigenvalues["basis"][0].times
    (line,) = ax.plot(times, data)
    ax.set_ylabel("Eigenvalue")
    ax.set_yscale(scale)
    ax.set_xlabel("time /s")
    ax.set_xlim(times[0], times[-1])
    return fig, ax, line
