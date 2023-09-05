from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from matplotlib import pyplot as plt

from surface_potential_analysis.probability_vector.probability_vector import (
    sum_probabilities_over_axis,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.axis.time_axis_like import AxisWithTimeLike
    from surface_potential_analysis.basis.basis import Basis
    from surface_potential_analysis.probability_vector.probability_vector import (
        ProbabilityVectorList,
    )
    from surface_potential_analysis.util.plot import Scale

if TYPE_CHECKING:
    _B0Inv = TypeVar("_B0Inv", bound=tuple[AxisWithTimeLike[int, int]])
    _B1Inv = TypeVar("_B1Inv", bound=Basis)


def plot_probability_against_time(
    probability: ProbabilityVectorList[_B0Inv, _B1Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot probability against time.

    Parameters
    ----------
    probability : ProbabilityVectorList[_B0Inv, _L0Inv]
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, list[Line2D]]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    times = probability["list_basis"][0].times

    lines: list[Line2D] = []
    for n in range(probability["vectors"].shape[1]):
        (line,) = ax.plot(times, (probability["vectors"][:, n]))
        lines.append(line)

    ax.set_title("Plot of probability against time")
    ax.set_xlabel("time /s")
    ax.set_ylabel("occupation probability")
    ax.set_yscale(scale)
    return fig, ax, lines


def plot_total_probability_against_time(
    probability: ProbabilityVectorList[_B0Inv, _B1Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot total probability against time.

    Parameters
    ----------
    probability : ProbabilityVectorList[_B0Inv, _L0Inv]
    times : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, list[Line2D]]
    """
    summed = sum_probabilities_over_axis(probability)
    return plot_probability_against_time(summed, ax=ax, scale=scale)
