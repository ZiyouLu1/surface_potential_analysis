from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.axis.util import BasisUtil
from surface_potential_analysis.probability_vector.conversion import (
    convert_probability_vector_to_momentum_basis,
    convert_probability_vector_to_position_basis,
)
from surface_potential_analysis.probability_vector.probability_vector import (
    ProbabilityVector,
    sum_probabilities,
)
from surface_potential_analysis.stacked_basis.util import get_max_idx
from surface_potential_analysis.util.util import get_data_in_axes, get_measured_data

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.axis.axis_like import BasisLike
    from surface_potential_analysis.axis.stacked_axis import StackedBasisLike
    from surface_potential_analysis.axis.time_axis_like import BasisWithTimeLike
    from surface_potential_analysis.probability_vector.probability_vector import (
        ProbabilityVectorList,
    )
    from surface_potential_analysis.types import SingleStackedIndexLike
    from surface_potential_analysis.util.plot import Scale
    from surface_potential_analysis.util.util import Measure

if TYPE_CHECKING:
    _B0 = TypeVar("_B0", bound=BasisWithTimeLike[int, int])
    _B1 = TypeVar("_B1", bound=BasisLike[int, int])
    _SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])


def plot_probability_against_time(
    probability: ProbabilityVectorList[_B0, _B1],
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
    times = probability["basis"][0].times

    lines: list[Line2D] = []
    for n in range(probability["data"].shape[1]):
        (line,) = ax.plot(times, (probability["data"][:, n]))
        lines.append(line)

    ax.set_title("Plot of probability against time")
    ax.set_xlabel("time /s")
    ax.set_ylabel("occupation probability")
    ax.set_yscale(scale)
    return fig, ax, lines


def plot_total_probability_against_time(
    probability: ProbabilityVectorList[_B0, _SB0],
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
    summed = sum_probabilities(probability)
    return plot_probability_against_time(summed, ax=ax, scale=scale)


# ruff: noqa: PLR0913


def plot_probability_1d_k(
    state: ProbabilityVector[StackedBasisLike[*tuple[Any, ...]]],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an state in 1d along the given axis.

    Parameters
    ----------
    state : StateVector[_B0Inv]
    idx : SingleStackedIndexLike, optional
        index in the perpendicular directions, by default (0,0)
    axis : int, optional
        axis along which to plot, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    converted = convert_probability_vector_to_momentum_basis(state)
    idx = get_max_idx(converted, axes) if idx is None else idx
    data_slice: list[slice | int | np.integer[Any]] = list(idx)
    data_slice.insert(axes[0], slice(None))

    util = BasisUtil(converted["basis"])
    coordinates = util.fundamental_stacked_nk_points[0]
    points = get_data_in_axes(converted["data"].reshape(util.shape), axes, idx)
    data = get_measured_data(points, measure)

    (line,) = ax.plot(np.fft.fftshift(coordinates), np.fft.fftshift(data))
    ax.set_xlabel(f"k{axes[0]} axis")
    ax.set_ylabel("Probability")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_probability_1d_x(
    state: ProbabilityVector[StackedBasisLike[*tuple[Any, ...]]],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an state in 1d along the given axis.

    Parameters
    ----------
    state : StateVector[_B0Inv]
    idx : SingleStackedIndexLike, optional
        index in the perpendicular directions, by default (0,0)
    axis : int, optional
        axis along which to plot, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    converted = convert_probability_vector_to_position_basis(state)
    idx = get_max_idx(converted, axes) if idx is None else idx

    util = BasisUtil(converted["basis"])
    fundamental_x_points = util.fundamental_x_points_stacked
    coordinates = np.linalg.norm(fundamental_x_points, axis=0)
    points = get_data_in_axes(converted["data"].reshape(util.shape), axes, idx)
    data = get_measured_data(points, measure)

    (line,) = ax.plot(coordinates, data)
    ax.set_xlabel(f"x{(axes[0] % 3)} axis")
    ax.set_ylabel("Probability")
    ax.set_yscale(scale)
    return fig, ax, line
