from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.basis_config.util import (
    BasisConfigUtil,
    get_fundamental_projected_x_points,
)
from surface_potential_analysis.util.util import slice_along_axis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis._types import IndexLike, SingleFlatIndexLike
    from surface_potential_analysis.basis import BasisLike
    from surface_potential_analysis.basis_config.basis_config import (
        BasisConfig,
    )

    _BX0Inv = TypeVar("_BX0Inv", bound=BasisLike[Any, Any])
    _BX1Inv = TypeVar("_BX1Inv", bound=BasisLike[Any, Any])
    _BX2Inv = TypeVar("_BX2Inv", bound=BasisLike[Any, Any])


def plot_projected_x_points_2d(
    basis: BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv],
    idx: SingleFlatIndexLike,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the coordinates in the basis perpendicular to z_axis.

    Parameters
    ----------
    basis : BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]
    idx : int
        idx in z_axis direction
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to which to plot he points
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    coordinates = get_fundamental_projected_x_points(basis, z_axis)[
        slice_along_axis(idx, (z_axis % 3) + 1)
    ].reshape(2, -1)

    (line,) = ax.plot(*coordinates)
    line.set_linestyle(" ")
    line.set_marker("x")

    ax.set_xlabel(f"x{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")  # noqa: PLR2004
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, line


def plot_fundamental_projected_x_at_points(
    basis: BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv],
    points: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    z_axis: Literal[0, 1, 2, -1, -2, -3] = 2,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the coordinates in the basis perpendicular to z_axis.

    Parameters
    ----------
    basis : BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]
    points : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        coordinates to plot
    z_axis : Literal[0, 1, 2, -1, -2, -3], optional
        axis perpendicular to which to plot the points, by default 2
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    coordinates = get_fundamental_projected_x_points(basis, z_axis)[:, *points]

    (line,) = ax.plot(*coordinates)
    line.set_linestyle(" ")
    line.set_marker("x")

    ax.set_xlabel(f"x{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")  # noqa: PLR2004
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, line


def plot_fundamental_projected_x_at_index(
    basis: BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv],
    idx: IndexLike,
    z_axis: Literal[0, 1, 2, -1, -2, -3] = 2,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Given an index-like object, plot all the projected points in the given basis.

    Parameters
    ----------
    basis : BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]
    idx : IndexLike
        index to plot
    z_axis : Literal[0, 1, 2, -1, -2, -3], optional
        axis perpendicular to which to plot the points, by default 2
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    util = BasisConfigUtil(basis)
    idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    points = np.array(idx).reshape(3, -1)
    return plot_fundamental_projected_x_at_points(basis, points, z_axis, ax=ax)
