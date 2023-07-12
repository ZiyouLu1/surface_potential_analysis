from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, Unpack

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.axis.conversion import axis_as_single_point_axis
from surface_potential_analysis.basis.brillouin_zone import (
    decrement_brillouin_zone,
    get_all_brag_point,
)
from surface_potential_analysis.basis.util import (
    AxisWithLengthBasisUtil,
    project_k_points_along_axes,
    project_x_points_along_axes,
)
from surface_potential_analysis.util.util import slice_ignoring_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis._types import IndexLike, SingleStackedIndexLike
    from surface_potential_analysis.basis.basis import AxisWithLengthBasis

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _B0Inv = TypeVar("_B0Inv", bound=AxisWithLengthBasis[Any])
    _NDInv = TypeVar("_NDInv", bound=int)


def plot_k_points_projected_2d(
    basis: _B0Inv,
    axes: tuple[int, int],
    points: np.ndarray[tuple[_NDInv, Unpack[_S0Inv]], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Given a set of k_points, plot the points perpendicular to kz_axis.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]
        direction perpendicular to which the points should be projected onto
    points : np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]
        points to project
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    projected_points = project_k_points_along_axes(points, basis, axes)

    (line,) = ax.plot(*projected_points.reshape(2, -1))
    line.set_linestyle(" ")
    line.set_marker("x")

    ax.set_xlabel(f"k{axes[0]} axis")
    ax.set_ylabel(f"k{axes[1]} axis")
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, line


def plot_brillouin_zone_points_projected_2d(
    basis: _B0Inv,
    axes: tuple[int, int],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the k coordinates in the first brillouin zone.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    util = AxisWithLengthBasisUtil(
        (basis[0], basis[1], axis_as_single_point_axis(basis[2]))
    )
    coordinates = decrement_brillouin_zone(basis, util.nk_points)
    coordinates = decrement_brillouin_zone(basis, coordinates)
    points = util.get_k_points_at_index(coordinates)

    return plot_k_points_projected_2d(basis, axes, points, ax=ax)


def plot_bragg_points_projected_2d(
    basis: _B0Inv,
    axes: tuple[int, int],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the bragg points in the first band projected perpendicular to kz_axis.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    points = get_all_brag_point(basis)
    return plot_k_points_projected_2d(basis, axes, points, ax=ax)


def plot_fundamental_k_in_plane_projected_2d(
    basis: _B0Inv,
    axes: tuple[int, int],
    idx: SingleStackedIndexLike,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the k coordinates in the basis perpendicular to kz_axis.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]
        axis perpendicular to which to plot the points
    idx : SingleStackedIndexLike
        idx in kz_axis direction
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    util = AxisWithLengthBasisUtil(basis)
    points = util.fundamental_k_points.reshape(3, *util.fundamental_shape)[
        slice_ignoring_axes(idx, axes)
    ].reshape(3, -1)
    return plot_k_points_projected_2d(basis, axes, points, ax=ax)


def plot_x_points_projected_2d(
    basis: _B0Inv,
    axes: tuple[int, int],
    points: np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Given a set of x_points, plot the points perpendicular to z_axis.

    Parameters
    ----------
    basis : _B0Inv
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        direction perpendicular to which the points should be projected onto
    points : np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]
        points to project
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    projected_points = project_x_points_along_axes(points, basis, axes)

    (line,) = ax.plot(*projected_points.reshape(2, -1))
    line.set_linestyle(" ")
    line.set_marker("x")

    ax.set_xlabel(f"x{axes[0]} axis")
    ax.set_ylabel(f"x{axes[1]} axis")
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, line


def plot_fundamental_x_in_plane_projected_2d(
    basis: _B0Inv,
    axes: tuple[int, int],
    idx: SingleStackedIndexLike,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the fundamental x coordinates in the basis perpendicular to z_axis through the idx plane.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]
        axis perpendicular to which to plot the points
    idx : SingleStackedIndexLike
        idx in kz_axis direction
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    util = AxisWithLengthBasisUtil(basis)
    points = (
        np.array(util.fundamental_x_points)
        .reshape(util.ndim, *util.fundamental_shape)[
            slice_ignoring_axes(idx, (0, *[1 + ax for ax in axes]))
        ]
        .reshape(util.ndim, -1)
    )
    return plot_x_points_projected_2d(basis, axes, points, ax=ax)


def plot_fundamental_x_at_index_projected_2d(
    basis: _B0Inv,
    idx: IndexLike,
    axes: tuple[int, int],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Given an index-like object, plot all the projected points in the given basis.

    Parameters
    ----------
    basis : _B0Inv
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
    util = AxisWithLengthBasisUtil(basis)
    idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    points = (
        np.array(util.fundamental_nx_points)
        .reshape(util.ndim, *util.fundamental_shape)[:, *idx]
        .reshape(util.ndim, -1)
    )
    return plot_x_points_projected_2d(basis, axes, points, ax=ax)
