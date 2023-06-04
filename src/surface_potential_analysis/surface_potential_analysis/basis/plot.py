from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, Unpack

from matplotlib import pyplot as plt

from surface_potential_analysis.axis.conversion import axis_as_single_point_axis
from surface_potential_analysis.basis.brillouin_zone import (
    decrement_brillouin_zone_3d,
    get_all_brag_point,
)
from surface_potential_analysis.basis.util import (
    Basis3dUtil,
    project_k_points_along_axis,
    project_x_points_along_axis,
)
from surface_potential_analysis.util.util import slice_along_axis

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis._types import IndexLike3d, SingleFlatIndexLike
    from surface_potential_analysis.axis import AxisLike3d
    from surface_potential_analysis.basis.basis import (
        Basis3d,
    )

    _A3d0Inv = TypeVar("_A3d0Inv", bound=AxisLike3d[Any, Any])
    _A3d1Inv = TypeVar("_A3d1Inv", bound=AxisLike3d[Any, Any])
    _A3d2Inv = TypeVar("_A3d2Inv", bound=AxisLike3d[Any, Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


def plot_k_points_projected_2d(
    basis: Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv],
    kz_axis: Literal[0, 1, 2, -1, -2, -3],
    points: np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Given a set of k_points, plot the points perpendicular to kz_axis.

    Parameters
    ----------
    basis : Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]
    kz_axis : Literal[0, 1, 2, -1, -2, -3]
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
    projected_points = project_k_points_along_axis(basis, points, kz_axis)

    (line,) = ax.plot(*projected_points.reshape(2, -1))
    line.set_linestyle(" ")
    line.set_marker("x")

    ax.set_xlabel(f"k{0 if (kz_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"k{2 if (kz_axis % 3) != 2 else 1} axis")  # noqa: PLR2004
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, line


def plot_brillouin_zone_points_projected_2d(
    basis: Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv],
    kz_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the k coordinates in the first brillouin zone.

    Parameters
    ----------
    basis : Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]
    kz_axis : Literal[0, 1, 2, -1, -2, -3]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    util = Basis3dUtil((basis[0], basis[1], axis_as_single_point_axis(basis[2])))
    coordinates = decrement_brillouin_zone_3d(basis, util.nk_points)
    coordinates = decrement_brillouin_zone_3d(basis, coordinates)
    points = util.get_k_points_at_index(coordinates)

    return plot_k_points_projected_2d(basis, kz_axis, points, ax=ax)


def plot_bragg_points_projected_2d(
    basis: Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv],
    kz_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the bragg points in the first band projected perpendicular to kz_axis.

    Parameters
    ----------
    basis : Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]
    kz_axis : Literal[0, 1, 2, -1, -2, -3]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    points = get_all_brag_point(basis)
    return plot_k_points_projected_2d(basis, kz_axis, points, ax=ax)


def plot_fundamental_k_in_plane_projected_2d(
    basis: Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv],
    idx: SingleFlatIndexLike,
    kz_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the k coordinates in the basis perpendicular to kz_axis.

    Parameters
    ----------
    basis : Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]
    idx : int
        idx in kz_axis direction
    kz_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to which to plot the points
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    util = Basis3dUtil(basis)
    points = util.fundamental_k_points.reshape(3, *util.fundamental_shape)[
        slice_along_axis(idx, (kz_axis % 3) + 1)
    ].reshape(3, -1)
    return plot_k_points_projected_2d(basis, kz_axis, points, ax=ax)


def plot_x_points_projected_2d(
    basis: Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    points: np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Given a set of x_points, plot the points perpendicular to z_axis.

    Parameters
    ----------
    basis : Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]
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
    projected_points = project_x_points_along_axis(basis, points, z_axis)

    (line,) = ax.plot(*projected_points.reshape(2, -1))
    line.set_linestyle(" ")
    line.set_marker("x")

    ax.set_xlabel(f"x{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")  # noqa: PLR2004
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, line


def plot_fundamental_x_in_plane_projected_2d(
    basis: Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv],
    idx: SingleFlatIndexLike,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the fundamental x coordinates in the basis perpendicular to z_axis through the idx plane.

    Parameters
    ----------
    basis : Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]
    idx : int
        idx in z_axis direction
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to which to plot the points
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    util = Basis3dUtil(basis)
    points = util.fundamental_x_points.reshape(3, *util.fundamental_shape)[
        slice_along_axis(idx, (z_axis % 3) + 1)
    ].reshape(3, -1)
    return plot_x_points_projected_2d(basis, z_axis, points, ax=ax)


def plot_fundamental_x_at_index_projected_2d(
    basis: Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv],
    idx: IndexLike3d,
    z_axis: Literal[0, 1, 2, -1, -2, -3] = 2,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Given an index-like object, plot all the projected points in the given basis.

    Parameters
    ----------
    basis : Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]
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
    util = Basis3dUtil(basis)
    idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    points = util.fundamental_x_points.reshape(3, *util.fundamental_shape)[
        :, *idx
    ].reshape(3, -1)
    return plot_x_points_projected_2d(basis, z_axis, points, ax=ax)
