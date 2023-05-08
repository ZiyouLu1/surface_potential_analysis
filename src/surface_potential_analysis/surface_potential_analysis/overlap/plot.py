from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    calculate_cumulative_k_distances_along_path,
    get_fundamental_projected_k_points,
    get_fundamental_projected_x_points,
)
from surface_potential_analysis.util import (
    Measure,
    get_measured_data,
    slice_along_axis,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis import PositionBasis

    from .overlap import Overlap, OverlapTransform

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def plot_overlap_2d_x(
    overlap: Overlap[
        BasisConfig[PositionBasis[_L0Inv], PositionBasis[_L1Inv], PositionBasis[_L2Inv]]
    ],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the overlap in momentum space.

    Parameters
    ----------
    overlap : OverlapTransform
    idx : int
        index along z_axis
    z_axis : Literal[0, 1, 2,-1, -2, -3]
        axis perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    coordinates = get_fundamental_projected_x_points(overlap["basis"], z_axis)[
        slice_along_axis(idx, (z_axis % 3) + 1)
    ]
    util = BasisConfigUtil(overlap["basis"])
    points = overlap["vector"].reshape(*util.shape)[slice_along_axis(idx, z_axis)]
    data = get_measured_data(points, measure)

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"k{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"k{2 if (z_axis % 3) != 2 else 1} axis")  # noqa: PLR2004

    return fig, ax, mesh


def plot_overlap_2d_k(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the overlap in momentum space.

    Parameters
    ----------
    overlap : OverlapTransform
    idx : int
        index along z_axis
    z_axis : Literal[0, 1, 2,-1, -2, -3]
        axis perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    # TODO: shifted transform
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    coordinates = get_fundamental_projected_k_points(overlap["basis"], z_axis)[
        slice_along_axis(idx, (z_axis % 3) + 1)
    ]
    util = BasisConfigUtil(overlap["basis"])
    points = overlap["vector"].reshape(*util.shape)[slice_along_axis(idx, z_axis)]
    data = np.fft.ifftshift(get_measured_data(points, measure))
    shifted_coordinates = np.fft.ifftshift(coordinates)

    mesh = ax.pcolormesh(*shifted_coordinates, data, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"k{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"k{2 if (z_axis % 3) != 2 else 1} axis")  # noqa: PLR2004

    return fig, ax, mesh


def plot_overlap_k0k1(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv],
    idx: int,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot of the overlap in k perpendicular to the k2 axis.

    Parameters
    ----------
    overlap : OverlapTransform[_L0Inv, _L1Inv, _L2Inv]
    idx : int
        index along k2
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_overlap_2d_k(overlap, idx, 2, ax=ax, measure=measure, scale=scale)


def plot_overlap_k1k2(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv],
    idx: int,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot of the overlap in k perpendicular to the k0 axis.

    Parameters
    ----------
    overlap : OverlapTransform[_L0Inv, _L1Inv, _L2Inv]
    idx : int
        index along k0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_overlap_2d_k(overlap, idx, 0, ax=ax, measure=measure, scale=scale)


def plot_overlap_k2k0(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv],
    idx: int,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot of the overlap in k perpendicular to the k1 axis.

    Parameters
    ----------
    overlap : OverlapTransform[_L0Inv, _L1Inv, _L2Inv]
    idx : int
        index along k1
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_overlap_2d_k(overlap, idx, 1, ax=ax, measure=measure, scale=scale)


def plot_overlap_along_path_k(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the overlap transform along the given path.

    Parameters
    ----------
    overlap : OverlapTransform
    path : np.ndarray[tuple[3, int], np.dtype[np.int_]]
        path, as a list of index for each coordinate
    wrap_distances : bool, optional
        should the coordinates be wrapped into the unit cell, by default False
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;], optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = BasisConfigUtil(overlap["basis"])
    points = overlap["vector"].reshape(util.shape)[*path]
    data = get_measured_data(points, measure)
    distances = calculate_cumulative_k_distances_along_path(
        overlap["basis"], path, wrap_distances=wrap_distances
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    ax.set_xlabel("Distance along path")
    return fig, ax, line


def plot_overlap_along_k_diagonal(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv],
    k2_ind: int = 0,
    *,
    measure: Measure = "abs",
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the overlap transform in the x0, x1 diagonal.

    Parameters
    ----------
    overlap : OverlapTransform[_L0Inv, _L1Inv, _L2Inv]
    k2_ind : int, optional
        index in the k2 direction, by default 0
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;], optional
        measure, by default "abs"
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    util = BasisConfigUtil(overlap["basis"])
    path = np.array([[i, i, k2_ind] for i in range(util.shape[0])]).T

    return plot_overlap_along_path_k(overlap, path, measure=measure, scale=scale, ax=ax)


def plot_overlap_along_k0(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv],
    k1_ind: int = 0,
    k2_ind: int = 0,
    *,
    measure: Measure = "abs",
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot overlap transform in the k0 direction.

    Parameters
    ----------
    overlap : OverlapTransform[_L0Inv,_L1Inv,_L2Inv]
    k1_ind : int, optional
        index along k1, by default 0
    k2_ind : int, optional
        index along k2, by default 0
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;], optional
        measure, by default "abs"
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    util = BasisConfigUtil(overlap["basis"])
    path = np.array([[i, k1_ind, k2_ind] for i in range(util.shape[0])]).T

    return plot_overlap_along_path_k(overlap, path, measure=measure, scale=scale, ax=ax)
