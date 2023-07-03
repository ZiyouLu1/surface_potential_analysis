from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.axis.util import AxisUtil
from surface_potential_analysis.basis.util import (
    Basis3dUtil,
    BasisUtil,
    calculate_cumulative_x_distances_along_path,
    get_x_coordinates_in_axes,
)
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_position_basis,
)
from surface_potential_analysis.util.plot import (
    animate_through_surface,
    get_norm_with_clim,
)
from surface_potential_analysis.util.util import (
    Measure,
    get_measured_data,
    slice_ignoring_axes,
)

from ._comparison_points import (
    get_100_comparison_points_x2,
    get_111_comparison_points_x2,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis._types import (
        SingleFlatIndexLike,
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.basis.basis import Basis
    from surface_potential_analysis.potential.potential import Potential
    from surface_potential_analysis.util.plot import Scale

    from .potential import FundamentalPositionBasisPotential3d

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def plot_potential_1d_x(
    potential: Potential[_B0Inv],
    axis: int = 0,
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the potential along the given axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int]
    axis : Literal[0, 1, 2, -1, -2, -3]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    idx = tuple(0 for _ in range(len(potential["basis"]) - 1)) if idx is None else idx

    util = AxisUtil(potential["basis"][axis])
    coordinates = np.linalg.norm(util.fundamental_x_points, axis=0)
    data = potential["vector"].reshape(BasisUtil(potential["basis"]).shape)[
        slice_ignoring_axes(idx, (axis,))
    ]

    (line,) = ax.plot(coordinates, data)
    ax.set_xlabel(f"x{axis} axis")
    ax.set_ylabel("Energy /J")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_potential_1d_comparison(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    comparison_points: Mapping[
        str, tuple[tuple[int, int], Literal[0, 1, 2, -1, -2, -3]]
    ],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot the potential in 1d at the given comparison points.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    comparison_points : Mapping[ str, tuple[tuple[int, int], Literal[0, 1, 2,
        map of axis label to ((idx), axis) to pass to plot_potential_1d
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    lines: list[Line2D] = []
    for label, (idx, axis) in comparison_points.items():
        _, _, line = plot_potential_1d_x(potential, axis, idx, ax=ax, scale=scale)
        line.set_label(label)
        lines.append(line)
    ax.legend()
    return fig, ax, lines


def plot_potential_1d_x2_comparison_111(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    offset: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot the potential along the x2 at the relevant 111 sites.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    offset : tuple[int, int]
        index of the fcc site
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    (s0, s1, _) = Basis3dUtil(potential["basis"]).shape
    points = get_111_comparison_points_x2((s0, s1), offset)
    comparison_points = {k: (v, 2) for (k, v) in points.items()}
    return plot_potential_1d_comparison(
        potential, comparison_points, ax=ax, scale=scale  # type: ignore[arg-type]
    )


def plot_potential_1d_x2_comparison_100(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    offset: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, list[Line2D]]:
    """
    Plot the potential along the x2 at the relevant 100 sites.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    offset : tuple[int, int]
        index of the fcc site
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    (s0, s1, _) = Basis3dUtil(potential["basis"]).shape
    points = get_100_comparison_points_x2((s0, s1), offset)
    comparison_points = {k: (v, 2) for (k, v) in points.items()}
    return plot_potential_1d_comparison(
        potential, comparison_points, ax=ax, scale=scale  # type: ignore[arg-type]
    )


def plot_potential_2d_x(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the potential in 2d, perpendicular to z_axis at idx along z_axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    idx : SingleFlatIndexLike
        index along z_axis
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to direction of plot
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = BasisUtil(potential["basis"])
    idx = tuple(0 for _ in range(util.ndim - 2)) if idx is None else idx

    coordinates = get_x_coordinates_in_axes(potential["basis"], axes, idx)
    data = potential["vector"].reshape(util.shape)[slice_ignoring_axes(idx, axes)]

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    norm = get_norm_with_clim(scale, mesh.get_clim())
    mesh.set_norm(norm)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{axes[0]} axis")
    ax.set_ylabel(f"x{axes[1]} axis")

    return fig, ax, mesh


def plot_potential_x0x1(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    x2_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the potential in 2d, perpendicular to x3 at x3_idx.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    x3_idx : SingleFlatIndexLike
        index along x3_axis
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_potential_2d_x(potential, (0, 1), (x2_idx,), ax=ax, scale=scale)


def plot_potential_x1x2(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    x0_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the potential in 2d, perpendicular to x2 at x2_idx.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    x2_idx : SingleFlatIndexLike
        index along x2_axis
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_potential_2d_x(potential, (1, 2), (x0_idx,), ax=ax, scale=scale)


def plot_potential_x2x0(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    x1_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the potential in 2d, perpendicular to x1 at x1_idx.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    x1_idx : SingleFlatIndexLike
        index along x1_axis
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_potential_2d_x(potential, (2, 0), (x1_idx,), ax=ax, scale=scale)


def plot_potential_difference_2d_x(
    potential0: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    potential1: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two potentials in 2d, perpendicular to z_axis.

    Parameters
    ----------
    potential0 : Potential[_L0Inv, _L1Inv, _L2Inv]
    potential1 : Potential[_L0Inv, _L1Inv, _L2Inv]
    idx : SingleFlatIndexLike
        index along z_axis
    z_axis : Literal[0, 1, 2,
        axis perpendicular to which to plot the data
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv] = {
        "basis": potential0["basis"],
        "vector": potential0["vector"] - potential1["vector"],
    }
    return plot_potential_2d_x(potential, axes, idx, ax=ax, scale=scale)


def animate_potential_3d_x(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate potential in the direction perpendicular to z_axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis to animate through
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    util = BasisUtil(potential["basis"])
    coordinates = get_x_coordinates_in_axes(potential["basis"], z_axis)
    points = potential["vector"].reshape(util.fundamental_shape)
    data = get_measured_data(points, measure)
    fig, ax, ani = animate_through_surface(  # type: ignore[misc]
        coordinates, data, z_axis, ax=ax, scale=scale, clim=clim
    )
    ax.set_title(f"Animation of the potential perpendicular to the x{z_axis % 3} axis")

    return fig, ax, ani


def animate_potential_x0x1(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the potential in the direction perpendicular to x0x1.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axes, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_potential_3d_x(potential, 2, ax=ax, scale=scale, clim=clim)


def animate_potential_x1x2(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the potential in the direction perpendicular to x0x1.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axes, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_potential_3d_x(potential, 0, ax=ax, scale=scale, clim=clim)


def animate_potential_x2x0(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the potential in the direction perpendicular to x0x1.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axes, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_potential_3d_x(potential, 1, ax=ax, scale=scale, clim=clim)


def animate_potential_difference_2d_x(
    potential0: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    potential1: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Animate the absolute relative difference between two potentials in 2d, perpendicular to z_axis.

    Parameters
    ----------
    potential0 : Potential[_L0Inv, _L1Inv, _L2Inv]
    potential1 : Potential[_L0Inv, _L1Inv, _L2Inv]
    z_axis : Literal[0, 1, 2,
        axis perpendicular to which to plot the data
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv] = {
        "basis": potential0["basis"],
        "vector": np.abs(
            (potential0["vector"] - potential1["vector"]) / potential0["vector"]
        ),
    }
    return animate_potential_3d_x(potential, z_axis, ax=ax, scale=scale, clim=clim)


def plot_potential_along_path(
    potential: Potential[_B0Inv],
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the potential along the given path.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    path : np.ndarray[tuple[Literal[_ND], int], np.dtype[np.int_]]
    wrap_distances : bool, optional
        should the coordinates be wrapped into the unit cell, by default False
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    converted = convert_potential_to_position_basis(potential)
    util = BasisUtil(converted["basis"])

    data = get_measured_data(converted["vector"].reshape(util.shape)[*path], measure)
    distances = calculate_cumulative_x_distances_along_path(
        converted["basis"], path, wrap_distances=wrap_distances
    )

    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    return fig, ax, line


def get_minimum_path(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]],
    axis: Literal[0, 1, 2, -1, -2, -3] = -1,
) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]:
    """
    Find the minimum path, taking the smallest value along the given axis.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
        Potential to take the minimum of
    path : np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]]
        Path for the coordinates perpendicular to axis
    axis : Literal[0, 1, 2,-1,-2,-3] optional
        Axis over which to search for the minimum of potential, by default -1

    Returns
    -------
    np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        The complete path after finding the minimum at each coordinate
    """
    util = BasisUtil(potential["basis"])
    min_idx = np.argmin(potential["vector"].reshape(util.shape), axis=axis)
    min_idx_path = min_idx[*path]
    full_path = np.empty((3, path.shape[1]))
    full_path[1 if (axis % 3) == 0 else 0] = path[0]
    full_path[1 if (axis % 3) == 2 else 3] = path[1]  # noqa: PLR2004
    full_path[axis] = min_idx_path
    return full_path  # type: ignore[no-any-return]


def plot_potential_minimum_along_path(
    potential: FundamentalPositionBasisPotential3d[_L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]],
    axis: Literal[0, 1, 2, -1, -2, -3] = -1,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the potential along the path, taking the minimum in the axis direction.

    Parameters
    ----------
    potential : Potential[_L0Inv, _L1Inv, _L2Inv]
    path : np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]]
        path perpendicular to axis
    axis : Literal[0, 1, 2, -1, -2, -3], optional
        axis to take the minimum along, by default -1
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    full_path = get_minimum_path(potential, path, axis)
    return plot_potential_along_path(potential, full_path, ax=ax, scale=scale)  # type: ignore[arg-type]
