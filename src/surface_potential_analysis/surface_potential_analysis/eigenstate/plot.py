from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import Normalize, SymLogNorm

from surface_potential_analysis.basis.basis import BasisUtil
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    calculate_cumulative_x_distances_along_path,
    get_fundamental_projected_k_points,
    get_fundamental_projected_x_points,
)
from surface_potential_analysis.eigenstate.conversion import (
    convert_eigenstate_to_momentum_basis,
    convert_eigenstate_to_position_basis,
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

    from .eigenstate import Eigenstate, PositionBasisEigenstate

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def plot_eigenstate_1d_x(
    eigenstate: Eigenstate[_BC0Inv],
    idx: tuple[int, int] = (0, 0),
    axis: Literal[0, 1, 2, -1, -2, -3] = 2,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an eigenstate in 1d along the given axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    idx : tuple[int, int], optional
        index in the perpendicular directions, by default (0,0)
    z_axis : Literal[0, 1, 2, -1, -2, -3], optional
        axis along which to plot, by default 2
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

    util = BasisUtil(eigenstate["basis"][axis])
    coordinates = np.linalg.norm(util.fundamental_x_points, axis=0)
    data_slice: list[slice | int] = [slice(None), slice(None), slice(None)]
    data_slice[1 if (axis % 3) == 0 else 0] = idx[0]
    data_slice[1 if (axis % 3) == 2 else 2] = idx[1]  # noqa: PLR2004
    converted = convert_eigenstate_to_position_basis(eigenstate)
    points = converted["vector"][tuple(data_slice)]
    data = get_measured_data(points, measure)

    (line,) = ax.plot(coordinates, data)
    ax.set_xlabel(f"x{(axis % 3)} axis")
    ax.set_ylabel("Eigenstate /Au")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_eigenstate_2d_k(
    eigenstate: Eigenstate[_BC0Inv],
    idx: int,
    kz_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d, perpendicular to kz_axis in momentum basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    idx : int
        index along z_axis to plot
    kz_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    converted = convert_eigenstate_to_momentum_basis(eigenstate)

    coordinates = get_fundamental_projected_k_points(converted["basis"], kz_axis)[
        slice_along_axis(idx, (kz_axis % 3) + 1)
    ]
    util = BasisConfigUtil(converted["basis"])
    points = converted["vector"].reshape(*util.shape)[slice_along_axis(idx, kz_axis)]
    data = get_measured_data(points, measure)

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"k{0 if (kz_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"k{2 if (kz_axis % 3) != 2 else 1} axis")  # noqa: PLR2004

    return fig, ax, mesh


def plot_eigenstate_k0k1(
    eigenstate: Eigenstate[_BC0Inv],
    k2_idx: int,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the k2 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    k2_idx : int
        index along the k2 axis to plot
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
    return plot_eigenstate_2d_k(
        eigenstate, k2_idx, 2, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_k1k2(
    eigenstate: Eigenstate[_BC0Inv],
    k0_idx: int,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the k0 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    k0_idx : int
        index along the k0 axis to plot
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
    return plot_eigenstate_2d_k(
        eigenstate, k0_idx, 0, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_k2k0(
    eigenstate: Eigenstate[_BC0Inv],
    k1_idx: int,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the k1 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    k1_idx : int
        index along the k1 axis to plot
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
    return plot_eigenstate_2d_k(
        eigenstate, k1_idx, 1, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_2d_x(
    eigenstate: Eigenstate[_BC0Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d, perpendicular to z_axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    idx : int
        index along z_axis to plot
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    converted = convert_eigenstate_to_position_basis(eigenstate)

    coordinates = get_fundamental_projected_x_points(converted["basis"], z_axis)[
        slice_along_axis(idx, (z_axis % 3) + 1)
    ]
    util = BasisConfigUtil(converted["basis"])
    points = converted["vector"].reshape(*util.shape)[slice_along_axis(idx, z_axis)]
    data = get_measured_data(points, measure)

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")  # noqa: PLR2004

    return fig, ax, mesh


def plot_eigenstate_x0x1(
    eigenstate: Eigenstate[_BC0Inv],
    x2_idx: int,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the x2 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    x2_idx : int
        index along the x2 axis to plot
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
    return plot_eigenstate_2d_x(
        eigenstate, x2_idx, 2, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_x1x2(
    eigenstate: Eigenstate[_BC0Inv],
    x0_idx: int,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the x0 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    x0_idx : int
        index along the x0 axis to plot
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
    return plot_eigenstate_2d_x(
        eigenstate, x0_idx, 0, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_x2x0(
    eigenstate: Eigenstate[_BC0Inv],
    x1_idx: int,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the x1 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    x1_idx : int
        index along the x1 axis to plot
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
    return plot_eigenstate_2d_x(
        eigenstate, x1_idx, 1, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_difference_2d_x(
    eigenstate_0: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    eigenstate_1: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two eigenstates in 2d, perpendicular to z_axis.

    Parameters
    ----------
    eigenstate_0 : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    eigenstate_1 : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    idx : int
        index along z_axis to plot
    z_axis : Literal[0, 1, 2, -1, -2, -3]
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
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv] = {
        "basis": eigenstate_0["basis"],
        "vector": eigenstate_0["vector"] - eigenstate_1["vector"],
    }
    return plot_eigenstate_2d_x(
        eigenstate, idx, z_axis, ax=ax, measure=measure, scale=scale
    )


def _get_norm(
    scale: Literal["symlog", "linear"],
    clim: tuple[float | None, float | None] = (None, None),
) -> Normalize:
    match scale:
        case "linear":
            return Normalize(vmin=clim[0], vmax=clim[1])
        case "symlog":
            return SymLogNorm(
                vmin=clim[0],
                vmax=clim[1],
                linthresh=None if clim[1] is None else 1e-4 * clim[1],
            )


def animate_eigenstate_3d_x(
    eigenstate: Eigenstate[_BC0Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate an eigenstate in 3d, perpendicular to z_axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    converted = convert_eigenstate_to_position_basis(eigenstate)

    coordinates = get_fundamental_projected_x_points(converted["basis"], z_axis)
    util = BasisConfigUtil(converted["basis"])
    points = converted["vector"].reshape(*util.shape)
    data = get_measured_data(points, measure)

    mesh0 = ax.pcolormesh(
        *coordinates[slice_along_axis(0, (z_axis % 3) + 1)],
        data[slice_along_axis(0, (z_axis % 3))],
        shading="nearest",
    )

    frames: list[list[QuadMesh]] = []
    for i in range(data.shape[z_axis]):
        mesh = ax.pcolormesh(
            *coordinates[slice_along_axis(i, (z_axis % 3) + 1)],
            data[slice_along_axis(i, (z_axis % 3))],
            shading="nearest",
        )
        frames.append([mesh])

    max_clim: float = np.max([i[0].get_clim()[1] for i in frames])
    min_clim: float = (
        0 if measure == "abs" else np.min([i[0].get_clim()[0] for i in frames])
    )
    clim = (min_clim, max_clim)
    norm = _get_norm(scale, clim)
    for (mesh,) in frames:
        mesh.set_norm(norm)
        mesh.set_clim(*clim)

    mesh0.set_norm(norm)
    mesh0.set_clim(*clim)

    ani = ArtistAnimation(fig, frames)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")  # noqa: PLR2004

    return fig, ax, ani


def animate_eigenstate_x0x1(
    eigenstate: Eigenstate[_BC0Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    plot an eigenstate in 3d perpendicular to the x2 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_eigenstate_3d_x(eigenstate, 2, ax=ax, measure=measure, scale=scale)


def animate_eigenstate_x1x2(
    eigenstate: Eigenstate[_BC0Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    plot an eigenstate in 3d perpendicular to the x0 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_eigenstate_3d_x(eigenstate, 0, ax=ax, measure=measure, scale=scale)


def animate_eigenstate_x2x0(
    eigenstate: Eigenstate[_BC0Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    plot an eigenstate in 3d perpendicular to the x1 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_eigenstate_3d_x(eigenstate, 1, ax=ax, measure=measure, scale=scale)


def plot_eigenstate_along_path(
    eigenstate: Eigenstate[_BC0Inv],
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an eigenstate in 1d along the given path in position basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path, as a list of [x0_coords, x1_coords, x2_coords]
    wrap_distances : bool, optional
        should the coordinates be wrapped into the unit cell, by default False
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    converted = convert_eigenstate_to_position_basis(eigenstate)

    util = BasisConfigUtil(converted["basis"])
    points = converted["vector"].reshape(*util.shape)[*path]
    data = get_measured_data(points, measure)
    distances = calculate_cumulative_x_distances_along_path(
        converted["basis"], path, wrap_distances=wrap_distances
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    ax.set_xlabel("distance /m")
    return fig, ax, line
