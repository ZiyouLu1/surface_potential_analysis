from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import Normalize, SymLogNorm

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    PositionBasisConfigUtil,
    get_fundamental_projected_x_points,
)
from surface_potential_analysis.eigenstate.conversion import convert_eigenstate_to_basis
from surface_potential_analysis.util import (
    calculate_cumulative_distances_along_path,
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


def _convert_eigenstate_to_position(
    eigenstate: Eigenstate[_BC0Inv],
) -> PositionBasisEigenstate[Any, Any, Any]:
    util = BasisConfigUtil(eigenstate["basis"])
    return convert_eigenstate_to_basis(
        eigenstate, util.get_fundamental_basis_in("position")
    )


def plot_eigenstate_2d(
    eigenstate: Eigenstate[_BC0Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
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
    eigenstate = _convert_eigenstate_to_position(eigenstate)

    coordinates = get_fundamental_projected_x_points(eigenstate["basis"], z_axis)[
        slice_along_axis(idx, (z_axis % 3) + 1)
    ]
    util = PositionBasisConfigUtil(eigenstate["basis"])
    points = eigenstate["vector"].reshape(*util.shape)[slice_along_axis(idx, z_axis)]
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
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the x2 axis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
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
    return plot_eigenstate_2d(
        eigenstate, x2_idx, 2, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_x1x2(
    eigenstate: Eigenstate[_BC0Inv],
    x0_idx: int,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the x0 axis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
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
    return plot_eigenstate_2d(
        eigenstate, x0_idx, 0, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_x2x0(
    eigenstate: Eigenstate[_BC0Inv],
    x1_idx: int,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the x1 axis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
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
    return plot_eigenstate_2d(
        eigenstate, x1_idx, 1, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_difference_2d(
    eigenstate0: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    eigenstate1: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two eigenstates in 2d, perpendicular to z_axis.

    Parameters
    ----------
    eigenstate0 : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    eigenstate1 : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
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
        "basis": eigenstate0["basis"],
        "vector": eigenstate0["vector"] - eigenstate1["vector"],
    }
    return plot_eigenstate_2d(
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


def animate_eigenstate_3d(
    eigenstate: Eigenstate[_BC0Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate an eigenstate in 3d, perpendicular to z_axis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
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
    eigenstate = _convert_eigenstate_to_position(eigenstate)

    coordinates = get_fundamental_projected_x_points(eigenstate["basis"], z_axis)
    util = PositionBasisConfigUtil(eigenstate["basis"])
    points = eigenstate["vector"].reshape(*util.shape)
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
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    plot an eigenstate in 3d perpendicular to the x2 axis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
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
    return animate_eigenstate_3d(eigenstate, 2, ax=ax, measure=measure, scale=scale)


def animate_eigenstate_x1x2(
    eigenstate: Eigenstate[_BC0Inv],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    plot an eigenstate in 3d perpendicular to the x0 axis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
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
    return animate_eigenstate_3d(eigenstate, 0, ax=ax, measure=measure, scale=scale)


def animate_eigenstate_x2x0(
    eigenstate: Eigenstate[_BC0Inv],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    plot an eigenstate in 3d perpendicular to the x1 axis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
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
    return animate_eigenstate_3d(eigenstate, 1, ax=ax, measure=measure, scale=scale)


def plot_eigenstate_along_path(
    eigenstate: Eigenstate[_BC0Inv],
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an eigenstate in 1d along the given path in position basis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path, as a list of [x0_coords, x1_coords, x2_coords]
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
    eigenstate = _convert_eigenstate_to_position(eigenstate)

    util = PositionBasisConfigUtil(eigenstate["basis"])
    points = eigenstate["vector"].reshape(*util.shape)[*path]
    data = get_measured_data(points, measure)
    distances = calculate_cumulative_distances_along_path(
        path, util.fundamental_x_points.reshape(3, *util.shape)
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    return fig, ax, line
