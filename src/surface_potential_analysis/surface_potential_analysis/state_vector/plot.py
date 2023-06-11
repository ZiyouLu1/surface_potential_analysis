from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.axis.util import Axis3dUtil
from surface_potential_analysis.basis.util import (
    Basis3dUtil,
    BasisUtil,
    calculate_cumulative_x_distances_along_path,
    get_fundamental_projected_k_points,
    get_fundamental_projected_x_points,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_eigenstate_to_momentum_basis,
    convert_eigenstate_to_position_basis,
)
from surface_potential_analysis.util.plot import animate_through_surface
from surface_potential_analysis.util.util import (
    Measure,
    get_measured_data,
    slice_along_axis,
)

if TYPE_CHECKING:
    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis._types import SingleFlatIndexLike
    from surface_potential_analysis.basis.basis import (
        Basis,
        Basis3d,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.util.plot import Scale

    from .state_vector import FundamentalPositionBasisEigenstate3d, StateVector3d

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)


def plot_eigenstate_1d_x(
    eigenstate: StateVector[_B0Inv],
    idx: tuple[int, ...] | None = None,
    axis: Literal[0, 1, 2, -1, -2, -3] = 0,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an eigenstate in 1d along the given axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    idx : tuple[int, int], optional
        index in the perpendicular directions, by default (0,0)
    axis : Literal[0, 1, 2, -1, -2, -3], optional
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

    fundamental_x_points = Axis3dUtil(eigenstate["basis"][axis]).fundamental_x_points
    coordinates = np.linalg.norm(fundamental_x_points, axis=0)

    idx = tuple(0 for _ in range(len(eigenstate["basis"]) - 1)) if idx is None else idx
    data_slice: list[slice | int] = list(idx)
    data_slice.insert(axis, slice(None))

    converted = convert_eigenstate_to_position_basis(eigenstate)
    util = BasisUtil(converted["basis"])
    points = converted["vector"].reshape(util.shape)[tuple(data_slice)]
    data = get_measured_data(points, measure)

    (line,) = ax.plot(coordinates, data)
    ax.set_xlabel(f"x{(axis % 3)} axis")
    ax.set_ylabel("Eigenstate /Au")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_eigenstate_2d_k(
    eigenstate: StateVector3d[_B3d0Inv],
    idx: SingleFlatIndexLike,
    kz_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d, perpendicular to kz_axis in momentum basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    idx : SingleFlatIndexLike
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
    converted = convert_eigenstate_to_momentum_basis(eigenstate)  # type: ignore[var-annotated,arg-type]

    coordinates = get_fundamental_projected_k_points(converted["basis"], kz_axis)[
        slice_along_axis(idx, (kz_axis % 3) + 1)
    ]
    util = Basis3dUtil(converted["basis"])
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
    eigenstate: StateVector3d[_B3d0Inv],
    k2_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the k2 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    k2_idx : SingleFlatIndexLike
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
    eigenstate: StateVector3d[_B3d0Inv],
    k0_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the k0 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    k0_idx : SingleFlatIndexLike
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
    eigenstate: StateVector3d[_B3d0Inv],
    k1_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the k1 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    k1_idx : SingleFlatIndexLike
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
    eigenstate: StateVector3d[_B3d0Inv],
    idx: SingleFlatIndexLike,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d, perpendicular to z_axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    idx : SingleFlatIndexLike
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
    converted = convert_eigenstate_to_position_basis(eigenstate)  # type: ignore[var-annotated,arg-type]

    coordinates = get_fundamental_projected_x_points(converted["basis"], z_axis)[  # type: ignore[type-var]
        slice_along_axis(idx, (z_axis % 3) + 1)
    ]
    util = BasisUtil(converted["basis"])
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
    eigenstate: StateVector3d[_B3d0Inv],
    x2_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the x2 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    x2_idx : SingleFlatIndexLike
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
    eigenstate: StateVector3d[_B3d0Inv],
    x0_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the x0 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    x0_idx : SingleFlatIndexLike
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
    eigenstate: StateVector3d[_B3d0Inv],
    x1_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an eigenstate in 2d perpendicular to the x1 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    x1_idx : SingleFlatIndexLike
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
    eigenstate_0: FundamentalPositionBasisEigenstate3d[_L0Inv, _L1Inv, _L2Inv],
    eigenstate_1: FundamentalPositionBasisEigenstate3d[_L0Inv, _L1Inv, _L2Inv],
    idx: SingleFlatIndexLike,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two eigenstates in 2d, perpendicular to z_axis.

    Parameters
    ----------
    eigenstate_0 : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    eigenstate_1 : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    idx : SingleFlatIndexLike
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
    eigenstate: FundamentalPositionBasisEigenstate3d[_L0Inv, _L1Inv, _L2Inv] = {
        "basis": eigenstate_0["basis"],
        "vector": eigenstate_0["vector"] - eigenstate_1["vector"],
    }
    return plot_eigenstate_2d_x(
        eigenstate, idx, z_axis, ax=ax, measure=measure, scale=scale
    )


def animate_eigenstate_3d_x(
    eigenstate: StateVector3d[_B3d0Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate an eigenstate in 3d, perpendicular to z_axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
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
    converted = convert_eigenstate_to_position_basis(eigenstate)  # type: ignore[var-annotated,arg-type]

    coordinates = get_fundamental_projected_x_points(converted["basis"], z_axis)  # type: ignore[type-var]
    util = BasisUtil(converted["basis"])
    points = converted["vector"].reshape(*util.shape)
    data = get_measured_data(points, measure)

    c_min = 0 if clim[0] is None and measure == "abs" else clim[0]
    return animate_through_surface(
        coordinates, data, z_axis, ax=ax, scale=scale, clim=(c_min, clim[1])
    )


def animate_eigenstate_x0x1(
    eigenstate: StateVector3d[_B3d0Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    plot an eigenstate in 3d perpendicular to the x2 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
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
    eigenstate: StateVector3d[_B3d0Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    plot an eigenstate in 3d perpendicular to the x0 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
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
    eigenstate: StateVector3d[_B3d0Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    plot an eigenstate in 3d perpendicular to the x1 axis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
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
    eigenstate: StateVector3d[_B3d0Inv],
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an eigenstate in 1d along the given path in position basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
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
    converted = convert_eigenstate_to_position_basis(eigenstate)  # type: ignore[var-annotated,arg-type]

    util = BasisUtil(converted["basis"])
    points = converted["vector"].reshape(*util.shape)[*path]
    data = get_measured_data(points, measure)
    distances = calculate_cumulative_x_distances_along_path(
        converted["basis"], path, wrap_distances=wrap_distances  # type: ignore[arg-type]
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    ax.set_xlabel("distance /m")
    return fig, ax, line
