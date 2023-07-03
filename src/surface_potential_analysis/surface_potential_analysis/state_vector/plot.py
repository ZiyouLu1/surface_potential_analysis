from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.axis.util import Axis3dUtil, AxisUtil
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
    calculate_cumulative_x_distances_along_path,
    get_k_coordinates_in_axes,
    get_x_coordinates_in_axes,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
    convert_state_vector_to_momentum_basis,
    convert_state_vector_to_position_basis,
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

if TYPE_CHECKING:
    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis._types import (
        SingleFlatIndexLike,
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.basis.basis import (
        Basis,
        Basis3d,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.util.plot import Scale

    from .state_vector import StateVector3d

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=Basis[Any])
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])


# ruff: noqa: PLR0913


def plot_state_vector_1d_k(
    state: StateVector[_B0Inv],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an eigenstate in 1d along the given axis.

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

    coordinates = AxisUtil(state["basis"][axes[0]]).fundamental_nk_points

    idx = tuple(0 for _ in range(len(state["basis"]) - 1)) if idx is None else idx
    data_slice: list[slice | int | np.integer[Any]] = list(idx)
    data_slice.insert(axes[0], slice(None))

    converted = convert_state_vector_to_momentum_basis(state)
    util = BasisUtil(converted["basis"])
    points = converted["vector"].reshape(util.shape)[tuple(data_slice)]
    data = get_measured_data(points, measure)

    (line,) = ax.plot(np.fft.fftshift(coordinates), np.fft.fftshift(data))
    ax.set_xlabel(f"k{axes[0]} axis")
    ax.set_ylabel("State /Au")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_state_vector_1d_x(
    state: StateVector[_B0Inv],
    axis: int = 0,
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an eigenstate in 1d along the given axis.

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

    fundamental_x_points = Axis3dUtil(state["basis"][axis]).fundamental_x_points
    coordinates = np.linalg.norm(fundamental_x_points, axis=0)

    idx = tuple(0 for _ in range(len(state["basis"]) - 1)) if idx is None else idx

    converted = convert_state_vector_to_position_basis(state)
    util = BasisUtil(converted["basis"])
    points = converted["vector"].reshape(util.shape)[slice_ignoring_axes(idx, (axis,))]
    data = get_measured_data(points, measure)

    (line,) = ax.plot(coordinates, data)
    ax.set_xlabel(f"x{(axis % 3)} axis")
    ax.set_ylabel("Eigenstate /Au")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_state_vector_2d_k(
    state: StateVector[_B0Inv],
    axes: tuple[int, int],
    idx: SingleStackedIndexLike | None = None,
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
    converted = convert_state_vector_to_momentum_basis(state)
    util = BasisUtil(converted["basis"])

    idx = tuple(0 for _ in range(len(state["basis"]) - 2)) if idx is None else idx
    coordinates = get_k_coordinates_in_axes(converted["basis"], axes, idx)

    points = converted["vector"].reshape(*util.shape)[slice_ignoring_axes(idx, axes)]
    data = get_measured_data(points, measure)

    data = np.fft.ifftshift(data)
    coordinates = np.fft.ifftshift(coordinates, axes=(1, 2))

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    norm = get_norm_with_clim(scale, mesh.get_clim())
    mesh.set_norm(norm)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"k{axes[0]} axis")
    ax.set_ylabel(f"k{axes[1]} axis")

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
    return plot_state_vector_2d_k(
        eigenstate, (0, 1), (k2_idx,), ax=ax, measure=measure, scale=scale
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
    return plot_state_vector_2d_k(
        eigenstate, (1, 2), (k0_idx,), ax=ax, measure=measure, scale=scale
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
    return plot_state_vector_2d_k(
        eigenstate, (2, 0), (k1_idx,), ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_2d_x(
    eigenstate: StateVector[_B0Inv],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
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
    converted = convert_state_vector_to_position_basis(eigenstate)
    util = BasisUtil(converted["basis"])
    idx = tuple(0 for _ in range(util.ndim - 2)) if idx is None else idx

    coordinates = get_x_coordinates_in_axes(converted["basis"], axes, idx)

    points = converted["vector"].reshape(*util.shape)[slice_ignoring_axes(idx, axes)]
    data = get_measured_data(points, measure)

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    norm = get_norm_with_clim(scale, mesh.get_clim())
    mesh.set_norm(norm)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{axes[0]} axis")
    ax.set_ylabel(f"x{axes[1]} axis")

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
        eigenstate, (0, 1), (x2_idx,), ax=ax, measure=measure, scale=scale
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
        eigenstate, (1, 2), (x0_idx,), ax=ax, measure=measure, scale=scale
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
        eigenstate, (2, 0), (x1_idx,), ax=ax, measure=measure, scale=scale
    )


def plot_state_vector_difference_1d_k(
    state_0: StateVector[_B0Inv],
    state_1: StateVector[_B1Inv],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the difference between two eigenstates in k.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateVector[_B1Inv]
    idx : SingleStackedIndexLike | None, optional
        index at each axis perpendicular to axis, by default None
    axis : int, optional
        axis to plot along, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    basis = basis_as_fundamental_momentum_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "vector": (converted_0["vector"] - converted_1["vector"])
        / np.max(
            [np.abs(converted_0["vector"]), np.abs(converted_1["vector"])], axis=0
        ),
    }
    return plot_state_vector_1d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)


def plot_state_vector_difference_2d_x(
    state_0: StateVector[_B0Inv],
    state_1: StateVector[_B1Inv],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two eigenstates in 2d, perpendicular to z_axis.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateVector[_B1Inv]
    idx : SingleStackedIndexLike
        index along each axis perpendicular to axes
    axes : tuple[int, int], optional
        axis to plot, by default (0, 1)
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
    basis = basis_as_fundamental_position_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    eigenstate: StateVector[Any] = {
        "basis": basis,
        "vector": (converted_0["vector"] - converted_1["vector"])
        / np.max(
            [np.abs(converted_0["vector"]), np.abs(converted_1["vector"])], axis=0
        ),
    }
    return plot_eigenstate_2d_x(
        eigenstate, axes, idx, ax=ax, measure=measure, scale=scale
    )


def plot_state_vector_difference_2d_k(
    state_0: StateVector[_B0Inv],
    state_1: StateVector[_B1Inv],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the difference between two eigenstates in k.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateVector[_B1Inv]
    idx : SingleStackedIndexLike | None, optional
        index at each axis perpendicular to axis, by default None
    axis : int, optional
        axis to plot along, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    basis = basis_as_fundamental_momentum_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "vector": (converted_0["vector"] - converted_1["vector"])
        / np.max(
            [np.abs(converted_0["vector"]), np.abs(converted_1["vector"])], axis=0
        ),
    }
    return plot_state_vector_2d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)


def animate_eigenstate_3d_x(
    eigenstate: StateVector[_B0Inv],
    axes: tuple[int, int],
    idx: SingleStackedIndexLike | None = None,
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
    converted = convert_state_vector_to_position_basis(eigenstate)

    coordinates = get_x_coordinates_in_axes(converted["basis"], axes, idx)
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
    return animate_eigenstate_3d_x(
        eigenstate, (0, 1), (0,), ax=ax, measure=measure, scale=scale
    )


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
    return animate_eigenstate_3d_x(
        eigenstate, (1, 2), (0,), ax=ax, measure=measure, scale=scale
    )


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
    return animate_eigenstate_3d_x(
        eigenstate, (2, 0), (0,), ax=ax, measure=measure, scale=scale
    )


def plot_state_vector_along_path(
    state: StateVector[_B0Inv],
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
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
    state : Eigenstate[_B3d0Inv]
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
    converted = convert_state_vector_to_position_basis(state)  # type: ignore[var-annotated,arg-type]

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
