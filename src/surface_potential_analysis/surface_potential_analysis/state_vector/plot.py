from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.axis.util import (
    BasisUtil,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.stacked_basis.util import (
    calculate_cumulative_x_distances_along_path,
    get_k_coordinates_in_axes,
    get_max_idx,
    get_x_coordinates_in_axes,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
    convert_state_vector_to_momentum_basis,
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.util.plot import (
    animate_through_surface_x,
    get_norm_with_clim,
)
from surface_potential_analysis.util.util import (
    Measure,
    get_data_in_axes,
    get_measured_data,
)

if TYPE_CHECKING:
    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.axis.stacked_axis import StackedBasisLike
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.types import (
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.util.plot import Scale


# ruff: noqa: PLR0913
def plot_state_1d_k(
    state: StateVector[StackedBasisLike[*tuple[Any, ...]]],
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

    converted = convert_state_vector_to_momentum_basis(state)
    idx = get_max_idx(converted, axes) if idx is None else idx
    data_slice: list[slice | int | np.integer[Any]] = list(idx)
    data_slice.insert(axes[0], slice(None))

    util = BasisUtil(converted["basis"])
    coordinates = util.fundamental_stacked_nk_points[0]
    points = get_data_in_axes(converted["data"].reshape(util.shape), axes, idx)
    data = get_measured_data(points, measure)

    (line,) = ax.plot(np.fft.fftshift(coordinates), np.fft.fftshift(data))
    ax.set_xlabel(f"k{axes[0]} axis")
    ax.set_ylabel("State /Au")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_state_1d_x(
    state: StateVector[StackedBasisLike[*tuple[Any, ...]]],
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

    converted = convert_state_vector_to_position_basis(state)
    idx = get_max_idx(converted, axes) if idx is None else idx

    util = BasisUtil(converted["basis"])
    fundamental_x_points = util.fundamental_x_points_stacked
    coordinates = np.linalg.norm(fundamental_x_points, axis=0)
    points = get_data_in_axes(converted["data"].reshape(util.shape), axes, idx)
    data = get_measured_data(points, measure)

    (line,) = ax.plot(coordinates, data)
    ax.set_xlabel(f"x{(axes[0] % 3)} axis")
    ax.set_ylabel("Eigenstate /Au")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_state_2d_k(
    state: StateVector[StackedBasisLike[*tuple[Any, ...]]],
    axes: tuple[int, int],
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an state in 2d, perpendicular to kz_axis in momentum basis.

    Parameters
    ----------
    state : Eigenstate[_B3d0Inv]
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

    idx = get_max_idx(converted, axes) if idx is None else idx
    coordinates = get_k_coordinates_in_axes(converted["basis"], axes, idx)
    points = get_data_in_axes(converted["data"].reshape(util.shape), axes, idx)

    data = np.fft.fftshift(get_measured_data(points, measure))
    coordinates = np.fft.fftshift(coordinates, axes=(1, 2))

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    norm = get_norm_with_clim(scale, mesh.get_clim())  # type: ignore Type of "get_clim" is partially unknown
    mesh.set_norm(norm)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"k{axes[0]} axis")
    ax.set_ylabel(f"k{axes[1]} axis")
    ax.text(
        0.05,
        0.95,
        f"k = {idx}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    return fig, ax, mesh


def plot_state_difference_2d_k(
    state_0: StateVector[StackedBasisLike[*tuple[Any, ...]]],
    state_1: StateVector[StackedBasisLike[*tuple[Any, ...]]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
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
    basis = stacked_basis_as_fundamental_momentum_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "data": (converted_0["data"] - converted_1["data"])
        / np.max([np.abs(converted_0["data"]), np.abs(converted_1["data"])], axis=0),
    }
    return plot_state_2d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)


def plot_state_2d_x(
    state: StateVector[StackedBasisLike[*tuple[Any, ...]]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an state in 2d, perpendicular to z_axis.

    Parameters
    ----------
    state : Eigenstate[_B3d0Inv]
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
    converted = convert_state_vector_to_position_basis(state)
    idx = get_max_idx(converted, axes) if idx is None else idx

    coordinates = get_x_coordinates_in_axes(converted["basis"], axes, idx)
    points = get_data_in_axes(
        converted["data"].reshape(converted["basis"].shape), axes, idx
    )
    data = get_measured_data(points, measure)

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    norm = get_norm_with_clim(scale, mesh.get_clim())  # type: ignore Type of "get_clim" is partially unknown
    mesh.set_norm(norm)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{axes[0]} axis")
    ax.set_ylabel(f"x{axes[1]} axis")
    ax.text(
        0.05,
        0.95,
        f"x = {idx}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    return fig, ax, mesh


def plot_state_difference_1d_k(
    state_0: StateVector[StackedBasisLike[*tuple[Any, ...]]],
    state_1: StateVector[StackedBasisLike[*tuple[Any, ...]]],
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
    basis = stacked_basis_as_fundamental_momentum_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "data": (converted_0["data"] - converted_1["data"])
        / np.max([np.abs(converted_0["data"]), np.abs(converted_1["data"])], axis=0),
    }
    return plot_state_1d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)


def plot_state_difference_2d_x(
    state_0: StateVector[StackedBasisLike[*tuple[Any, ...]]],
    state_1: StateVector[StackedBasisLike[*tuple[Any, ...]]],
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
    basis = stacked_basis_as_fundamental_position_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "data": (converted_0["data"] - converted_1["data"])
        / np.max([np.abs(converted_0["data"]), np.abs(converted_1["data"])], axis=0),
    }
    return plot_state_2d_x(state, axes, idx, ax=ax, measure=measure, scale=scale)


def animate_state_3d_x(
    state: StateVector[StackedBasisLike[*tuple[Any, ...]]],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate a state in 3d, perpendicular to z_axis.

    Parameters
    ----------
    state : Eigenstate[_B3d0Inv]
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
    converted = convert_state_vector_to_position_basis(state)
    util = BasisUtil(converted["basis"])
    points = converted["data"].reshape(*util.shape)

    return animate_through_surface_x(
        converted["basis"],
        points,
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
        clim=clim,
    )


def plot_state_along_path(
    state: StateVector[StackedBasisLike[*tuple[Any, ...]]],
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an state in 1d along the given path in position basis.

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
    points = converted["data"].reshape(*util.shape)[*path]
    data = get_measured_data(points, measure)
    distances = calculate_cumulative_x_distances_along_path(
        converted["basis"], path, wrap_distances=wrap_distances  # type: ignore[arg-type]
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    ax.set_xlabel("distance /m")
    return fig, ax, line
