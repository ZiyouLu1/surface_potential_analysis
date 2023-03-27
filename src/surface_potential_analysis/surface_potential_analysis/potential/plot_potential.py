from typing import Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from surface_potential_analysis.basis import PositionBasisUtil
from surface_potential_analysis.basis_config import (
    PositionBasisConfigUtil,
    get_projected_x_points,
)
from surface_potential_analysis.util import (
    calculate_cumulative_distances_along_path,
    slice_along_axis,
)

from .potential import Potential

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def plot_potential_1D(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    idx: tuple[int, int],
    axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = PositionBasisUtil(potential["basis"][axis])  # type: ignore
    coordinates = util.x_points
    data_slice: list[slice | int] = [slice(None), slice(None), slice(None)]
    data_slice[1 if (axis % 3) == 0 else 0] = idx[0]
    data_slice[1 if (axis % 3) == 2 else 2] = idx[1]
    data = potential["points"][tuple(data_slice)]

    (line,) = ax.plot(coordinates, data)
    ax.set_xlabel(f"{(axis % 3)} axis")
    ax.set_ylabel("Energy /J")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_potential_1D_comparison(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    comparison_points: dict[str, tuple[tuple[int, int], int]],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    for label, (idx, axis) in comparison_points.items():
        (_, _, line) = plot_potential_1D(potential, idx, axis, ax=ax, scale=scale)
        line.set_label(label)
    ax.legend()
    return fig, ax


def plot_potential_2D(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    coordinates = get_projected_x_points(potential["basis"], z_axis)[
        slice_along_axis(idx, (z_axis % 3) + 1)
    ]
    data = potential["points"][slice_along_axis(idx, z_axis)]

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")

    return fig, ax, mesh


def plot_potential_x0x1(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    x3_idx: int,
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    return plot_potential_2D(potential, x3_idx, 2, ax=ax, scale=scale)


def plot_potential_x1x2(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    x0_idx: int,
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    return plot_potential_2D(potential, x0_idx, 0, ax=ax, scale=scale)


def plot_potential_x2x0(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    x1_idx: int,
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    return plot_potential_2D(potential, x1_idx, 1, ax=ax, scale=scale)


def plot_potential_difference_2D(
    potential0: Potential[_L0Inv, _L1Inv, _L2Inv],
    potential1: Potential[_L0Inv, _L1Inv, _L2Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    potential: Potential[_L0Inv, _L1Inv, _L2Inv] = {
        "basis": potential0["basis"],
        "points": potential0["points"] - potential1["points"],
    }
    return plot_potential_2D(potential, idx, z_axis, ax=ax, scale=scale)


def animate_potential_3D(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    coordinates = get_projected_x_points(potential["basis"], z_axis)
    data = potential["points"]

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
    min_clim: float = np.min([i[0].get_clim()[0] for i in frames])
    for (mesh,) in frames:
        mesh.set_norm(scale)
        mesh.set_clim(min_clim, max_clim)
    mesh0.set_norm(scale)
    mesh0.set_clim(min_clim, max_clim)

    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")

    return fig, ax, mesh


def animate_potential_x0x1(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    return animate_potential_3D(potential, 2, ax=ax, scale=scale)


def animate_potential_x1x2(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    return animate_potential_3D(potential, 0, ax=ax, scale=scale)


def animate_potential_x2x0(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    return animate_potential_3D(potential, 1, ax=ax, scale=scale)


def plot_potential_along_path(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    data = potential["points"][*path]
    util = PositionBasisConfigUtil(potential["basis"])
    distances = calculate_cumulative_distances_along_path(
        path, util.x_points.reshape(3, *util.shape)
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    return fig, ax, line


def get_minimum_path(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]],
    axis: Literal[0, 1, 2, -1, -2, -3] = -1,
) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]:
    """
    Find the minimum path, taking the smallest value along the given axis

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
    min_idx = np.argmin(potential["points"], axis=axis)
    min_idx_path = min_idx[*path]
    full_path = np.empty((3, path.shape[1]))
    full_path[1 if (axis % 3) == 0 else 0] = path[0]
    full_path[1 if (axis % 3) == 2 else 3] = path[1]
    full_path[axis] = min_idx_path
    return full_path  # type: ignore


def plot_potential_minimum_along_path(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]],
    axis: Literal[0, 1, 2, -1, -2, -3] = -1,
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    full_path = get_minimum_path(potential, path, axis)
    return plot_potential_along_path(potential, full_path, ax=ax, scale=scale)