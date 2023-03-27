from typing import Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from surface_potential_analysis.basis_config import (
    PositionBasisConfigUtil,
    get_projected_x_points,
)
from surface_potential_analysis.util import (
    calculate_cumulative_distances_along_path,
    slice_along_axis,
)

from .eigenstate import PositionBasisEigenstate

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


_SInv = TypeVar("_SInv", bound=tuple[Any])


def get_measured_data(
    data: np.ndarray[_SInv, np.dtype[np.complex_]],
    measure: Literal["real", "imag", "abs"],
) -> np.ndarray[_SInv, np.dtype[np.float_]]:
    match measure:
        case "real":
            return np.real(data)  # type: ignore
        case "imag":
            return np.imag(data)  # type: ignore
        case "abs":
            return np.abs(data)  # type: ignore


def plot_eigenstate_2D(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    coordinates = get_projected_x_points(eigenstate["basis"], z_axis)[
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
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")

    return fig, ax, mesh


def plot_eigenstate_x0x1(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    x3_idx: int,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    return plot_eigenstate_2D(
        eigenstate, x3_idx, 2, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_x1x2(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    x0_idx: int,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    return plot_eigenstate_2D(
        eigenstate, x0_idx, 0, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_x2x0(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    x1_idx: int,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    return plot_eigenstate_2D(
        eigenstate, x1_idx, 1, ax=ax, measure=measure, scale=scale
    )


def plot_eigenstate_difference_2D(
    eigenstate0: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    eigenstate1: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv] = {
        "basis": eigenstate0["basis"],
        "vector": eigenstate0["vector"] - eigenstate1["vector"],
    }
    return plot_eigenstate_2D(
        eigenstate, idx, z_axis, ax=ax, measure=measure, scale=scale
    )


def animate_eigenstate_3D(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    coordinates = get_projected_x_points(eigenstate["basis"], z_axis)
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


def animate_eigenstate_x0x1(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    return animate_eigenstate_3D(eigenstate, 2, ax=ax, measure=measure, scale=scale)


def animate_eigenstate_x1x2(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    return animate_eigenstate_3D(eigenstate, 0, ax=ax, measure=measure, scale=scale)


def animate_eigenstate_x2x0(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    return animate_eigenstate_3D(eigenstate, 1, ax=ax, measure=measure, scale=scale)


def plot_eigenstate_along_path(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = PositionBasisConfigUtil(eigenstate["basis"])
    points = eigenstate["vector"].reshape(*util.shape)[*path]
    data = get_measured_data(points, measure)
    distances = calculate_cumulative_distances_along_path(
        path, util.x_points.reshape(3, *util.shape)
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    return fig, ax, line
