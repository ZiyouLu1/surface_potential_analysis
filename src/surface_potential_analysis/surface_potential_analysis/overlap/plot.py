from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from surface_potential_analysis.basis_config.basis_config import (
    MomentumBasisConfigUtil,
    get_fundamental_projected_k_points,
)
from surface_potential_analysis.eigenstate.plot import get_measured_data
from surface_potential_analysis.util import (
    calculate_cumulative_distances_along_path,
    slice_along_axis,
)

from .overlap import OverlapTransform


def plot_overlap_transform_2d(
    overlap: OverlapTransform,
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    # TODO: shifted transform
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    coordinates = get_fundamental_projected_k_points(overlap["basis"], z_axis)[
        slice_along_axis(idx, (z_axis % 3) + 1)
    ]
    util = MomentumBasisConfigUtil(overlap["basis"])
    points = overlap["vector"].reshape(*util.shape)[slice_along_axis(idx, z_axis)]
    data = get_measured_data(points, measure)

    mesh = ax.pcolormesh(*coordinates, data, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel(f"kx{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"kx{2 if (z_axis % 3) != 2 else 1} axis")

    return fig, ax, mesh


def plot_overlap_transform_along_path(
    overlap: OverlapTransform,
    path: np.ndarray[tuple[3, int], np.dtype[np.int_]],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = overlap["points"][*path]
    data = get_measured_data(points, measure)
    util = MomentumBasisConfigUtil(overlap["basis"])
    distances = calculate_cumulative_distances_along_path(
        path, util.fundamental_k_points.reshape(3, *util.shape)
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(line)
    ax.set_xlabel("Distance along path")
    return fig, ax, line


def plot_overlap_transform_along_diagonal(
    overlap: OverlapTransform,
    kz_ind: int = 0,
    *,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    util = MomentumBasisConfigUtil(overlap["basis"])
    path = np.array([[i, i, kz_ind] for i in range(util.shape[0])]).T

    return plot_overlap_transform_along_path(overlap, path, measure=measure, ax=ax)


def plot_overlap_transform_along_x0(
    overlap: OverlapTransform,
    kz_ind: int = 0,
    kx1_ind: int = 0,
    *,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    util = MomentumBasisConfigUtil(overlap["basis"])
    path = np.array([[i, kx1_ind, kz_ind] for i in range(util.shape[0])]).T

    return plot_overlap_transform_along_path(overlap, path, measure=measure, ax=ax)
