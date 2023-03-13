from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from surface_potential_analysis.energy_data_plot import (
    calculate_cumulative_distances_along_path,
)
from surface_potential_analysis.overlap_transform import OverlapTransform
from surface_potential_analysis.surface_config import (
    SurfaceConfig,
    get_surface_xy_points,
)
from surface_potential_analysis.surface_config_plot import (
    plot_ft_points_on_surface_xy,
    plot_points_on_surface_x0z,
    plot_points_on_surface_xy,
)


def plot_overlap_transform_xy(
    overlap: OverlapTransform,
    ikz=0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
) -> tuple[Figure, Axes, QuadMesh]:
    reciprocal_surface: SurfaceConfig = {
        "delta_x0": (
            overlap["dkx0"][0] * overlap["points"].shape[0],
            overlap["dkx0"][1] * overlap["points"].shape[0],
        ),
        "delta_x1": (
            overlap["dkx1"][0] * overlap["points"].shape[1],
            overlap["dkx1"][1] * overlap["points"].shape[1],
        ),
    }

    fig, ax, mesh = plot_points_on_surface_xy(
        reciprocal_surface,
        np.fft.fftshift(overlap["points"], axes=(0, 1)).tolist(),
        z_ind=ikz,
        ax=ax,
        measure=measure,
    )
    ax.set_xlabel("kx direction")
    ax.set_ylabel("ky direction")
    mesh.set_norm(norm)  # type: ignore
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    return fig, ax, mesh


def plot_overlap_xy(
    overlap: OverlapTransform,
    ikz=0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
) -> tuple[Figure, Axes, QuadMesh]:
    reciprocal_surface: SurfaceConfig = {
        "delta_x0": (
            overlap["dkx0"][0] * overlap["points"].shape[0],
            overlap["dkx0"][1] * overlap["points"].shape[0],
        ),
        "delta_x1": (
            overlap["dkx1"][0] * overlap["points"].shape[1],
            overlap["dkx1"][1] * overlap["points"].shape[1],
        ),
    }

    fig, ax, mesh = plot_ft_points_on_surface_xy(
        reciprocal_surface,
        overlap["points"].tolist(),
        z_ind=ikz,
        ax=ax,
        measure=measure,
    )
    ax.set_xlabel("x direction")
    ax.set_ylabel("y direction")
    mesh.set_norm(norm)  # type: ignore
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    return fig, ax, mesh


def plot_overlap_transform_x0z(
    overlap: OverlapTransform,
    x1_ind: int = 0,
    *,
    measure: Literal["real", "imag", "abs"] = "abs",
    ax: Axes | None = None,
    norm: Literal["symlog", "linear"] = "symlog",
) -> tuple[Figure, Axes, QuadMesh]:
    reciprocal_surface: SurfaceConfig = {
        "delta_x0": (
            overlap["dkx0"][0] * overlap["points"].shape[0],
            overlap["dkx0"][1] * overlap["points"].shape[0],
        ),
        "delta_x1": (
            overlap["dkx1"][0] * overlap["points"].shape[1],
            overlap["dkx1"][1] * overlap["points"].shape[1],
        ),
    }
    z_points = overlap["dkz"] * np.arange(overlap["points"].shape[2])

    fig, ax, mesh = plot_points_on_surface_x0z(
        reciprocal_surface,
        np.fft.fftshift(overlap["points"], axes=(0,)).tolist(),
        z_points.tolist(),
        x1_ind=x1_ind,
        ax=ax,
        measure=measure,
    )

    ax.set_xlabel("kx0 direction")
    ax.set_ylabel("kz direction")
    mesh.set_norm(norm)  # type: ignore
    # ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    return fig, ax, mesh


def plot_overlap_transform_along_path(
    overlap: OverlapTransform,
    path: NDArray,
    *,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    overlap_shape = np.shape(overlap["points"])
    points = overlap["points"][*path]
    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    elif measure == "abs":
        data = np.abs(points)
    else:
        data = np.unwrap(np.angle(points))

    kxy_points = get_surface_xy_points(
        {
            "delta_x0": (
                overlap["dkx0"][0] * overlap_shape[0],
                overlap["dkx0"][1] * overlap_shape[0],
            ),
            "delta_x1": (
                overlap["dkx1"][0] * overlap_shape[1],
                overlap["dkx1"][1] * overlap_shape[1],
            ),
        },
        shape=(overlap_shape[0], overlap_shape[1]),
    )

    distances = calculate_cumulative_distances_along_path(path[0:2], kxy_points)
    (line,) = ax.plot(distances, data)
    return fig, ax, line


def plot_overlap_transform_along_diagonal(
    overlap: OverlapTransform,
    kz_ind: int = 0,
    *,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:

    path = np.array([[i, i, kz_ind] for i in range(overlap["points"].shape[0])])
    new_overlap: OverlapTransform = {
        "points": np.fft.fftshift(overlap["points"], axes=(0, 1)),
        "dkx0": overlap["dkx0"],
        "dkx1": overlap["dkx1"],
        "dkz": overlap["dkz"],
    }
    return plot_overlap_transform_along_path(
        new_overlap, np.moveaxis(path, -1, 0), measure=measure, ax=ax
    )


def plot_overlap_transform_along_x0(
    overlap: OverlapTransform,
    kz_ind: int = 0,
    kx1_ind: int = 0,
    *,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:

    path = np.array([[i, kx1_ind, kz_ind] for i in range(overlap["points"].shape[0])])
    new_overlap: OverlapTransform = {
        "points": np.fft.fftshift(overlap["points"], axes=(0)),
        "dkx0": overlap["dkx0"],
        "dkx1": overlap["dkx1"],
        "dkz": overlap["dkz"],
    }
    return plot_overlap_transform_along_path(
        new_overlap, np.moveaxis(path, -1, 0), measure=measure, ax=ax
    )
