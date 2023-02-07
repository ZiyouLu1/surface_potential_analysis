from typing import List, Literal, Tuple

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from .energy_eigenstate import (
    EigenstateConfigUtil,
    EnergyEigenstates,
    get_eigenstate_list,
)
from .wavepacket_grid import WavepacketGrid, get_wavepacket_grid_xy_points


def plot_wavepacket_grid_xy(
    grid: WavepacketGrid,
    z_ind=0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
) -> Tuple[Figure, Axes, QuadMesh]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    if measure == "real":
        data = np.real(grid["points"])
    elif measure == "imag":
        data = np.imag(grid["points"])
    else:
        data = np.abs(grid["points"])

    coordinates = get_wavepacket_grid_xy_points(grid).reshape(
        data.shape[0], data.shape[1], 2
    )
    mesh = ax.pcolormesh(
        coordinates[:, :, 0], coordinates[:, :, 1], data[:, :, z_ind], shading="nearest"
    )
    return fig, ax, mesh


def animate_wavepacket_grid_3D_in_xy(
    grid: WavepacketGrid,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
) -> Tuple[Figure, Axes, matplotlib.animation.ArtistAnimation]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    _, _, mesh0 = plot_wavepacket_grid_xy(grid, 0, ax=ax, measure=measure)

    frames: List[List[QuadMesh]] = []
    for z_ind in range(np.array(grid["points"]).shape[2]):

        _, _, mesh = plot_wavepacket_grid_xy(grid, z_ind, ax=ax, measure=measure)
        frames.append([mesh])

    max_clim = np.max([i[0].get_clim()[1] for i in frames])
    for (mesh,) in frames:
        mesh.set_clim(0, max_clim)
        mesh.set_norm(norm)  # type: ignore
    mesh0.set_clim(0, max_clim)
    mesh0.set_norm(norm)  # type: ignore
    ani = matplotlib.animation.ArtistAnimation(fig, frames)

    ax.set_xlabel("X direction")
    ax.set_ylabel("Y direction")

    fig.colorbar(mesh0, ax=ax, format="%4.1e")

    return (fig, ax, ani)


def plot_wavepacket_grid_in_x1z(
    grid: WavepacketGrid,
    x2_ind: int,
    *,
    measure: Literal["real", "imag", "abs"] = "abs",
    ax: Axes | None = None,
) -> Tuple[Figure, Axes, QuadMesh]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    if measure == "real":
        data = np.real(grid["points"])
    elif measure == "imag":
        data = np.imag(grid["points"])
    else:
        data = np.abs(grid["points"])

    x1_points = np.linspace(0, np.linalg.norm(grid["delta_x1"]), data.shape[0])
    z_points = np.linspace(0, grid["delta_z"], data.shape[2])
    x1v, zv = np.meshgrid(x1_points, z_points, indexing="ij")
    mesh = ax.pcolormesh(x1v, zv, data[:, x2_ind, :], shading="nearest")
    return (fig, ax, mesh)


def animate_wavepacket_grid_3D_in_x1z(
    grid: WavepacketGrid,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
) -> Tuple[Figure, Axes, matplotlib.animation.ArtistAnimation]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    _, _, mesh0 = plot_wavepacket_grid_in_x1z(grid, 0, ax=ax, measure=measure)

    frames: List[List[QuadMesh]] = []
    for x2_ind in range(np.array(grid["points"]).shape[1]):

        _, _, mesh = plot_wavepacket_grid_in_x1z(grid, x2_ind, ax=ax, measure=measure)
        frames.append([mesh])

    max_clim = np.max([i[0].get_clim()[1] for i in frames])
    for (mesh,) in frames:
        mesh.set_clim(0, max_clim)
        mesh.set_norm(norm)  # type: ignore
    mesh0.set_clim(0, max_clim)
    mesh0.set_norm(norm)  # type: ignore
    ani = matplotlib.animation.ArtistAnimation(fig, frames)

    ax.set_xlabel("X direction")
    ax.set_ylabel("Y direction")

    fig.colorbar(mesh0, ax=ax, format="%4.1e")

    return (fig, ax, ani)


def plot_wavepacket_grid_x1(
    grid: WavepacketGrid,
    x2_ind=0,
    z_ind=0,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    points = np.array(grid["points"])[:, x2_ind, z_ind]

    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    else:
        data = np.abs(points)

    x1_points = np.linspace(0, np.linalg.norm(grid["delta_x1"]), data.shape[0])
    (line,) = ax.plot(x1_points, data)
    return fig, ax, line


def plot_wavepacket_in_xy(
    eigenstates: EnergyEigenstates,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, AxesImage]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])

    x_points = np.linspace(-util.delta_x1[0], util.delta_x1[0], 60)
    y_points = np.linspace(0, util.delta_x2[1], 30)

    xv, yv = np.meshgrid(x_points, y_points)
    points = np.array([xv.ravel(), yv.ravel(), np.zeros_like(xv.ravel())]).T

    X = np.zeros_like(xv, dtype=complex)
    for eigenstate in get_eigenstate_list(eigenstates):
        print("i")
        wfn = util.calculate_wavefunction_fast(eigenstate, points)
        X += (wfn).reshape(xv.shape)
    im = ax1.imshow(np.abs(X / len(eigenstates["eigenvectors"])))
    im.set_extent((x_points[0], x_points[-1], y_points[0], y_points[-1]))
    return (fig, ax1, im)
