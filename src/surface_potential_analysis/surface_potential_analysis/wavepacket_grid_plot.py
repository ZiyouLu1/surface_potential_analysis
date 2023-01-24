from typing import List, Literal

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from .energy_eigenstate import (
    EigenstateConfigUtil,
    EnergyEigenstates,
    get_eigenstate_list,
)
from .wavepacket_grid import WavepacketGridLegacy, sort_wavepacket


def plot_wavepacket_grid_xy(
    grid: WavepacketGridLegacy,
    z_ind=0,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
):
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    sorted_grid = sort_wavepacket(grid)
    points = np.array(sorted_grid["points"])[:, :, z_ind]

    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    else:
        data = np.abs(points)

    img = a.imshow(data, origin="lower")
    img.set_extent(
        [
            sorted_grid["x_points"][0],
            sorted_grid["x_points"][-1],
            sorted_grid["y_points"][0],
            sorted_grid["y_points"][-1],
        ]
    )
    return fig, a, img


def plot_wavepacket_grid_z_2D(
    grid: WavepacketGridLegacy,
    ax: Axes | None = None,
    *,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog"
):
    fig, axs = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(grid["points"])
    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    else:
        data = np.abs(points)

    img = axs.imshow(data[:, :, 0])
    img.set_extent(
        [
            grid["x_points"][0],
            grid["x_points"][-1],
            grid["y_points"][0],
            grid["y_points"][-1],
        ]
    )
    img.set_clim(np.min(data), np.max(data))
    img.set_norm(norm)  # type: ignore
    ims: List[List[AxesImage]] = []
    for z_ind in range(points.shape[2]):

        img = axs.imshow(data[:, :, z_ind], animated=True)
        img.set_extent(
            [
                grid["x_points"][0],
                grid["x_points"][-1],
                grid["y_points"][0],
                grid["y_points"][-1],
            ]
        )
        img.set_clim(np.min(data), np.max(data))
        img.set_norm(norm)  # type: ignore
        ims.append([img])

    ani = matplotlib.animation.ArtistAnimation(fig, ims)
    return (fig, ax, ani)


def plot_wavepacket_grid_y_2D(
    grid: WavepacketGridLegacy,
    ax: Axes | None = None,
    *,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal[
        "linear",
        "log",
        "symlog",
    ] = "symlog"
):
    fig, axs = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(grid["points"])
    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    else:
        data = np.abs(points)

    ims: List[List[AxesImage]] = []

    img = axs.imshow(data[:, 0, ::-1].T)
    img.set_clim(np.min(data), np.max(data))
    img.set_norm("symlog")  # type: ignore
    img.set_extent(
        [
            grid["x_points"][0],
            grid["x_points"][-1],
            grid["z_points"][0],
            grid["z_points"][-1],
        ]
    )
    for y_ind in range(points.shape[1]):
        img = axs.imshow(data[:, y_ind, ::-1].T, animated=True)
        img.set_extent(
            [
                grid["x_points"][0],
                grid["x_points"][-1],
                grid["z_points"][0],
                grid["z_points"][-1],
            ]
        )
        img.set_norm("symlog")  # type: ignore
        img.set_clim(np.min(data), np.max(data))
        ims.append([img])

    ani = matplotlib.animation.ArtistAnimation(fig, ims, interval=80, repeat_delay=0)
    return (fig, ax, ani)


def plot_wavepacket_grid_xz(
    grid: WavepacketGridLegacy,
    y_ind=0,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
):
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    points = np.array(grid["points"])[:, y_ind, :]

    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    else:
        data = np.abs(points)

    img = a.imshow(data)
    img.set_extent(
        [
            grid["x_points"][0],
            grid["x_points"][-1],
            grid["z_points"][0],
            grid["z_points"][-1],
        ]
    )
    return fig, a, img


def plot_wavepacket_grid_x(
    grid: WavepacketGridLegacy,
    y_ind=0,
    z_ind=0,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
):
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    points = np.array(grid["points"])[:, y_ind, z_ind]

    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    else:
        data = np.abs(points)

    (line,) = a.plot(grid["x_points"], data)
    return fig, a, line


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
