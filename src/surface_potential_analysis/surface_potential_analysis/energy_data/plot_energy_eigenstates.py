from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .energy_eigenstates import EnergyEigenstates, WavepacketGrid, sort_wavepacket


def plot_eigenstate_positions(
    eigenstates: EnergyEigenstates, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    (line,) = ax1.plot(eigenstates["kx_points"], eigenstates["ky_points"])
    line.set_linestyle("None")
    line.set_marker("x")

    return fig, ax1, line


def plot_lowest_band_in_kx(eigenstates: EnergyEigenstates, ax: Axes | None = None):
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    kx_points = eigenstates["kx_points"]
    eigenvalues = eigenstates["eigenvalues"]

    (line,) = a.plot(kx_points, eigenvalues)
    return fig, a, line


def plot_wavepacket_grid_xy(
    grid: WavepacketGrid,
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


def plot_wavepacket_grid_xz(
    grid: WavepacketGrid,
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
    grid: WavepacketGrid,
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
