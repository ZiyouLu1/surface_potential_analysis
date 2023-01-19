import math
from typing import List, Sequence, Tuple

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from .energy_data import (
    EnergyGrid,
    EnergyPoints,
    add_back_symmetry_points,
    get_energy_points_xy_locations,
)


def plot_z_direction_energy_comparison(
    data: EnergyGrid, otherData: EnergyGrid, ax: Axes | None = None
) -> Tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    plot_z_direction_energy_data(data, ax=ax)
    plot_z_direction_energy_data(otherData, ax=ax, ls="--")

    return fig, ax


def plot_z_direction_energy_data(
    data: EnergyGrid,
    ax: Axes | None = None,
    ls=None,
) -> Tuple[Figure, Axes, Tuple[Line2D, Line2D, Line2D]]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    heights = data["z_points"]
    points = np.array(data["points"], dtype=float)
    middle_x_index = math.floor(points.shape[0] / 2)

    top_energy = points[0, 0]
    bridge_energy = points[middle_x_index, 0]
    hollow_energy = points[middle_x_index, math.floor(points.shape[1] / 2)]

    (l1,) = ax.plot(heights, top_energy, label="Top Site", ls=ls)
    (l2,) = ax.plot(heights, bridge_energy, label="Bridge Site", ls=ls)
    (l3,) = ax.plot(heights, hollow_energy, label="Hollow Site", ls=ls)

    ax.set_title("Plot of energy at the Top and Hollow sites")
    ax.set_ylabel("Energy / J")
    ax.set_xlabel("relative z position /m")

    ax.legend()

    return fig, ax, (l1, l2, l3)


def plot_x_direction_energy_data(data: EnergyGrid) -> None:
    fig, ax = plt.subplots()

    with_symmetry = add_back_symmetry_points(data)
    heights = with_symmetry["x_points"]
    points = np.array(with_symmetry["points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    top_equilibrium = np.argmin(points[0, 0])
    hollow_equilibrium = np.argmin(points[middle_x_index, middle_y_index])

    hollow_eq_energy = points[:, middle_y_index, hollow_equilibrium]
    top_eq_energy = points[:, middle_y_index, top_equilibrium]
    hollow_max_energy = points[:, middle_y_index, 0]

    ax.plot(heights, top_eq_energy, label="Near Top Equilibrium")
    ax.plot(heights, hollow_eq_energy, label="Near Hollow Equilibrium")
    ax.plot(heights, hollow_max_energy, label="Near Hollow Maximum")

    ax.set_title("Plot of energy in the x direction")
    ax.set_ylabel("Energy / eV")
    ax.set_xlabel("relative z position /m")
    ax.legend()

    fig.tight_layout()
    fig.show()
    fig.savefig("temp.png")


def plot_xz_plane_energy(data: EnergyGrid) -> Figure:
    fig, axs = plt.subplots(nrows=2, ncols=3)

    with_symmetry = add_back_symmetry_points(data)
    x_points = np.array(with_symmetry["x_points"])
    y_points = np.array(with_symmetry["y_points"])
    z_points = np.array(with_symmetry["z_points"])
    points = np.array(with_symmetry["points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    max_potential = 1e-18

    bridge_energies = np.clip(points[::, 0, ::-1].transpose(), 0, max_potential)
    hollow_energies = np.clip(
        points[::, middle_x_index, ::-1].transpose(), 0, max_potential
    )
    top_hollow_energies = np.clip(points.diagonal()[::-1], 0, max_potential)

    extent = [x_points[0], x_points[-1], z_points[0], z_points[-1]]
    axs[0][0].imshow(bridge_energies, extent=extent)
    axs[0][2].imshow(hollow_energies, extent=extent)
    extent = [
        np.sqrt(2) * x_points[0],
        np.sqrt(2) * x_points[-1],
        data["z_points"][0],
        data["z_points"][-1],
    ]
    axs[0][1].imshow(top_hollow_energies, extent=extent)

    extent = [x_points[0], x_points[-1], y_points[0], y_points[-1]]
    bottom_energies = np.clip(points[::, ::, 0], 0, max_potential)
    axs[1][0].imshow(bottom_energies, extent=extent)
    equilibrium_z = np.argmin(points[middle_x_index, middle_y_index])
    equilibrium_energies = np.clip(points[::, ::, equilibrium_z], 0, max_potential)
    axs[1][2].imshow(equilibrium_energies, extent=extent)

    axs[0][1].sharey(axs[0][0])
    axs[0][2].sharey(axs[0][0])
    axs[1][0].sharex(axs[0][0])
    axs[1][2].sharex(axs[0][2])

    axs[0][0].set_xlabel("x Position")
    axs[0][0].set_ylabel("z position /m")

    axs[0][0].set_title("Top-Bridge Site")
    axs[0][1].set_title("Top-Hollow Site")
    axs[0][2].set_title("Bridge-Hollow Site")
    axs[1][0].set_title("Bottom Energies")
    axs[1][2].set_title("Equilibrium Energies")

    fig.suptitle("Plot of energy through several planes perpendicular to xy")
    fig.tight_layout()
    return fig


def plot_energy_point_z(
    energy_points: EnergyPoints, xy: Tuple[float, float], ax: Axes | None = None
):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    idx = np.argwhere(
        np.logical_and(
            np.array(energy_points["x_points"]) == xy[0],
            np.array(energy_points["y_points"]) == xy[1],
        )
    )
    points = np.array(energy_points["points"])[idx]
    z_points = np.array(energy_points["z_points"])[idx]

    (line,) = ax.plot(z_points, points)
    ax.set_xlabel("z")
    ax.set_ylabel("Energy")
    return fig, ax, line


def plot_all_energy_points_z(energy_points: EnergyPoints, ax: Axes | None = None):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = get_energy_points_xy_locations(energy_points)
    for (x, y) in points:
        _, _, line = plot_energy_point_z(energy_points, (x, y), ax)
        line.set_label(f"{x:.2}, {y:.2}")

    ax.set_title("Plot of Energy against z for each (x,y) point")
    return fig, ax


def plot_energy_points_location(energy_points: EnergyPoints, ax: Axes | None = None):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = get_energy_points_xy_locations(energy_points)
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    (line,) = ax.plot(x_points, y_points)
    line.set_marker("x")
    line.set_linestyle("")

    return fig, ax, line


def get_energy_grid_frame(
    data: NDArray, ax: Axes, clim: Tuple[float, float], extent: Sequence[float]
) -> AxesImage:
    img = ax.imshow(data)
    img.set_extent(extent)
    img.set_clim(*clim)
    img.set_norm("symlog")  # type: ignore
    return img


def plot_energy_grid_3D_xy(data: EnergyGrid, ax: Axes | None = None):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(data["points"])
    extent = [
        data["x_points"][0],
        data["x_points"][-1],
        data["y_points"][0],
        data["y_points"][-1],
    ]
    clim = (np.min(points), np.max(points))
    get_energy_grid_frame(points[:, :, 0].T, ax, clim, extent)

    ims: List[List[AxesImage]] = []
    for z_ind in range(points.shape[2]):

        img = get_energy_grid_frame(points[:, :, z_ind].T, ax, clim, extent)
        ims.append([img])

    ani = matplotlib.animation.ArtistAnimation(fig, ims)

    ax.set_xlabel("X direction")
    ax.set_ylabel("Y direction")

    return fig, ax, ani


def plot_energy_grid_3D_xz(data: EnergyGrid, ax: Axes | None = None):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(data["points"])
    extent = [
        data["x_points"][0],
        data["x_points"][-1],
        data["z_points"][0],
        data["z_points"][-1],
    ]
    clim = (np.min(points), np.max(points))
    get_energy_grid_frame(points[:, 0, ::-1].T, ax, clim, extent)

    ims: List[List[AxesImage]] = []
    for y_ind in range(points.shape[1]):

        img = get_energy_grid_frame(points[:, y_ind, ::-1].T, ax, clim, extent)
        ims.append([img])

    ani = matplotlib.animation.ArtistAnimation(fig, ims)

    ax.set_xlabel("X direction")
    ax.set_ylabel("Z height above top layer")

    return fig, ax, ani


def compare_energy_grid_to_all_raw_points(
    raw_points: EnergyPoints, grid: EnergyGrid, ax: Axes | None = None
):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = get_energy_points_xy_locations(raw_points)
    x_points = np.array(grid["x_points"])
    y_points = np.array(grid["y_points"])
    grid_points = np.array(grid["points"])

    cols = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    for (i, (x, y)) in enumerate(points):
        _, _, line = plot_energy_point_z(raw_points, (x, y), ax)
        line.set_color(cols[i])
        line.set_linestyle("")
        line.set_marker("x")

        ix = (np.abs(x_points - x)).argmin()
        iy = (np.abs(y_points - y)).argmin()

        (line,) = ax.plot(grid["z_points"], grid_points[ix, iy, :])
        line.set_label(f"{x:.2}, {y:.2}")
        line.set_color(cols[i])

    return fig, ax


def plot_energy_point_locations_on_grid(
    raw_points: EnergyPoints, grid: EnergyGrid, ax: Axes | None = None
):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = get_energy_points_xy_locations(raw_points)
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    (line,) = ax.plot(x_points, y_points)
    line.set_marker("x")
    line.set_linestyle("")

    z_points = np.unique(raw_points["z_points"])

    grid_z_points = np.array(grid["z_points"])
    grid_points = np.array(grid["points"])
    extent = [
        grid["x_points"][0],
        grid["x_points"][-1],
        grid["y_points"][0],
        grid["y_points"][-1],
    ]
    clim = (np.min(grid_points), np.max(grid_points))

    ims: List[Tuple[AxesImage, Line2D]] = []

    for z in z_points:

        iz = (np.abs(grid_z_points - z)).argmin()

        img = get_energy_grid_frame(grid_points[:, :, iz].T, ax, clim, extent)
        ims.append((img, line))

    ani = matplotlib.animation.ArtistAnimation(fig, ims, interval=500)

    ax.set_xlabel("X line")
    ax.set_ylabel("Y line")

    return fig, ax, ani
