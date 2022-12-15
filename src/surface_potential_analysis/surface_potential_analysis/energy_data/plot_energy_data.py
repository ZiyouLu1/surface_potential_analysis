import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .energy_data import EnergyData, add_back_symmetry_points


def plot_z_direction_energy_comparison(
    data: EnergyData, otherData: EnergyData, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    plot_z_direction_energy_data(data, ax=a)
    plot_z_direction_energy_data(otherData, ax=a, ls="--")

    return fig, a


def plot_z_direction_energy_data(
    data: EnergyData,
    ax: Axes | None = None,
    ls=None,
) -> tuple[Figure, Axes]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    heights = data["z_points"]
    points = np.array(data["points"], dtype=float)
    middle_x_index = math.floor(points.shape[0] / 2)

    top_energy = points[0, 0]
    bridge_energy = points[middle_x_index, 0]
    hollow_energy = points[middle_x_index, math.floor(points.shape[1] / 2)]

    a.plot(heights, top_energy, label="Top Site", ls=ls)
    a.plot(heights, bridge_energy, label="Bridge Site", ls=ls)
    a.plot(heights, hollow_energy, label="Hollow Site", ls=ls)

    a.set_title("Plot of energy at the Top and Hollow sites")
    a.set_ylabel("Energy / J")
    a.set_xlabel("relative z position /m")

    a.legend()

    return fig, a


def plot_x_direction_energy_data(data: EnergyData) -> None:
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


def plot_xz_plane_energy(data: EnergyData) -> None:
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
    fig.show()
    fig.savefig("temp.png")
