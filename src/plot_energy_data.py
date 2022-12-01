import math

import matplotlib.pyplot as plt
import numpy as np

from energy_data import EnergyData, add_back_symmetry_points


def plot_z_direction_energy_data(
    data: EnergyData, otherData: EnergyData | None = None
) -> None:
    fig, ax = plt.subplots()

    heights = data["z_points"]
    points = np.array(data["points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    max_potential = 1e-18

    top_energy = points[0, 0]
    bridge_energy = points[middle_x_index, 0]
    hollow_energy = points[middle_x_index, math.floor(points.shape[1] / 2)]

    ax.plot(heights, top_energy, label="Top Site")
    ax.plot(heights, bridge_energy, label="Bridge Site")
    ax.plot(heights, hollow_energy, label="Hollow Site")

    if otherData is not None:
        heights = otherData["z_points"]
        points = np.array(otherData["points"])
        middle_x_index = math.floor(points.shape[0] / 2)

        top_energy = points[0, 0]
        bridge_energy = points[middle_x_index, 0]
        hollow_energy = points[middle_x_index, math.floor(points.shape[1] / 2)]

        ax.plot(heights, top_energy, label="Top Site", ls="--")
        ax.plot(heights, bridge_energy, label="Bridge Site", ls="--")
        ax.plot(heights, hollow_energy, label="Hollow Site", ls="--")

    ax.set_title("Plot of energy at the Top and Hollow sites")
    ax.set_ylabel("Energy / J")
    ax.set_xlabel("relative z position /m")
    ax.set_ylim(bottom=0, top=max_potential)
    ax.legend()

    fig.tight_layout()
    fig.show()
    fig.savefig("temp.png")


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
