import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from energy_data import EnergyData, EnergyInterpolation, add_back_symmetry_points
from hamiltonian import SurfaceHamiltonian
from sho_config import SHOConfig


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


def plot_interpolation_with_sho(
    interpolation: EnergyInterpolation, sho_config: SHOConfig
):
    fig, ax = plt.subplots()

    points = np.array(interpolation["points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    start_z = sho_config["z_offset"]
    end_z = interpolation["dz"] * (points.shape[2] - 1) + sho_config["z_offset"]
    z_points = np.linspace(start_z, end_z, points.shape[2])

    ax.plot(z_points, points[middle_x_index, middle_y_index])
    sho_pot = 0.5 * sho_config["mass"] * (sho_config["sho_omega"] * z_points) ** 2
    ax.plot(z_points, sho_pot)

    max_potential = 1e-18
    ax.set_ylim(0, max_potential)

    fig.tight_layout()
    fig.show()
    fig.savefig("temp.png")


def plot_energy_eigenvalues(hamiltonian: SurfaceHamiltonian):
    fig, ax = plt.subplots()
    for e in hamiltonian.eigenvalues:
        ax.plot([0, 1], [e, e])

    fig.tight_layout()
    fig.show()
    fig.savefig("temp.png")


def plot_ground_state(hamiltonian: SurfaceHamiltonian):
    amin = np.argmin(hamiltonian.eigenvalues)
    eigenvector = hamiltonian.eigenvectors[:, amin]
    fig, ax = plt.subplots()

    z_points = np.linspace(hamiltonian.z_points[0], hamiltonian.z_points[-1], 1000)
    points = np.array(
        [(hamiltonian.delta_x / 2, hamiltonian.delta_y / 2, z) for z in z_points]
    )
    wfn = np.abs(hamiltonian.calculate_wavefunction(points, eigenvector))
    z_max = z_points[np.argmax(wfn)]
    ax.plot(z_points - z_max, wfn, label="Z direction")

    x_points = np.linspace(hamiltonian.x_points[0], hamiltonian.x_points[-1], 1000)
    points = np.array([(x, hamiltonian.delta_y / 2, z_max) for x in x_points])
    ax.plot(
        x_points - hamiltonian.delta_x / 2,
        np.abs(hamiltonian.calculate_wavefunction(points, eigenvector)),
        label="X-Y direction through bridge",
    )

    points = np.array([(x, x, z_max) for x in x_points])
    ax.plot(
        np.sqrt(2) * (x_points - hamiltonian.delta_x / 2),
        np.abs(hamiltonian.calculate_wavefunction(points, eigenvector)),
        label="X-Y direction through Top",
    )
    ax.set_title("Plot of the ground state wavefunction about the probability maximum")
    ax.legend()

    fig.tight_layout()
    fig.show()
    fig.savefig("temp.png")
