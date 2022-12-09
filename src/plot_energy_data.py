import math
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from energy_data import (
    EnergyData,
    EnergyEigenstates,
    EnergyInterpolation,
    add_back_symmetry_points,
)
from hamiltonian import SurfaceHamiltonian
from sho_config import SHOConfig


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


def plot_interpolation_with_sho(
    interpolation: EnergyInterpolation, sho_config: SHOConfig, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(interpolation["points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    start_z = sho_config["z_offset"]
    end_z = interpolation["dz"] * (points.shape[2] - 1) + sho_config["z_offset"]
    z_points = np.linspace(start_z, end_z, points.shape[2])

    a.plot(z_points, points[middle_x_index, middle_y_index])
    sho_pot = 0.5 * sho_config["mass"] * (sho_config["sho_omega"] * z_points) ** 2
    a.plot(z_points, sho_pot)

    max_potential = 1e-18
    a.set_ylim(0, max_potential)

    return fig, a


def plot_energy_eigenvalues(
    hamiltonian: SurfaceHamiltonian, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    for e in hamiltonian.eigenvalues(0, 0):
        a.plot([0, 1], [e, e])

    return fig, a


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_density_of_states(
    hamiltonian: SurfaceHamiltonian, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    eigenvalues = hamiltonian.eigenvalues(0, 0)
    de = eigenvalues[1:] - eigenvalues[0:-1]
    (line,) = a.plot(1 / moving_average(de))

    return fig, a, line


def plot_eigenvector_z(
    hamiltonian: SurfaceHamiltonian,
    eigenvector: Iterable[float],
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    z_points = np.linspace(hamiltonian.z_points[0], hamiltonian.z_points[-1], 1000)
    points = np.array(
        [(hamiltonian.delta_x / 2, hamiltonian.delta_y / 2, z) for z in z_points]
    )

    wfn = np.abs(hamiltonian.calculate_wavefunction(points, eigenvector))
    (line,) = a.plot(z_points, wfn)

    return fig, a, line


def plot_eigenvector_through_bridge(
    hamiltonian: SurfaceHamiltonian,
    eigenvector: Iterable[float],
    ax: Axes | None = None,
    view: Literal["abs"] | Literal["angle"] = "abs",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    x_points = np.linspace(hamiltonian.x_points[0], hamiltonian.x_points[-1], 1000)
    points = np.array([(x, hamiltonian.delta_y / 2, 0) for x in x_points])
    wfn = hamiltonian.calculate_wavefunction(points, eigenvector)
    ##TODO: wfn = wfn * exp(1j*np.angle(wfn[middle]))
    (line,) = ax1.plot(
        x_points - hamiltonian.delta_x / 2,
        np.abs(wfn) if view == "abs" else np.angle(wfn),
    )

    return fig, ax1, line


def plot_nth_eigenvector(
    hamiltonian: SurfaceHamiltonian, n=0, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    eigenvalue_index = np.argpartition(hamiltonian.eigenvalues(0, 0), n)[n]
    eigenvector = hamiltonian.eigenvectors(0, 0)[:, eigenvalue_index]

    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    _, _, line = plot_eigenvector_z(hamiltonian, eigenvector, ax)
    line.set_label("Z direction")

    _, _, line = plot_eigenvector_through_bridge(hamiltonian, eigenvector, ax)
    line.set_label("X-Y through bridge")

    x_points = np.linspace(hamiltonian.x_points[0], hamiltonian.x_points[-1], 1000)
    points = np.array([(x, x, 0) for x in x_points])
    a.plot(
        np.sqrt(2) * (x_points - hamiltonian.delta_x / 2),
        np.abs(hamiltonian.calculate_wavefunction(points, eigenvector)),
        label="X-Y through Top",
    )
    a.set_title(f"Plot of the n={n} wavefunction")
    a.legend()

    return (fig, a)


def plot_first_4_eigenvectors(hamiltonian: SurfaceHamiltonian) -> Figure:
    fig, axs = plt.subplots(2, 2)
    axes = [axs[0][0], axs[1][0], axs[0][1], axs[1][1]]
    for (n, ax) in enumerate(axes):
        plot_nth_eigenvector(hamiltonian, n, ax=ax)

    fig.tight_layout()

    return fig


def plot_lowest_band_in_kx(
    eigenstates: EnergyEigenstates, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    kx_points = eigenstates["kx_points"]
    eigenvalues = eigenstates["eigenvalues"]

    (line,) = a.plot(kx_points, eigenvalues)
    return fig, a, line


def plot_bands_occupation(
    hamiltonian: SurfaceHamiltonian, temperature: float = 50.0, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    eigenvalues = hamiltonian.eigenvalues(0, 0)
    normalized_eigenvalues = eigenvalues - np.min(hamiltonian.eigenvalues(0, 0))
    beta = 1 / (scipy.constants.Boltzmann * temperature)
    occupations = np.exp(-normalized_eigenvalues * beta)
    (line,) = a.plot(occupations / np.sum(occupations))

    return fig, a, line


def plot_eigenstate_positions(
    eigenstates: EnergyEigenstates, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    (line,) = ax1.plot(eigenstates["kx_points"], eigenstates["ky_points"])
    line.set_linestyle(None)
    line.set_marker("x")

    return fig, ax1, line
