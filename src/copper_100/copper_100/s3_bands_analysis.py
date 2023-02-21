import json
from pathlib import Path
from typing import List, TypedDict

import numpy as np
import scipy.constants
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from surface_potential_analysis.surface_hamiltonian_plot import (
    plot_bands_occupation,
    plot_first_4_eigenvectors,
)

from .s2_hamiltonian import generate_hamiltonian, generate_hamiltonian_relaxed
from .surface_data import get_data_path, save_figure


class CopperEigenvalues(TypedDict):
    center: List[float]
    k_max: List[float]


def save_copper_eigenvalues(data: CopperEigenvalues, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f)


def load_copper_eigenvalues(path: Path) -> CopperEigenvalues:
    with path.open("r") as f:
        return json.load(f)


def generate_copper_eigenvalues():

    # hamiltonian = generate_hamiltonian(resolution=(25, 25, 16))

    # eigenvalues_origin, _ = hamiltonian.calculate_eigenvalues(0, 0)
    # sorted_eigenvalues_origin = np.sort(eigenvalues_origin)

    # max_kx = (np.abs(hamiltonian.dkx0[0]) + np.abs(hamiltonian.dkx1[0])) / 2
    # max_ky = (np.abs(hamiltonian.dkx0[1]) + np.abs(hamiltonian.dkx1[1])) / 2
    # eigenvalues_k_max, _ = hamiltonian.calculate_eigenvalues(max_kx, max_ky)
    # sorted_eigenvalues_k_max = np.sort(eigenvalues_k_max)

    # values_not_relaxed: CopperEigenvalues = {
    #     "center": sorted_eigenvalues_origin.tolist(),
    #     "k_max": sorted_eigenvalues_k_max.tolist(),
    # }
    # path = get_data_path("copper_eigenvalues_not_relaxed.json")
    # save_copper_eigenvalues(values_not_relaxed, path)

    hamiltonian = generate_hamiltonian_relaxed(resolution=(21, 21, 15))

    eigenvalues_origin, _ = hamiltonian.calculate_eigenvalues(0, 0)
    sorted_eigenvalues_origin = np.sort(eigenvalues_origin)

    max_kx = (np.abs(hamiltonian.dkx0[0]) + np.abs(hamiltonian.dkx1[0])) / 2
    max_ky = (np.abs(hamiltonian.dkx0[1]) + np.abs(hamiltonian.dkx1[1])) / 2
    eigenvalues_k_max, _ = hamiltonian.calculate_eigenvalues(max_kx, max_ky)
    sorted_eigenvalues_k_max = np.sort(eigenvalues_k_max)

    values_relaxed: CopperEigenvalues = {
        "center": sorted_eigenvalues_origin.tolist(),
        "k_max": sorted_eigenvalues_k_max.tolist(),
    }
    path = get_data_path("copper_eigenvalues_relaxed.json")
    save_copper_eigenvalues(values_relaxed, path)


def plot_first_copper_bands():
    h = generate_hamiltonian_relaxed(resolution=(8, 8, 13))
    fig = plot_first_4_eigenvectors(h)
    save_figure(fig, "copper_first_4_bands.png")
    fig.show()


def plot_copper_bands_occupation():
    h = generate_hamiltonian_relaxed(resolution=(8, 8, 13))
    # Plot the eigenstate occupation. Need to think about there 'mu' is
    # i.e. we have more than one hydrogen adsorbed on the surface
    # And interaction between hydrogen would also ruin things
    fig, ax, _ = plot_bands_occupation(h, temperature=60)
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Occupation Probability")
    ax.set_title(
        "Plot of occupation probability of each band according to the Boltzmann distribution"
    )
    fig.show()
    save_figure(fig, "copper_bands_occupation.png")


def list_first_copper_band_with_widths():

    print("----------------------------------------")
    print("Not relaxed data")

    path = get_data_path("copper_eigenvalues_not_relaxed.json")
    eigenvalues = load_copper_eigenvalues(path)

    print("k=(0,0)")
    print(np.subtract(eigenvalues["center"], eigenvalues["center"][0])[:5])
    print("k=(max, max)")
    print(np.subtract(eigenvalues["k_max"], eigenvalues["center"][0])[:5])
    print("bandwidths")
    print(np.subtract(eigenvalues["k_max"], eigenvalues["center"])[:5])

    print("----------------------------------------")
    print("Relaxed data")

    path = get_data_path("copper_eigenvalues_relaxed.json")
    eigenvalues = load_copper_eigenvalues(path)

    print("k=(0,0)")
    print(np.subtract(eigenvalues["center"], eigenvalues["center"][0])[:5])
    print("k=(max, max)")
    print(np.subtract(eigenvalues["k_max"], eigenvalues["center"][0])[:5])
    print("bandwidths")
    print(np.subtract(eigenvalues["k_max"], eigenvalues["center"])[:5])

    print("----------------------------------------")


def find_band_with_1mev_bandwidth():
    """
    Activated tunnelling has an energy of 197meV

    John: The key thing is not the physical barrier,
    but the energy wrt the ground state of the first band with a decent (eg 1meV) bandwidth
    """

    print("----------------------------------------")
    print("Relaxed data")

    path = get_data_path("copper_eigenvalues_not_relaxed.json")
    eigenvalues = load_copper_eigenvalues(path)
    bandwidths = np.subtract(eigenvalues["k_max"], eigenvalues["center"])
    first_relevant = int(
        np.argmax(bandwidths > 1 * 10**-3 * scipy.constants.elementary_charge)
    )

    print("band index", first_relevant)
    print("band width", bandwidths[first_relevant])
    print("k=0", eigenvalues["center"][first_relevant] - eigenvalues["center"][0])
    print("bandwidths", bandwidths[: first_relevant + 3])
    energies = np.subtract(eigenvalues["center"], eigenvalues["center"][0])
    print("energies", energies[: first_relevant + 3])

    print("----------------------------------------")


def find_band_with_relevant_energy():
    """
    Activated tunnelling has an energy of 197meV - which band would this correspond to?
    """

    print("----------------------------------------")
    print("Relaxed data")

    hamiltonian = generate_hamiltonian_relaxed(resolution=(10, 10, 14))

    eigenvalues_origin, _ = hamiltonian.calculate_eigenvalues(0, 0)
    eigenvalues_origin = np.sort(eigenvalues_origin)

    max_kx = (np.abs(hamiltonian.dkx0[0]) + np.abs(hamiltonian.dkx1[0])) / 2
    max_ky = (np.abs(hamiltonian.dkx0[1]) + np.abs(hamiltonian.dkx1[1])) / 2
    eigenvalues_max, _ = hamiltonian.calculate_eigenvalues(max_kx, max_ky)
    eigenvalues_max = np.sort(eigenvalues_max)

    bandwidths = np.abs(eigenvalues_origin - eigenvalues_max)
    first_relevant = int(
        np.argmax(bandwidths > 180 * 10**-3 * scipy.constants.elementary_charge)
    )
    last_relevant = int(
        np.argmax(bandwidths > 200 * 10**-3 * scipy.constants.elementary_charge)
    )

    print("band index", first_relevant, last_relevant)
    print("band width", bandwidths[first_relevant : last_relevant + 1])
    print(
        "k=0",
        eigenvalues_origin[first_relevant : last_relevant + 1] - eigenvalues_origin[0],
    )
    print(bandwidths[first_relevant : last_relevant + 1])

    print("----------------------------------------")


def calculate_bandwidths(eigenvalues: CopperEigenvalues) -> List[float]:
    return np.abs(np.subtract(eigenvalues["k_max"], eigenvalues["center"]))


def calculate_tst_rate_contributions(
    temperature: float, eigenvalues: CopperEigenvalues
) -> NDArray:
    bandwidths = calculate_bandwidths(eigenvalues)
    frequencies = bandwidths / (scipy.constants.hbar * np.pi)
    energies = np.array(eigenvalues["center"])
    tunnelling_contributions = np.exp(
        -energies / (scipy.constants.Boltzmann * temperature)
    )
    normalization = np.sum(tunnelling_contributions)
    return (frequencies * tunnelling_contributions) / normalization


def calculate_tst_rate(temperature: float, eigenvalues: CopperEigenvalues) -> float:
    return np.sum(calculate_tst_rate_contributions(temperature, eigenvalues))


def plot_tst_rate_arrhenius():
    path = get_data_path("copper_eigenvalues_not_relaxed.json")
    eigenvalues = load_copper_eigenvalues(path)

    temperatures = np.linspace(60, 150, 20)
    rates = [calculate_tst_rate(t, eigenvalues) for t in temperatures]

    fig, ax = plt.subplots()
    log_rate = np.log(rates)
    inverse_temp = 1000 / temperatures
    ax.plot(inverse_temp, log_rate)
    fig.show()
    ax.set_xlabel(r"$\frac{1000}{T}$")
    ax.set_ylabel(r"Hop rate $s^{-1}$")
    ax.set_yscale("log")

    rate_120k = calculate_tst_rate(120, eigenvalues)
    rate_190k = calculate_tst_rate(190, eigenvalues)
    # Ln(R) = Ln(R0) - E / KT
    # E / K * (1/T1 - 1/T2) =
    difference_rate = np.log(rate_120k) - np.log(rate_190k)
    difference_t = ((1 / 120) - (1 / 190)) / scipy.constants.Boltzmann
    energy = difference_rate / difference_t
    print(energy)

    print(np.argmax(calculate_tst_rate_contributions(130, eigenvalues)))
    input()


def plot_first_4_eigenstates():
    hamiltonian = generate_hamiltonian_relaxed(resolution=(8, 8, 13))

    fig = plot_first_4_eigenvectors(hamiltonian)
    fig.show()
    input()
