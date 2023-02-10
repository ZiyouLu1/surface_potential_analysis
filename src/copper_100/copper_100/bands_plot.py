import numpy as np
import scipy.constants

from surface_potential_analysis.surface_hamiltonian_plot import (
    plot_bands_occupation,
    plot_first_4_eigenvectors,
)

from .s2_hamiltonian import generate_hamiltonian, generate_hamiltonian_relaxed
from .surface_data import save_figure


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

    hamiltonian = generate_hamiltonian(resolution=(12, 12, 15))

    eigenvalues_origin, _ = hamiltonian.calculate_eigenvalues(0, 0)
    sorted_eigenvalues_origin = np.sort(eigenvalues_origin)

    print("k=(0,0)")
    print(sorted_eigenvalues_origin[:5])
    print((sorted_eigenvalues_origin - sorted_eigenvalues_origin[0])[:5])

    max_kx = np.abs(hamiltonian.dkx0[0]) + np.abs(hamiltonian.dkx1[0])
    max_ky = np.abs(hamiltonian.dkx0[1]) + np.abs(hamiltonian.dkx1[1])
    eigenvalues_max, _ = hamiltonian.calculate_eigenvalues(max_kx, max_ky)
    sorted_eigenvalues_max = np.sort(eigenvalues_max)

    print("k=k_max")
    print(sorted_eigenvalues_max[:5])
    print((sorted_eigenvalues_max - sorted_eigenvalues_max[0])[:5])

    print("band width")
    print((sorted_eigenvalues_origin - sorted_eigenvalues_max)[:5])

    print("----------------------------------------")
    print("Relaxed data")

    hamiltonian = generate_hamiltonian_relaxed(resolution=(10, 10, 14))

    eigenvalues_origin, _ = hamiltonian.calculate_eigenvalues(0, 0)
    sorted_eigenvalues_origin = np.sort(eigenvalues_origin)

    print("Not relaxed k=(0,0)")
    print(sorted_eigenvalues_origin[:5])
    print((sorted_eigenvalues_origin - sorted_eigenvalues_origin[0])[:5])

    max_kx = np.abs(hamiltonian.dkx0[0]) + np.abs(hamiltonian.dkx1[0])
    max_ky = np.abs(hamiltonian.dkx0[1]) + np.abs(hamiltonian.dkx1[1])
    eigenvalues_max, _ = hamiltonian.calculate_eigenvalues(max_kx, max_ky)
    sorted_eigenvalues_max = np.sort(eigenvalues_max)

    print("k=k_max")
    print(sorted_eigenvalues_max[:5])
    print((sorted_eigenvalues_max - sorted_eigenvalues_max[0])[:5])

    print("band width")
    print((sorted_eigenvalues_origin - sorted_eigenvalues_max)[:5])

    print("----------------------------------------")


def find_band_with_1mev_bandwidth():
    """
    Activated tunnelling has an energy of 197meV

    John: The key thing is not the physical barrier,
    but the energy wrt the ground state of the first band with a decent (eg 1meV) bandwidth
    """

    print("----------------------------------------")
    print("Relaxed data")

    hamiltonian = generate_hamiltonian_relaxed(resolution=(10, 10, 14))

    eigenvalues_origin, _ = hamiltonian.calculate_eigenvalues(0, 0)
    eigenvalues_origin = np.sort(eigenvalues_origin)

    max_kx = np.abs(hamiltonian.dkx0[0]) + np.abs(hamiltonian.dkx1[0])
    max_ky = np.abs(hamiltonian.dkx0[1]) + np.abs(hamiltonian.dkx1[1])
    eigenvalues_max, _ = hamiltonian.calculate_eigenvalues(max_kx, max_ky)
    eigenvalues_max = np.sort(eigenvalues_max)

    bandwidths = np.abs(eigenvalues_origin - eigenvalues_max)
    print(bandwidths, 1 * 10**-3 * scipy.constants.elementary_charge)
    first_relevant = np.argmax(
        bandwidths > 1 * 10**-3 * scipy.constants.elementary_charge
    )

    print("band index", first_relevant)
    print("band width", bandwidths[first_relevant])
    print("k=0", eigenvalues_origin[first_relevant] - eigenvalues_origin[0])
    print(bandwidths[: first_relevant + 1])

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

    max_kx = np.abs(hamiltonian.dkx0[0]) + np.abs(hamiltonian.dkx1[0])
    max_ky = np.abs(hamiltonian.dkx0[1]) + np.abs(hamiltonian.dkx1[1])
    eigenvalues_max, _ = hamiltonian.calculate_eigenvalues(max_kx, max_ky)
    eigenvalues_max = np.sort(eigenvalues_max)

    bandwidths = np.abs(eigenvalues_origin - eigenvalues_max)
    first_relevant = np.argmax(
        bandwidths > 180 * 10**-3 * scipy.constants.elementary_charge
    )
    last_relevant = np.argmax(
        bandwidths > 200 * 10**-3 * scipy.constants.elementary_charge
    )

    print("band index", first_relevant, last_relevant)
    print("band width", bandwidths[first_relevant : last_relevant + 1])
    print(
        "k=0",
        eigenvalues_origin[first_relevant : last_relevant + 1] - eigenvalues_origin[0],
    )
    print(bandwidths[first_relevant : last_relevant + 1])

    print("----------------------------------------")


def plot_first_4_eigenstates():
    hamiltonian = generate_hamiltonian_relaxed(resolution=(8, 8, 13))

    fig = plot_first_4_eigenvectors(hamiltonian)
    fig.show()
    input()
