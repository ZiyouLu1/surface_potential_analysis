import numpy as np

from surface_potential_analysis.surface_hamiltonian_plot import (
    plot_bands_occupation,
    plot_first_4_eigenvectors,
)

from .hamiltonian import generate_hamiltonian, generate_hamiltonian_relaxed
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

    max_kx = np.abs(hamiltonian.dkx1[0]) + np.abs(hamiltonian.dkx2[0])
    max_ky = np.abs(hamiltonian.dkx1[1]) + np.abs(hamiltonian.dkx2[1])
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

    max_kx = np.abs(hamiltonian.dkx1[0]) + np.abs(hamiltonian.dkx2[0])
    max_ky = np.abs(hamiltonian.dkx1[1]) + np.abs(hamiltonian.dkx2[1])
    eigenvalues_max, _ = hamiltonian.calculate_eigenvalues(max_kx, max_ky)
    sorted_eigenvalues_max = np.sort(eigenvalues_max)

    print("k=k_max")
    print(sorted_eigenvalues_max[:5])
    print((sorted_eigenvalues_max - sorted_eigenvalues_max[0])[:5])

    print("band width")
    print((sorted_eigenvalues_origin - sorted_eigenvalues_max)[:5])

    print("----------------------------------------")


def plot_first_4_eigenstates():
    hamiltonian = generate_hamiltonian_relaxed(resolution=(8, 8, 13))

    fig = plot_first_4_eigenvectors(hamiltonian)
    fig.show()
    input()
