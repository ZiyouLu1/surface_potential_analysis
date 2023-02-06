import numpy as np

from surface_potential_analysis.surface_hamiltonian_plot import (
    plot_bands_occupation,
    plot_first_4_eigenvectors,
)

from .hamiltonian import generate_hamiltonian_relaxed
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


def list_first_copper_band_energies():
    hamiltonian = generate_hamiltonian_relaxed(resolution=(8, 8, 13))

    eigenvalues, _ = hamiltonian.calculate_eigenvalues(0, 0)
    sorted_eigenvalues = np.sort(eigenvalues)
    print(sorted_eigenvalues[:5])
    print((sorted_eigenvalues - sorted_eigenvalues[0])[:5])


def plot_first_4_eigenstates():
    hamiltonian = generate_hamiltonian_relaxed(resolution=(8, 8, 13))

    fig = plot_first_4_eigenvectors(hamiltonian)
    fig.show()
    input()
