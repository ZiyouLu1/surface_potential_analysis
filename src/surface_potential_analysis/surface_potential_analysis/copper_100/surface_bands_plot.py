import numpy as np

from ..energy_data.plot_energy_eigenstates import plot_lowest_band_in_kx
from ..hamiltonian import calculate_energy_eigenstates
from ..plot_surface_hamiltonian import plot_bands_occupation, plot_first_4_eigenvectors
from .surface_data import save_figure
from .surface_hamiltonian import generate_hamiltonian


def plot_first_copper_bands():
    h = generate_hamiltonian(resolution=(12, 12, 10))
    fig = plot_first_4_eigenvectors(h)
    save_figure(fig, "copper_first_4_bands.png")
    fig.show()


def list_first_copper_band_energies():
    h = generate_hamiltonian(resolution=(12, 12, 15))
    e_vals, _ = h.calculate_eigenvalues(0, 0)
    print(list(np.sort(e_vals)[:40]))


def plot_copper_band_structure():
    h = generate_hamiltonian(resolution=(12, 12, 10))

    kx_points = np.linspace(-h.dkx / 2, h.dkx / 2, 21)
    ky_points = np.zeros_like(kx_points)
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)

    fig, ax, _ = plot_lowest_band_in_kx(eigenstates)
    ax.set_title("Plot of energy against k for the lowest band of Copper for Ky=0")
    ax.set_xlabel("K /$m^-1$")
    ax.set_ylabel("energy / J")
    fig.show()
    save_figure(fig, "copper_lowest_band.png")


def plot_copper_bands_occupation():
    h = generate_hamiltonian(resolution=(12, 12, 10))
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


def compare_first_5_bands():
    hamiltonian = generate_hamiltonian(resolution=(12, 12, 15))

    eigenvalues, _ = hamiltonian.calculate_eigenvalues(0, 0)
    sorted_eigenvalues = np.sort(eigenvalues)
    print(sorted_eigenvalues[:5])
    print((sorted_eigenvalues - sorted_eigenvalues[0])[:5])


def plot_first_4_eigenstates():
    hamiltonian = generate_hamiltonian(resolution=(12, 12, 15))

    fig = plot_first_4_eigenvectors(hamiltonian)
    fig.show()
    input()
