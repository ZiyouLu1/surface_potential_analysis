import numpy as np

from surface_potential_analysis.energy_eigenstate import save_energy_eigenstates
from surface_potential_analysis.hamiltonian import calculate_energy_eigenstates

from .hamiltonian import generate_hamiltonian_relaxed
from .surface_data import get_data_path


def generate_eigenstates_data():
    h1 = generate_hamiltonian_relaxed(resolution=(14, 14, 10))

    kx_points = np.linspace(-h1.dkx1[0] / 2, 0, 5)
    ky_points = np.zeros_like(kx_points, dtype=float)

    h = generate_hamiltonian_relaxed(resolution=(8, 8, 13))
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    path = get_data_path("eigenstates_8_8_13.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian_relaxed(resolution=(6, 6, 14))
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    path = get_data_path("eigenstates_6_6_14.json")
    save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian_relaxed(resolution=(8, 8, 15))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_8_8_15.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian_relaxed(resolution=(10, 10, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_10_10_14.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian_relaxed(resolution=(10, 10, 15))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_10_10_15.json")
    # save_energy_eigenstates(eigenstates, path)
