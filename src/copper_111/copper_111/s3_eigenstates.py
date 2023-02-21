import numpy as np

from surface_potential_analysis.energy_eigenstate import save_energy_eigenstates
from surface_potential_analysis.hamiltonian import calculate_energy_eigenstates

from .s2_hamiltonian import generate_hamiltonian
from .surface_data import get_data_path


def generate_eigenstates_data():
    h = generate_hamiltonian(resolution=(12, 12, 13))

    kx_points = np.linspace(0, (np.abs(h.dkx0[0]) + np.abs(h.dkx1[0])) / 2, 5)
    ky_points = np.linspace(0, (np.abs(h.dkx0[1]) + np.abs(h.dkx1[1])) / 2, 5)

    h = generate_hamiltonian(resolution=(23, 23, 10))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(10))
    )
    path = get_data_path("eigenstates_23_23_10.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian(resolution=(23, 23, 12))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(10))
    )
    path = get_data_path("eigenstates_23_23_12.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian(resolution=(25, 25, 16))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(10))
    )
    path = get_data_path("eigenstates_25_25_16.json")
    save_energy_eigenstates(eigenstates, path)
