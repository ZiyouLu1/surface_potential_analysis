import numpy as np

from surface_potential_analysis.energy_eigenstate import save_energy_eigenstates
from surface_potential_analysis.hamiltonian import calculate_energy_eigenstates

from .hamiltonian import generate_hamiltonian
from .surface_data import get_data_path


def generate_eigenstates_data():
    h1 = generate_hamiltonian(resolution=(14, 14, 10))

    kx_points = np.linspace(-h1.dkx0[0] / 2, 0, 5)
    ky_points = np.zeros_like(kx_points, dtype=float)

    # h = generate_hamiltonian(resolution=(15, 15, 13))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_15_15_13.json")
    # save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian(resolution=(15, 15, 14))
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    path = get_data_path("eigenstates_15_15_14.json")
    save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(14, 14, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_14_14_14.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(13, 13, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_13_13_14.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(12, 12, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_12_12_14.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(11, 11, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_11_11_14.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(10, 10, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_10_10_14.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(10, 10, 15))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_10_10_15.json")
    # save_energy_eigenstates(eigenstates, path)
