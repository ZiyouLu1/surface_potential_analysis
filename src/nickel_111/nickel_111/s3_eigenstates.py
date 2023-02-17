import numpy as np

from surface_potential_analysis.energy_eigenstate import save_energy_eigenstates
from surface_potential_analysis.hamiltonian import calculate_energy_eigenstates

from .s2_hamiltonian import generate_hamiltonian, generate_hamiltonian_john
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

    h = generate_hamiltonian(resolution=(23, 23, 14))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(10))
    )
    path = get_data_path("eigenstates_23_23_14.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian(resolution=(23, 23, 16))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(10))
    )
    path = get_data_path("eigenstates_23_23_16.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian(resolution=(25, 25, 16))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(10))
    )
    path = get_data_path("eigenstates_25_25_16.json")
    save_energy_eigenstates(eigenstates, path)


def generate_eigenstates_data_john():
    h = generate_hamiltonian_john(resolution=(12, 12, 13))

    kx_points = np.linspace(-h.dkx0[0] / 2, 0, 5)
    ky_points = np.zeros_like(kx_points)

    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_12_12_13.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(12, 12, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_12_12_14.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(10, 10, 13))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_10_10_13.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(14, 14, 13))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_14_14_13.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(12, 12, 12))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_12_12_12.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(15, 15, 12))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_15_15_12.json")
    # save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian_john(resolution=(12, 18, 12))
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    path = get_data_path("eigenstates_12_18_12.json")
    save_energy_eigenstates(eigenstates, path)
