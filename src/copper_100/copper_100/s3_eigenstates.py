import numpy as np

from surface_potential_analysis.energy_eigenstate import save_energy_eigenstates
from surface_potential_analysis.hamiltonian import calculate_energy_eigenstates

from .s2_hamiltonian import generate_hamiltonian, generate_hamiltonian_relaxed
from .surface_data import get_data_path


def generate_eigenstates_data_relaxed():
    h1 = generate_hamiltonian_relaxed(resolution=(14, 14, 10))

    kx_points = np.linspace(-h1.dkx0[0] / 2, 0, 5)
    ky_points = np.zeros_like(kx_points, dtype=float)

    h = generate_hamiltonian_relaxed(resolution=(17, 17, 13))
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    path = get_data_path("eigenstates_17_17_13.json")
    save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian_relaxed(resolution=(6, 6, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path("eigenstates_6_6_14.json")
    # save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian_relaxed(resolution=(17, 17, 15))
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    path = get_data_path("eigenstates_17_17_15.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian_relaxed(resolution=(21, 21, 14))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(5))
    )
    path = get_data_path("eigenstates_21_21_14.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian_relaxed(resolution=(21, 21, 15))
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    path = get_data_path("eigenstates_21_21_15.json")
    save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian_relaxed(resolution=(12, 12, 15))
    # eigenstates = calculate_energy_eigenstates(
    #     h, kx_points, ky_points, include_bands=list(range(5))
    # )
    # path = get_data_path("eigenstates_12_12_15.json")
    # save_energy_eigenstates(eigenstates, path)


def generate_eigenstates_data():
    h1 = generate_hamiltonian(resolution=(14, 14, 10))

    kx_points = np.linspace(-h1.dkx0[0] / 2, 0, 5)
    ky_points = np.zeros_like(kx_points, dtype=float)

    h = generate_hamiltonian(resolution=(25, 25, 14))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(20))
    )
    path = get_data_path("eigenstates_25_25_14.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian(resolution=(23, 23, 14))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(20))
    )
    path = get_data_path("eigenstates_23_23_14.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian(resolution=(23, 23, 15))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(20))
    )
    path = get_data_path("eigenstates_23_23_15.json")
    save_energy_eigenstates(eigenstates, path)
