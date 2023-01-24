import numpy as np

from surface_potential_analysis.energy_eigenstate import save_energy_eigenstates
from surface_potential_analysis.hamiltonian import calculate_energy_eigenstates

from .hamiltonian import generate_hamiltonian
from .surface_data import get_data_path


def generate_eigenstates_data():
    h1 = generate_hamiltonian(resolution=(14, 14, 10))

    kx_points = np.linspace(-h1.dkx1[0] / 2, h1.dkx1[0] / 2, 11)
    ky_points = np.zeros_like(kx_points)

    # eigenstates1 = calculate_energy_eigenstates(h1, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_14_14_10.json")
    # save_energy_eigenstates(eigenstates1, path)

    # h2 = generate_hamiltonian(resolution=(12, 12, 10))
    # eigenstates2 = calculate_energy_eigenstates(h2, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_12_12_10.json")
    # save_energy_eigenstates(eigenstates2, path)

    # h3 = generate_hamiltonian(resolution=(12, 12, 12))
    # eigenstates3 = calculate_energy_eigenstates(h3, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_12_12_12.json")
    # save_energy_eigenstates(eigenstates3, path)

    # h = generate_hamiltonian(resolution=(12, 12, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_12_12_14.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(12, 12, 15))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_12_12_15.json")
    # save_energy_eigenstates(eigenstates, path)

    # h6 = generate_hamiltonian(resolution=(13, 13, 15))
    # eigenstates6 = calculate_energy_eigenstates(h6, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_12_12_16.json")
    # save_energy_eigenstates(eigenstates6, path)

    h = generate_hamiltonian(resolution=(10, 10, 15))
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    path = get_data_path("copper_eigenstates_10_10_15.json")
    save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(13, 13, 15))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_13_13_15.json")
    # save_energy_eigenstates(eigenstates, path)
