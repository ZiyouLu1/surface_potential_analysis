from __future__ import annotations

import numpy as np

from copper_100.s1_potential import get_interpolated_potential

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

    h = generate_hamiltonian(resolution=(23, 23, 16))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(20))
    )
    path = get_data_path("eigenstates_23_23_16.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian(resolution=(25, 25, 16))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(20))
    )
    path = get_data_path("eigenstates_25_25_16.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian(resolution=(23, 23, 17))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(20))
    )
    path = get_data_path("eigenstates_23_23_17.json")
    save_energy_eigenstates(eigenstates, path)

    h = generate_hamiltonian(resolution=(23, 23, 18))
    eigenstates = calculate_energy_eigenstates(
        h, kx_points, ky_points, include_bands=list(range(20))
    )
    path = get_data_path("eigenstates_23_23_18.json")
    save_energy_eigenstates(eigenstates, path)


def generate_oversampled_eigenstates_data():
    """Check that oversampling the potential has no effect on the resulting eigenstates."""
    data = get_interpolated_potential((46, 46, 100))
    data = interpolate_energy_grid_xy_fourier(data, (92, 92, 100))
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 179704637926161.6,
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "resolution": (23, 23, 5),
    }
    z_offset = -9.848484848484871e-11
    util = SurfaceHamiltonianUtil(config, as_interpolation(data), z_offset)
    e_values, e_states = util.calculate_eigenvalues(0, 0)
    eigenstates: EnergyEigenstatesLegacy = {
        "eigenstate_config": util._config,
        "eigenvalues": e_values[:20],
        "eigenvectors": [e["eigenvector"] for (i, e) in enumerate(e_states) if i < 20],
        "kx_points": [e["kx"] for (i, e) in enumerate(e_states) if i < 20],
        "ky_points": [e["ky"] for (i, e) in enumerate(e_states) if i < 20],
    }
    path = get_data_path("oversampled_eigenstates.json")
    save_energy_eigenstates(eigenstates, path)

    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 179704637926161.6,
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "resolution": (46, 46, 5),
    }
    z_offset = -9.848484848484871e-11
    util = SurfaceHamiltonianUtil(config, as_interpolation(data), z_offset)
    e_values, e_states = util.calculate_eigenvalues(0, 0)
    eigenstates: EnergyEigenstatesLegacy = {
        "eigenstate_config": util._config,
        "eigenvalues": e_values[:20],
        "eigenvectors": [e["eigenvector"] for (i, e) in enumerate(e_states) if i < 20],
        "kx_points": [e["kx"] for (i, e) in enumerate(e_states) if i < 20],
        "ky_points": [e["ky"] for (i, e) in enumerate(e_states) if i < 20],
    }
    path = get_data_path("not_oversampled_eigenstates.json")
    save_energy_eigenstates(eigenstates, path)
