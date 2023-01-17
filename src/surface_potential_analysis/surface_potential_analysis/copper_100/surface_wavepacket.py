from typing import List

import numpy as np

from ..energy_data.energy_eigenstate import (
    EnergyEigenstates,
    filter_eigenstates_grid,
    filter_eigenstates_n_point,
    load_energy_eigenstates,
    normalize_eigenstate_phase,
    save_energy_eigenstates,
)
from ..energy_data.wavepacket_grid import (
    calculate_wavepacket_grid,
    save_wavepacket_grid,
)
from ..hamiltonian import generate_energy_eigenstates_grid
from .surface_data import get_data_path
from .surface_hamiltonian import generate_hamiltonian


def generate_eigenstates_grid():
    h = generate_hamiltonian(resolution=(14, 14, 15))
    path = get_data_path("copper_eigenstates_grid_5.json")

    generate_energy_eigenstates_grid(path, h, grid_size=4)


def generate_eigenstates_grid_offset():
    h = generate_hamiltonian(resolution=(12, 12, 15))
    path = get_data_path("copper_eigenstates_grid_offset.json")

    generate_energy_eigenstates_grid(path, h, grid_size=4, include_zero=False)


def generate_normalized_eigenstates_grid():
    path = get_data_path("copper_eigenstates_grid_3.json")
    eigenstates = load_energy_eigenstates(path)

    normalized = normalize_eigenstate_phase(eigenstates)
    path = get_data_path("copper_eigenstates_grid_normalized3.json")
    save_energy_eigenstates(normalized, path)


def remove_k_from_eigenstates_grid(
    eigenstates: EnergyEigenstates, kx_points: List[float], ky_points: List[float]
) -> EnergyEigenstates:
    removed = np.zeros_like(eigenstates["kx_points"], dtype=bool)
    for kx in kx_points:
        print(kx, np.equal(eigenstates["kx_points"], kx))
        removed = np.logical_or(removed, np.equal(eigenstates["kx_points"], kx))
    for ky in ky_points:
        removed = np.logical_or(removed, np.equal(eigenstates["ky_points"], ky))
    print(removed)
    filtered = np.logical_not(removed)
    return {
        "eigenstate_config": eigenstates["eigenstate_config"],
        "eigenvalues": np.array(eigenstates["eigenvalues"])[filtered].tolist(),
        "eigenvectors": np.array(eigenstates["eigenvectors"])[filtered].tolist(),
        "kx_points": np.array(eigenstates["kx_points"])[filtered].tolist(),
        "ky_points": np.array(eigenstates["ky_points"])[filtered].tolist(),
    }


# Remove the extra point we repeated when generating eigenstates
def remove_max_k_point(eigenstates: EnergyEigenstates) -> EnergyEigenstates:
    kx_point = np.max(eigenstates["kx_points"])
    ky_point = np.max(eigenstates["ky_points"])
    return remove_k_from_eigenstates_grid(eigenstates, [kx_point], [ky_point])


def generate_wavepacket_grid():
    path = get_data_path("copper_eigenstates_grid_5.json")
    eigenstates = load_energy_eigenstates(path)

    normalized = normalize_eigenstate_phase(eigenstates)

    wavepacket = calculate_wavepacket_grid(normalized)
    path = get_data_path("copper_eigenstates_wavepacket_5.json")
    save_wavepacket_grid(wavepacket, path)


# Uses old data with repeating k point
def generate_wavepacket_grid_old():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)
    filtered = remove_max_k_point(eigenstates)

    wavepacket = calculate_wavepacket_grid(filtered)
    path = get_data_path("copper_eigenstates_wavepacket.json")
    save_wavepacket_grid(wavepacket, path)


# Does not give good results
def generate_wavepacket_grid_old_approx():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)
    filtered = remove_max_k_point(eigenstates)

    wavepacket = calculate_wavepacket_grid(filtered, cutoff=200)
    path = get_data_path("copper_eigenstates_wavepacket_approx2.json")
    save_wavepacket_grid(wavepacket, path)


def generate_wavepacket_grid_old_4_points():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    filtered = filter_eigenstates_n_point(remove_max_k_point(eigenstates), n=4)
    wavepacket = calculate_wavepacket_grid(filtered)
    path = get_data_path("copper_eigenstates_wavepacket_4_point_2.json")
    save_wavepacket_grid(wavepacket, path)


def generate_wavepacket_grid_old_4_points_approx():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    filtered = filter_eigenstates_n_point(remove_max_k_point(eigenstates), n=4)
    wavepacket = calculate_wavepacket_grid(filtered, cutoff=200)
    path = get_data_path("copper_eigenstates_wavepacket_4_point_approx.json")
    save_wavepacket_grid(wavepacket, path)


def filter_eigenstates_origin_point(eigenstates: EnergyEigenstates):
    return filter_eigenstates_grid(eigenstates, [0.0], [0.0])


# Generates a wavepacket as if the band was flat
def generate_wavepacket_grid_flat_band():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    filtered = remove_max_k_point((eigenstates))
    kx_points = filtered["kx_points"]
    ky_points = filtered["ky_points"]

    single_point_eigenstates = filter_eigenstates_origin_point(eigenstates)
    wavepacket = calculate_wavepacket_grid(single_point_eigenstates)

    xv, yv, zv = np.meshgrid(
        wavepacket["x_points"], wavepacket["y_points"], wavepacket["z_points"]
    )
    coords = np.array([xv.ravel(), yv.ravel(), zv.ravel()]).T

    points = np.array(wavepacket["points"]).ravel()
    delta_x = eigenstates["eigenstate_config"]["delta_x"]
    delta_y = eigenstates["eigenstate_config"]["delta_y"]

    wavepacket_points = np.zeros_like(points)
    for (kx, ky) in zip(kx_points, ky_points):
        wavepacket_points += (
            points
            * np.exp(1j * ((coords[:, 0] - (coords[:, 0] % delta_x)) * kx))
            * np.exp(1j * ((coords[:, 1] - (coords[:, 1] % delta_y)) * ky))
        )

    wavepacket_points /= len(kx_points)
    wavepacket["points"] = (wavepacket_points).reshape(xv.shape).tolist()

    path = get_data_path("copper_eigenstates_wavepacket_flat_band.json")
    save_wavepacket_grid(wavepacket, path)
