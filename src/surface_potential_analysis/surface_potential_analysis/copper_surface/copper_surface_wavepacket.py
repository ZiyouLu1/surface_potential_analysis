from typing import List

import numpy as np

from ..energy_data.energy_eigenstates import (
    EnergyEigenstates,
    load_energy_eigenstates,
    save_energy_eigenstates,
    save_wavepacket_grid,
)
from ..hamiltonian import (
    calculate_wavepacket_grid,
    calculate_wavepacket_grid_with_edge,
    generate_energy_eigenstates_grid,
    normalize_eigenstate_phase,
)
from .copper_surface_data import get_data_path
from .copper_surface_hamiltonian import generate_hamiltonian


def generate_eigenstates_grid():
    h = generate_hamiltonian(resolution=(12, 12, 15))
    path = get_data_path("copper_eigenstates_grid_4.json")

    generate_energy_eigenstates_grid(path, h, grid_size=8)


def generate_eigenstates_grid_offset():
    h = generate_hamiltonian(resolution=(12, 12, 15))
    path = get_data_path("copper_eigenstates_grid_offset.json")

    generate_energy_eigenstates_grid(path, h, grid_size=4)


def generate_normalized_eigenstates_grid():
    path = get_data_path("copper_eigenstates_grid_3.json")
    eigenstates = load_energy_eigenstates(path)

    h = generate_hamiltonian(eigenstates["resolution"])
    normalized = normalize_eigenstate_phase(h, eigenstates)
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    save_energy_eigenstates(normalized, path)


def filter_eigenstates_grid(
    eigenstates: EnergyEigenstates, kx_points: List[float], ky_points: List[float]
) -> EnergyEigenstates:
    removed = np.zeros_like(eigenstates["kx_points"], dtype=bool)
    for kx in kx_points:
        removed = np.logical_or(removed, np.equal(eigenstates["kx_points"], kx))
    for ky in ky_points:
        removed = np.logical_or(removed, np.equal(eigenstates["ky_points"], ky))

    filtered = np.logical_not(removed)
    return {
        "eigenstate_config": eigenstates["eigenstate_config"],
        "eigenvalues": np.array(eigenstates["eigenvalues"])[filtered].tolist(),
        "eigenvectors": np.array(eigenstates["eigenvectors"])[filtered].tolist(),
        "kx_points": np.array(eigenstates["kx_points"])[filtered].tolist(),
        "ky_points": np.array(eigenstates["ky_points"])[filtered].tolist(),
        "resolution": eigenstates["resolution"],
    }


# Remove the extra point we repeated when generating eigenstates
def fix_eigenstates_grid(eigenstates: EnergyEigenstates) -> EnergyEigenstates:
    kx_point = np.max(eigenstates["kx_points"])
    ky_point = np.max(eigenstates["ky_points"])
    return filter_eigenstates_grid(eigenstates, [kx_point], [ky_point])


def calculate_wavepacket_with_edge():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    wavepacket = calculate_wavepacket_grid_with_edge(eigenstates)
    path = get_data_path("copper_eigenstates_wavepacket_with_edge.json")
    save_wavepacket_grid(wavepacket, path)


def calculate_wavepacket():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)
    filtered = fix_eigenstates_grid(eigenstates)

    wavepacket = calculate_wavepacket_grid(filtered)
    path = get_data_path("copper_eigenstates_wavepacket.json")
    save_wavepacket_grid(wavepacket, path)


def calculate_wavepacket_approx():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)
    filtered = fix_eigenstates_grid(eigenstates)

    wavepacket = calculate_wavepacket_grid(filtered, cutoff=200)
    path = get_data_path("copper_eigenstates_wavepacket_approx2.json")
    save_wavepacket_grid(wavepacket, path)


def filter_eigenstates_4_point(eigenstates: EnergyEigenstates):
    kx_points = np.sort(np.unique(eigenstates["kx_points"]))[1::2].tolist()
    ky_points = np.sort(np.unique(eigenstates["ky_points"]))[1::2].tolist()
    return filter_eigenstates_grid(eigenstates, kx_points, ky_points)


def calculate_wavepacket_4_points_approx():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    filtered = fix_eigenstates_grid(filter_eigenstates_4_point(eigenstates))
    wavepacket = calculate_wavepacket_grid(filtered, cutoff=200)
    path = get_data_path("copper_eigenstates_wavepacket_4_point_approx.json")
    save_wavepacket_grid(wavepacket, path)


def calculate_wavepacket_4_points():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    filtered = fix_eigenstates_grid(filter_eigenstates_4_point(eigenstates))
    wavepacket = calculate_wavepacket_grid(filtered)
    path = get_data_path("copper_eigenstates_wavepacket_4_point.json")
    save_wavepacket_grid(wavepacket, path)


def filter_eigenstates_1_point(eigenstates: EnergyEigenstates):
    kx_points = list(set(kx for kx in eigenstates["kx_points"] if kx != 0))
    ky_points = list(set(ky for ky in eigenstates["ky_points"] if ky != 0))
    return filter_eigenstates_grid(eigenstates, kx_points, ky_points)


def calculate_wavepacket_one_point():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    filtered = filter_eigenstates_1_point(eigenstates)
    wavepacket = calculate_wavepacket_grid(filtered)

    xv, yv, zv = np.meshgrid(
        wavepacket["x_points"], wavepacket["y_points"], wavepacket["z_points"]
    )
    coords = np.array([xv.ravel(), yv.ravel(), zv.ravel()]).T

    points = np.array(wavepacket["points"]).ravel()
    delta_x = eigenstates["eigenstate_config"]["delta_x"]
    delta_y = eigenstates["eigenstate_config"]["delta_y"]
    points = (
        points
        * np.sinc((coords[:, 0] - (delta_x / 2)) / delta_x)
        * np.sinc((coords[:, 1] - (delta_y / 2)) / delta_y)
    )
    wavepacket["points"] = points.reshape(xv.shape).tolist()

    path = get_data_path("copper_eigenstates_wavepacket_1_point.json")
    save_wavepacket_grid(wavepacket, path)
