import numpy as np

from surface_potential_analysis.eigenstate.eigenstate import EigenstateConfigUtil
from surface_potential_analysis.energy_eigenstate import (
    EnergyEigenstatesLegacy,
    filter_eigenstates_grid,
    load_energy_eigenstates,
    normalize_eigenstate_phase,
)
from surface_potential_analysis.hamiltonian import generate_energy_eigenstates_grid
from surface_potential_analysis.wavepacket_grid import (
    calculate_wavepacket_grid,
    calculate_wavepacket_grid_fourier,
    save_wavepacket_grid,
)

from .s2_hamiltonian import generate_hamiltonian
from .surface_data import get_data_path


def normalize_eigenstate_phase_copper(data: EnergyEigenstatesLegacy):
    util = EigenstateConfigUtil(data["eigenstate_config"])
    origin_point = (util.delta_x0[0] / 2, util.delta_x1[1] / 2, 0)
    return normalize_eigenstate_phase(data, origin_point)


def generate_eigenstates_grid_relaxed():
    h = generate_hamiltonian(resolution=(21, 21, 14))
    path = get_data_path("eigenstates_grid_relaxed.json")

    # h = generate_hamiltonian(resolution=(12, 12, 15))
    # path = get_data_path("eigenstates_grid_relaxed_hd.json")

    generate_energy_eigenstates_grid(path, h, size=(4, 4))


def generate_eigenstates_grid_offset():
    h = generate_hamiltonian(resolution=(12, 12, 15))
    path = get_data_path("copper_eigenstates_grid_offset.json")

    generate_energy_eigenstates_grid(path, h, size=(4, 4), include_zero=False)


def remove_k_from_eigenstates_grid(
    eigenstates: EnergyEigenstatesLegacy, kx_points: list[float], ky_points: list[float]
) -> EnergyEigenstatesLegacy:
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
def remove_max_k_point(eigenstates: EnergyEigenstatesLegacy) -> EnergyEigenstatesLegacy:
    kx_point = np.max(eigenstates["kx_points"])
    ky_point = np.max(eigenstates["ky_points"])
    return remove_k_from_eigenstates_grid(eigenstates, [kx_point], [ky_point])


def filter_eigenstates_origin_point(eigenstates: EnergyEigenstatesLegacy):
    return filter_eigenstates_grid(eigenstates, [0.0], [0.0])


def generate_wavepacket_grid_relaxed():
    path = get_data_path("eigenstates_grid_relaxed_hd.json")
    eigenstates = load_energy_eigenstates(path)

    normalized = normalize_eigenstate_phase_copper(eigenstates)

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    wavepacket = calculate_wavepacket_grid(
        normalized,
        delta_x0=util.delta_x0,
        delta_x1=util.delta_x1,
        delta_z=8 * util.characteristic_z,
        shape=(49, 49, 41),
        offset=(util.delta_x0[0] / 2, util.delta_x1[1] / 2, -4 * util.characteristic_z),
    )
    path = get_data_path("relaxed_eigenstates_hd_wavepacket.json")
    save_wavepacket_grid(wavepacket, path)


def generate_wavepacket_grid_relaxed_low_resolution():
    path = get_data_path("eigenstates_grid_relaxed.json")
    eigenstates = load_energy_eigenstates(path)

    normalized = normalize_eigenstate_phase_copper(eigenstates)

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    wavepacket = calculate_wavepacket_grid(
        normalized,
        delta_x0=util.delta_x0,
        delta_x1=util.delta_x1,
        delta_z=12 * util.characteristic_z,
        shape=(24, 24, 21),
        offset=(util.delta_x0[0] / 2, util.delta_x1[1] / 2, -6 * util.characteristic_z),
    )
    print(wavepacket)
    path = get_data_path("relaxed_eigenstates_wavepacket_low_res.json")
    save_wavepacket_grid(wavepacket, path)


def generate_wavepacket_grid_relaxed_flat():
    path = get_data_path("eigenstates_grid_relaxed_hd.json")
    eigenstates = load_energy_eigenstates(path)

    normalized = normalize_eigenstate_phase_copper(eigenstates)

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    wavepacket = calculate_wavepacket_grid(
        normalized,
        delta_x0=util.delta_x0,
        delta_x1=util.delta_x1,
        delta_z=2 * util.characteristic_z,
        shape=(201, 201, 3),
        offset=(util.delta_x0[0] / 2, util.delta_x1[1] / 2, -1 * util.characteristic_z),
    )
    path = get_data_path("relaxed_eigenstates_hd_wavepacket_flat.json")
    save_wavepacket_grid(wavepacket, path)


def generate_wavepacket_grid_new_relaxed():
    path = get_data_path("eigenstates_grid_relaxed.json")
    eigenstates = load_energy_eigenstates(path)

    normalized = normalize_eigenstate_phase_copper(eigenstates)

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    wavepacket = calculate_wavepacket_grid_fourier(
        normalized,
        z_points=np.linspace(-util.characteristic_z, util.characteristic_z, 5).tolist(),
    )
    path = get_data_path("relaxed_eigenstates_wavepacket_new.json")
    save_wavepacket_grid(wavepacket, path)


def generate_eigenstates_grid():
    h = generate_hamiltonian(resolution=(23, 23, 18))
    save_bands = {k: get_data_path(f"eigenstates_grid_{k}.json") for k in range(20)}

    generate_energy_eigenstates_grid(h, size=(4, 4), save_bands=save_bands)
