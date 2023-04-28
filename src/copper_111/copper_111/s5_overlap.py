from __future__ import annotations

import numpy as np
from surface_potential_analysis.overlap.calculation import calculate_wavepacket_overlap
from surface_potential_analysis.overlap.overlap import save_overlap
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    calculate_normalisation,
)

from .surface_data import get_data_path


def generate_fcc_wavepacket() -> Wavepacket:
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    eigenstates = normalize_eigenstate_phase(eigenstates, (0, 0, 0))

    z_points = np.linspace(-3 * util.characteristic_z, 3 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    return grid


def generate_next_fcc_wavepacket() -> Wavepacket:
    """
    Generate a wavepacket grid of a neighboring fcc wavefunction.
    This is just the original wavepacket shifted by -delta_x0,
    which we can achieve by rolling the wavepacket.

    Returns
    -------
    WavepacketGrid
        Wavepacket at the next fcc site
    """
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    eigenstates = normalize_eigenstate_phase(eigenstates, (0, 0, 0))
    z_points = np.linspace(-3 * util.characteristic_z, 3 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    grid["points"] = np.roll(
        grid["points"], shift=-eigenstates["eigenstate_config"]["resolution"][0], axis=0
    ).tolist()
    return grid


def generate_hcp_wavepacket() -> Wavepacket:
    path = get_data_path("eigenstates_grid_1.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    eigenstates = normalize_eigenstate_phase(
        eigenstates,
        (
            (util.delta_x0[0] + util.delta_x1[0]) / 3,
            (util.delta_x0[1] + util.delta_x1[1]) / 3,
            0,
        ),
    )

    z_points = np.linspace(-3 * util.characteristic_z, 3 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    return grid


def generate_next_hcp_wavepacket() -> Wavepacket:
    """
    Generate a wavepacket grid of a neighboring hcp wavefunction.
    This is just the original wavepacket shifted by -delta_x0,
    which we can achieve by rolling the wavepacket.

    Returns
    -------
    WavepacketGrid
        Wavepacket at the next hcp site
    """
    path = get_data_path("eigenstates_grid_1.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    eigenstates = normalize_eigenstate_phase(
        eigenstates,
        (
            (util.delta_x0[0] + util.delta_x1[0]) / 3,
            (util.delta_x0[1] + util.delta_x1[1]) / 3,
            0,
        ),
    )

    z_points = np.linspace(-3 * util.characteristic_z, 3 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    grid["points"] = np.roll(
        grid["points"], shift=-eigenstates["eigenstate_config"]["resolution"][0], axis=0
    ).tolist()
    return grid


def calculate_overlap_factor() -> None:
    wavepacket_fcc = generate_fcc_wavepacket()
    print(calculate_normalisation(wavepacket_fcc))  # noqa: T201

    wavepacket_hcp = generate_hcp_wavepacket()
    print(calculate_normalisation(wavepacket_hcp))  # noqa: T201

    wavepacket_next_fcc = generate_next_fcc_wavepacket()
    wavepacket_next_hcp = generate_next_hcp_wavepacket()

    overlap_hcp_fcc = calculate_wavepacket_overlap(wavepacket_fcc, wavepacket_hcp)
    print(np.sum(overlap_hcp_fcc["vector"]))  # noqa: T201
    path = get_data_path("overlap_hcp_fcc.npz")
    save_overlap(path, overlap_hcp_fcc)

    overlap_fcc_fcc = calculate_wavepacket_overlap(wavepacket_fcc, wavepacket_next_fcc)
    print(np.sum(overlap_fcc_fcc["vector"]))  # noqa: T201
    path = get_data_path("overlap_fcc_fcc.npz")
    save_overlap(path, overlap_fcc_fcc)

    overlap_hcp_hcp = calculate_wavepacket_overlap(wavepacket_hcp, wavepacket_next_hcp)
    print(np.sum(overlap_hcp_hcp["vector"]))  # noqa: T201
    path = get_data_path("overlap_hcp_hcp.npz")
    save_overlap(path, overlap_hcp_hcp)
