import numpy as np

from surface_potential_analysis.eigenstate.eigenstate import EigenstateConfigUtil
from surface_potential_analysis.energy_eigenstate import (
    load_energy_eigenstates,
    normalize_eigenstate_phase,
)
from surface_potential_analysis.overlap_transform import (
    calculate_overlap_transform,
    save_overlap_transform,
)
from surface_potential_analysis.wavepacket_grid import (
    WavepacketGrid,
    calculate_inner_product,
    calculate_normalisation,
    calculate_wavepacket_grid_fourier,
)

from .surface_data import get_data_path


def generate_fcc_wavepacket() -> WavepacketGrid:
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    eigenstates = normalize_eigenstate_phase(eigenstates, (0, 0, 0))

    z_points = np.linspace(-3 * util.characteristic_z, 3 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    return grid


def generate_next_fcc_wavepacket() -> WavepacketGrid:
    """
    Generate a wavepacket grid of a neighboring fcc wavefunction.
    This is just the original wavepacket shifted by -delta_x0,
    which we can achieve by rolling the wavepacket

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


def generate_hcp_wavepacket() -> WavepacketGrid:
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


def generate_next_hcp_wavepacket() -> WavepacketGrid:
    """
    Generate a wavepacket grid of a neighboring hcp wavefunction.
    This is just the original wavepacket shifted by -delta_x0,
    which we can achieve by rolling the wavepacket

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


def calculate_overlap_factor():
    wavepacket_fcc = generate_fcc_wavepacket()
    # 0.9989302646383079 1000 -3 3
    print(calculate_normalisation(wavepacket_fcc))

    wavepacket_hcp = generate_hcp_wavepacket()
    # 0.9989198947450144
    print(calculate_normalisation(wavepacket_hcp))
    # -4.1023209529753966e-07 (should be 0)
    print(calculate_inner_product(wavepacket_fcc, wavepacket_hcp))

    wavepacket_next_fcc = generate_next_fcc_wavepacket()
    # -4.0051421011364586e-10 (should be 0)
    print(calculate_inner_product(wavepacket_fcc, wavepacket_next_fcc))

    wavepacket_next_hcp = generate_next_hcp_wavepacket()
    # 2.3287538434176675e-09 (should be 0)
    print(calculate_inner_product(wavepacket_hcp, wavepacket_next_hcp))

    transform_hcp_fcc = calculate_overlap_transform(wavepacket_fcc, wavepacket_hcp)
    path = get_data_path("overlap_transform_hcp_fcc.npz")
    save_overlap_transform(path, transform_hcp_fcc)

    transform_fcc_fcc = calculate_overlap_transform(wavepacket_fcc, wavepacket_next_fcc)
    path = get_data_path("overlap_transform_fcc_fcc.npz")
    save_overlap_transform(path, transform_fcc_fcc)

    transform_hcp_hcp = calculate_overlap_transform(wavepacket_hcp, wavepacket_next_hcp)
    path = get_data_path("overlap_transform_hcp_hcp.npz")
    save_overlap_transform(path, transform_hcp_hcp)
