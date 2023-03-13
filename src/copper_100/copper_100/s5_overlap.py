import numpy as np

from surface_potential_analysis.eigenstate import EigenstateConfigUtil
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
    calculate_wavepacket_grid_fourier_fourier,
    load_wavepacket_grid,
    save_wavepacket_grid,
)

from .surface_data import get_data_path


def generate_wavepacket() -> WavepacketGrid:
    path = get_data_path("eigenstates_grid_0.json")
    path = get_data_path("eigenstates_grid_large_0.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    origin_point = (util.delta_x0[0] / 2, util.delta_x1[1] / 2, 0)
    eigenstates = normalize_eigenstate_phase(eigenstates, origin_point)

    z_points = np.linspace(-5 * util.characteristic_z, 5 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    return grid
    path = get_data_path("wavepacket_0.json")
    save_wavepacket_grid(grid, path)
    return load_wavepacket_grid(path)


def generate_next_neighboring_wavepacket() -> WavepacketGrid:
    """
    Generate a wavepacket grid of a neighboring groundstate wavefunction.
    This is just the original wavepacket shifted by -delta_x0,
    which we can achieve by rolling the wavepacket

    Returns
    -------
    WavepacketGrid
        Wavepacket at the next site
    """
    path = get_data_path("eigenstates_grid_0.json")
    path = get_data_path("eigenstates_grid_large_0.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    origin_point = (-util.delta_x0[0] / 2, util.delta_x1[1] / 2, 0)
    eigenstates = normalize_eigenstate_phase(eigenstates, origin_point)
    z_points = np.linspace(-5 * util.characteristic_z, 5 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    return grid
    path = get_data_path("wavepacket_next_0.json")
    save_wavepacket_grid(grid, path)
    return load_wavepacket_grid(path)


def calculate_overlap_factor():
    wavepacket_0 = generate_wavepacket()
    # 0.998890676148038 1000 -3 3
    print(calculate_normalisation(wavepacket_0))

    wavepacket_next_0 = generate_next_neighboring_wavepacket()
    # -1.8273405679087626e-08 (should be 0)
    print(calculate_inner_product(wavepacket_0, wavepacket_next_0))

    transform_hcp_fcc = calculate_overlap_transform(wavepacket_0, wavepacket_next_0)
    path = get_data_path("overlap_transform_large_0_next_0.npz")
    save_overlap_transform(path, transform_hcp_fcc)


def generate_wavepacket_double_fourier() -> WavepacketGrid:
    path = get_data_path("eigenstates_grid_offset_0.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    origin_point = (util.delta_x0[0] / 2, util.delta_x1[1] / 2, 0)
    eigenstates = normalize_eigenstate_phase(eigenstates, origin_point)

    z_points = np.linspace(-3 * util.characteristic_z, 3 * util.characteristic_z, 20)
    grid_chunks = [
        calculate_wavepacket_grid_fourier_fourier(
            eigenstates, z_points, x0_lim=(-4, 4), x1_lim=(-4, 4)
        )
        for z in [0.0]
    ]
    grid = grid_chunks[0]
    # grid["points"] = np.concatenate(
    #     [chunk["points"] for chunk in grid_chunks], axis=-1
    # ).tolist()
    print(np.shape(grid["points"]))
    return grid


def generate_next_neighboring_wavepacket_double_fourier() -> WavepacketGrid:
    """
    Generate a wavepacket grid of a neighboring groundstate wavefunction.
    This is just the original wavepacket shifted by -delta_x0,
    which we can achieve by rolling the wavepacket

    Returns
    -------
    WavepacketGrid
        Wavepacket at the next site
    """
    path = get_data_path("eigenstates_grid_offset_0.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    origin_point = (-util.delta_x0[0] / 2, util.delta_x1[1] / 2, 0)
    eigenstates = normalize_eigenstate_phase(eigenstates, origin_point)

    z_points = np.linspace(-3 * util.characteristic_z, 3 * util.characteristic_z, 1000)
    grid_chunks = [
        calculate_wavepacket_grid_fourier_fourier(
            eigenstates, [z], x0_lim=(-4, 4), x1_lim=(-4, 4)
        )
        for z in z_points
    ]
    grid = grid_chunks[0]
    grid["z_points"] = z_points.tolist()

    points = np.empty(shape=(*np.shape(grid["points"]), len(grid_chunks)))
    for (i, g) in enumerate(grid_chunks):
        points[:, :, i] = g["points"][0]
    grid["points"] = points.tolist()
    # print(np.shape(grid["points"]))
    return grid


def calculate_overlap_factor_double_fourier():
    wavepacket_0 = generate_wavepacket_double_fourier()
    # 0.998890676148038 1000 -3 3
    print(calculate_normalisation(wavepacket_0))

    wavepacket_next_0 = generate_next_neighboring_wavepacket_double_fourier()
    # -1.8273405679087626e-08 (should be 0)
    print(calculate_inner_product(wavepacket_0, wavepacket_next_0))

    transform_hcp_fcc = calculate_overlap_transform(wavepacket_0, wavepacket_next_0)
    path = get_data_path("overlap_transform_0_next_0.npz")
    save_overlap_transform(path, transform_hcp_fcc)
