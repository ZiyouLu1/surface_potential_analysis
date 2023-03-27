import numpy as np
from numpy.typing import NDArray

from surface_potential_analysis.eigenstate.eigenstate import EigenstateConfigUtil
from surface_potential_analysis.energy_eigenstate import (
    load_energy_eigenstates,
    normalize_eigenstate_phase,
)
from surface_potential_analysis.interpolation import (
    interpolate_real_points_along_axis_fourier,
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
    load_wavepacket_grid,
    save_wavepacket_grid,
)

from .surface_data import get_data_path


def generate_fcc_wavepacket() -> WavepacketGrid:
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    eigenstates = normalize_eigenstate_phase(eigenstates, (0, 0, 0))

    z_points = np.linspace(-5 * util.characteristic_z, 5 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    return grid
    path = get_data_path("fcc_wavepacket.json")
    # save_wavepacket_grid(grid, path)
    return load_wavepacket_grid(path)


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
    z_points = np.linspace(-5 * util.characteristic_z, 5 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    grid["points"] = np.roll(
        grid["points"], shift=-eigenstates["eigenstate_config"]["resolution"][0], axis=0
    ).tolist()
    return grid
    path = get_data_path("next_fcc_wavepacket.json")
    save_wavepacket_grid(grid, path)
    return load_wavepacket_grid(path)


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

    z_points = np.linspace(-5 * util.characteristic_z, 5 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    return grid
    path = get_data_path("hcp_wavepacket.json")
    # save_wavepacket_grid(grid, path)
    return load_wavepacket_grid(path)


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

    z_points = np.linspace(-5 * util.characteristic_z, 5 * util.characteristic_z, 1000)
    grid = calculate_wavepacket_grid_fourier(
        eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    )
    grid["points"] = np.roll(
        grid["points"], shift=-eigenstates["eigenstate_config"]["resolution"][0], axis=0
    ).tolist()
    return grid
    path = get_data_path("next_hcp_wavepacket.json")
    save_wavepacket_grid(grid, path)
    return load_wavepacket_grid(path)


def calculate_overlap_factor():
    wavepacket_fcc = generate_fcc_wavepacket()
    # 0.9989499296071063 1000 -3 3
    # 0.9994494691023869 2000 -3 3
    # 0.9989999372435083 1000 -4 4
    print(calculate_normalisation(wavepacket_fcc))

    wavepacket_hcp = generate_hcp_wavepacket()
    # 0.9989454040074838
    print(calculate_normalisation(wavepacket_hcp))
    # -2.592593651271823e-07 (should be 0)
    print(calculate_inner_product(wavepacket_fcc, wavepacket_hcp))

    wavepacket_next_fcc = generate_next_fcc_wavepacket()
    # -1.381731140564679e-09 (should be 0)
    print(calculate_inner_product(wavepacket_fcc, wavepacket_next_fcc))

    wavepacket_next_hcp = generate_next_hcp_wavepacket()
    # 4.12815207838777e-09
    print(calculate_inner_product(wavepacket_hcp, wavepacket_next_hcp))

    transform_hcp_fcc = calculate_overlap_transform(wavepacket_fcc, wavepacket_hcp)
    path = get_data_path("overlap_transform_shifted_hcp_fcc.npz")
    save_overlap_transform(path, transform_hcp_fcc)

    transform_fcc_fcc = calculate_overlap_transform(wavepacket_fcc, wavepacket_next_fcc)
    path = get_data_path("overlap_transform_orthogonal_fcc_fcc.npz")
    save_overlap_transform(path, transform_fcc_fcc)

    transform_hcp_hcp = calculate_overlap_transform(wavepacket_hcp, wavepacket_next_hcp)
    path = get_data_path("overlap_transform_orthogonal_hcp_hcp.npz")
    save_overlap_transform(path, transform_hcp_hcp)


def interpolate_real_wavepacket_grid_points_fourier(
    grid: WavepacketGrid, shape: tuple[int, int]
) -> NDArray:
    return interpolate_real_points_along_axis_fourier(
        interpolate_real_points_along_axis_fourier(grid["points"], shape[0], axis=0),
        shape[1],
        axis=1,
    )


def calculate_overlap_factor_interpolated():
    """
    To test the effect of the finite resolution of the overlap
    We calculate the integral after doubling the resolution in the xy direction
    """
    wavepacket_fcc = generate_fcc_wavepacket()
    old_shape = np.shape(wavepacket_fcc["points"])[0:2]
    new_shape = (old_shape[0] * 2, old_shape[1] * 2)
    wavepacket_fcc["points"] = interpolate_real_wavepacket_grid_points_fourier(
        wavepacket_fcc, new_shape
    )

    wavepacket_hcp = generate_hcp_wavepacket()
    wavepacket_hcp["points"] = interpolate_real_wavepacket_grid_points_fourier(
        wavepacket_hcp, new_shape
    )
    transform_hcp_fcc = calculate_overlap_transform(wavepacket_fcc, wavepacket_hcp)
    path = get_data_path("overlap_transform_interpolated_hcp_fcc.npz")
    save_overlap_transform(path, transform_hcp_fcc)


def pad_wavepacket_xy(grid: WavepacketGrid, shape: tuple[int, int]) -> WavepacketGrid:
    points = np.asarray(grid["points"])
    old_shape = np.shape(points)
    pad_x0 = (shape[0] - old_shape[0]) // 2
    pad_x1 = (shape[1] - old_shape[1]) // 2
    padded = np.pad(
        points, [(pad_x0, pad_x0), (pad_x1, pad_x1), (0, 0)], mode="constant"
    )
    scale_x0 = shape[0] / old_shape[0]
    scale_x1 = shape[1] / old_shape[1]
    return {
        "delta_x0": (grid["delta_x0"][0] * scale_x0, grid["delta_x0"][1] * scale_x0),
        "delta_x1": (grid["delta_x1"][0] * scale_x1, grid["delta_x1"][1] * scale_x1),
        "points": padded.tolist(),
        "z_points": grid["z_points"],
    }


def calculate_overlap_factor_extended() -> None:
    """
    To test the effect of the finite grid has on the overlap factor
    since the overlap tends to zero far from the center we can just pad the with zeroes

    Even better we can pad it with zeroes that doesn't necessarily match up with the grid spacing
    """
    wavepacket_fcc = generate_fcc_wavepacket()
    old_shape = np.shape(wavepacket_fcc["points"])[0:2]
    new_shape = (round(old_shape[0] * 1.8), round(old_shape[1] * 1.8))
    print(old_shape, new_shape)
    wavepacket_fcc = pad_wavepacket_xy(wavepacket_fcc, new_shape)

    wavepacket_hcp = generate_hcp_wavepacket()
    wavepacket_hcp = pad_wavepacket_xy(wavepacket_hcp, new_shape)
    transform_hcp_fcc = calculate_overlap_transform(wavepacket_fcc, wavepacket_hcp)
    path = get_data_path("overlap_transform_extended_hcp_fcc.npz")
    save_overlap_transform(path, transform_hcp_fcc)
