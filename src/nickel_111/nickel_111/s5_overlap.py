from pathlib import Path
from typing import List, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray

from surface_potential_analysis.eigenstate import EigenstateConfigUtil
from surface_potential_analysis.energy_eigenstate import (
    load_energy_eigenstates,
    normalize_eigenstate_phase,
)
from surface_potential_analysis.surface_config import SurfaceConfigUtil
from surface_potential_analysis.wavepacket_grid import (
    WavepacketGrid,
    calculate_wavepacket_grid_fourier,
    load_wavepacket_grid,
    save_wavepacket_grid,
)

from .surface_data import get_data_path


def generate_fcc_wavepacket() -> WavepacketGrid:
    # path = get_data_path("eigenstates_grid_0.json")
    # eigenstates = load_energy_eigenstates(path)
    # util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    # eigenstates = normalize_eigenstate_phase(eigenstates, (0, 0, 0))

    # z_points = np.linspace(-3 * util.characteristic_z, 3 * util.characteristic_z, 1000)
    # grid = calculate_wavepacket_grid_new(
    #     eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    # )
    path = get_data_path("fcc_wavepacket.json")
    # save_wavepacket_grid(grid, path)
    return load_wavepacket_grid(path)


def generate_hcp_wavepacket() -> WavepacketGrid:
    # path = get_data_path("eigenstates_grid_1.json")
    # eigenstates = load_energy_eigenstates(path)
    # util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    # eigenstates = normalize_eigenstate_phase(
    #     eigenstates,
    #     (
    #         (util.delta_x0[0] + util.delta_x1[0]) / 3,
    #         (util.delta_x0[1] + util.delta_x1[1]) / 3,
    #         0,
    #     ),
    # )

    # z_points = np.linspace(-3 * util.characteristic_z, 3 * util.characteristic_z, 1000)
    # grid = calculate_wavepacket_grid_new(
    #     eigenstates, z_points.tolist(), x0_lim=(-4, 4), x1_lim=(-4, 4)
    # )
    path = get_data_path("hcp_wavepacket.json")
    # save_wavepacket_grid(grid, path)
    return load_wavepacket_grid(path)


class OverlapTransform(TypedDict):
    dkx0: Tuple[float, float]
    dkx1: Tuple[float, float]
    dkz: float
    points: NDArray


def save_overlap_transform(path: Path, transform: OverlapTransform):
    np.savez(path, **transform)


def load_overlap_transform(path: Path) -> OverlapTransform:
    loaded = np.load(path)
    return {
        "dkx0": (loaded["dkx0"][0], loaded["dkx0"][1]),
        "dkx1": (loaded["dkx1"][0], loaded["dkx1"][1]),
        "dkz": loaded["dkz"],
        "points": loaded["points"],
    }


def calculate_overlap_factor():
    wavepacket0 = generate_fcc_wavepacket()
    points0 = np.array(wavepacket0["points"])

    N = points0.shape[0] * points0.shape[1] * points0.shape[2]
    delta_z = wavepacket0["z_points"][-1] - wavepacket0["z_points"][0]
    # 0.9989499296071063 1000 -3 3
    # 0.9994494691023869 2000 -3 3
    # 0.9989999372435083 1000 -4 4
    print(np.sum(np.square(np.abs(points0))) * delta_z / N)

    wavepacket1 = generate_hcp_wavepacket()
    points1 = np.array(wavepacket1["points"])
    # 0.9989454040074838
    print(np.sum(np.square(np.abs(points1))) * delta_z / N)
    # -2.592593651271823e-07 (should be 0)
    print(np.sum(np.conj(points1) * points0) * delta_z / N)

    util = SurfaceConfigUtil(wavepacket0)

    # Note fft adds the 1/N factor here
    product = np.conj(points1) * points0 * delta_z
    overlap_fcc_hcp = np.fft.ifftn(product, axes=(0, 1, 2))
    transform: OverlapTransform = {
        "dkx0": util.dkx0,
        "dkx1": util.dkx1,
        "dkz": 2 * np.pi / delta_z,
        "points": overlap_fcc_hcp,
    }

    path = get_data_path("overlap_transform_hcp_fcc.npz")
    save_overlap_transform(path, transform)
