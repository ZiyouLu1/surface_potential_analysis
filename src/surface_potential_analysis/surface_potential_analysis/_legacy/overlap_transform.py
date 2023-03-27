from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from .surface_config import SurfaceConfigUtil
from .wavepacket_grid import WavepacketGrid


class OverlapTransform(TypedDict):
    dkx0: tuple[float, float]
    dkx1: tuple[float, float]
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


def calculate_overlap_transform(
    grid0: WavepacketGrid, grid1: WavepacketGrid
) -> OverlapTransform:
    util = SurfaceConfigUtil(grid0)
    delta_z = grid0["z_points"][-1] - grid0["z_points"][0]

    # Note fft adds the 1/N factor here
    product = np.conj(grid0["points"]) * np.asarray(grid1["points"]) * delta_z

    shifted = np.fft.ifftshift(product)
    overlap = np.fft.ifftn(shifted, axes=(0, 1, 2))

    return {
        "dkx0": util.dkx0,
        "dkx1": util.dkx1,
        "dkz": 2 * np.pi / delta_z,
        "points": overlap,
    }
