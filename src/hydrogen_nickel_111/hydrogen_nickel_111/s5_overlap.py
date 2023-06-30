from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from surface_potential_analysis.overlap.calculation import calculate_wavepacket_overlap
from surface_potential_analysis.overlap.overlap import Overlap3d, save_overlap
from surface_potential_analysis.util.decorators import npy_cached

from .s4_wavepacket import (
    get_two_point_normalized_wavepacket,
    get_wavepacket,
)
from .s4_wavepacket import (
    load_two_point_normalized_nickel_wavepacket_momentum as load_wavepacket,
)
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.basis.basis import FundamentalPositionBasis3d


def calculate_overlap_nickel() -> None:
    for i in range(6):
        for j in range(i, 6):
            for dx0, dx1 in [(-1, -1), (-1, 0), (-1, 1), (0, 0), (0, 1), (1, 1)]:
                wavepacket_i = load_wavepacket(i)
                wavepacket_j = load_wavepacket(j, (dx0, dx1))
                overlap_ij = calculate_wavepacket_overlap(wavepacket_i, wavepacket_j)
                path = get_data_path(f"overlap/overlap_{i}_{j}_{dx0 % 3}_{dx1 % 3}.npy")
                save_overlap(path, overlap_ij)


def _overlap_inner_cache(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Path | None:
    dx0i, dx1i = offset_i
    dx0j, dx1j = offset_j
    too_large_i = not (-2 < dx0i < 2 and -2 < dx1i < 2)  # noqa: PLR2004
    too_large_j = not (-2 < dx0j < 2 and -2 < dx1j < 2)  # noqa: PLR2004
    if too_large_i or too_large_j:
        print("Unable to cache", offset_j, offset_i)
        return None
    dx0i, dx1i = dx0i % 3, dx1i % 3
    dx0j, dx1j = dx0j % 3, dx1j % 3
    return get_data_path(f"overlap/overlap_{i}_{dx0i}_{dx1i}_{j}_{dx0j}_{dx1j}.npy")


@npy_cached(_overlap_inner_cache, load_pickle=True)
def _get_overlap_inner(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Overlap3d[FundamentalPositionBasis3d[int, int, Literal[250]]]:
    wavepacket_i = get_two_point_normalized_wavepacket(i, offset_i)
    wavepacket_j = get_two_point_normalized_wavepacket(j, offset_j)
    return calculate_wavepacket_overlap(wavepacket_i, wavepacket_j)


def get_overlap(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Overlap3d[FundamentalPositionBasis3d[int, int, Literal[250]]]:
    i, j = (i, j) if i < j else (j, i)
    offset_i, offset_j = (offset_i, offset_j) if i < j else (offset_j, offset_i)
    return _get_overlap_inner(i, j, offset_i, offset_j)


def get_fcc_hcp_energy_difference() -> np.float_:
    fcc = get_wavepacket(0)
    hcp = get_wavepacket(1)
    return np.average(fcc["energies"]) - np.average(hcp["energies"])
