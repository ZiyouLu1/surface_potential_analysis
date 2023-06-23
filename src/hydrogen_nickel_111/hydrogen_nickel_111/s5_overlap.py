from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from surface_potential_analysis.overlap.calculation import calculate_wavepacket_overlap
from surface_potential_analysis.overlap.overlap import Overlap3d, save_overlap
from surface_potential_analysis.util.decorators import npy_cached

from .s4_wavepacket import (
    get_two_point_normalized_wavepacket,
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


def _overlap_inner_cache(i: int, j: int, offset: tuple[int, int] = (0, 0)) -> Path:
    dx0, dx1 = offset
    return get_data_path(f"overlap/overlap_{i}_{j}_{dx0}_{dx1}.npy")


@npy_cached(_overlap_inner_cache)
def _get_overlap_inner(
    i: int, j: int, offset: tuple[int, int] = (0, 0)
) -> Overlap3d[FundamentalPositionBasis3d[int, int, Literal[250]]]:
    wavepacket_i = get_two_point_normalized_wavepacket(i)
    wavepacket_j = get_two_point_normalized_wavepacket(j, offset)
    return calculate_wavepacket_overlap(wavepacket_i, wavepacket_j)


def get_overlap(
    i: int, j: int, offset: tuple[int, int] = (0, 0)
) -> Overlap3d[FundamentalPositionBasis3d[int, int, Literal[250]]]:
    dx0, dx1 = offset
    i, j = (i, j) if i < j else (j, i)
    dx0, dx1 = (dx0 % 3, dx1 % 3) if i < j else ((-dx0) % 3, (-dx1) % 3)
    match (dx0, dx1):
        case (0, 2) | (1, 0) | (1, 2):
            dx0, dx1 = dx1, dx0
    return _get_overlap_inner(i, j, (dx0, dx1))
