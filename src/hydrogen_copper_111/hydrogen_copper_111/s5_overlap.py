from __future__ import annotations

from typing import TYPE_CHECKING

from surface_potential_analysis.overlap.calculation import calculate_wavepacket_overlap
from surface_potential_analysis.overlap.overlap import (
    Overlap3dBasis,
    get_overlap_cache_filename,
)
from surface_potential_analysis.util.decorators import npy_cached

from .s4_wavepacket import get_two_point_normalized_wavepacket_hydrogen
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.overlap.overlap import Overlap3d


def _overlap_inner_cache_hydrogen(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Path | None:
    filename = get_overlap_cache_filename(i, j, offset_i, offset_j)
    if filename is None:
        return None
    return get_data_path(filename)


@npy_cached(_overlap_inner_cache_hydrogen, load_pickle=True)
def _get_overlap_inner_hydrogen(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Overlap3d[Overlap3dBasis]:
    wavepacket_i = get_two_point_normalized_wavepacket_hydrogen(i, offset_i)
    wavepacket_j = get_two_point_normalized_wavepacket_hydrogen(j, offset_j)
    return calculate_wavepacket_overlap(wavepacket_i, wavepacket_j)  # type: ignore[return-value]


def get_overlap_hydrogen(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Overlap3d[Overlap3dBasis]:
    offset_i, offset_j = (offset_i, offset_j) if i < j else (offset_j, offset_i)
    i, j = (i, j) if i < j else (j, i)
    return _get_overlap_inner_hydrogen(i, j, offset_i, offset_j)
