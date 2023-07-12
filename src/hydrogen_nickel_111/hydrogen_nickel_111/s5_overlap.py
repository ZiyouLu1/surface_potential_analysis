from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from surface_potential_analysis.overlap.calculation import calculate_wavepacket_overlap
from surface_potential_analysis.util.decorators import npy_cached

from .s4_wavepacket import (
    get_two_point_normalized_wavepacket_deuterium,
    get_two_point_normalized_wavepacket_hydrogen,
)
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.basis.basis import FundamentalPositionBasis3d
    from surface_potential_analysis.overlap.overlap import Overlap3d


def _overlap_inner_cache_hydrogen(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Path | None:
    dx0i, dx1i = offset_i
    dx0j, dx1j = offset_j
    too_large_i = not (-2 < dx0i < 2 and -2 < dx1i < 2)  # noqa: PLR2004
    too_large_j = not (-2 < dx0j < 2 and -2 < dx1j < 2)  # noqa: PLR2004
    if too_large_i or too_large_j or i > 1 or j > 1:
        print("Unable to cache", offset_j, offset_i)  # noqa: T201
        return None
    dx0i, dx1i = dx0i % 3, dx1i % 3
    dx0j, dx1j = dx0j % 3, dx1j % 3
    return get_data_path(f"overlap/overlap_{i}_{dx0i}_{dx1i}_{j}_{dx0j}_{dx1j}.npy")


@npy_cached(_overlap_inner_cache_hydrogen, load_pickle=True)
def _get_overlap_inner_hydrogen(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Overlap3d[FundamentalPositionBasis3d[int, int, Literal[250]]]:
    wavepacket_i = get_two_point_normalized_wavepacket_hydrogen(i, offset_i)
    wavepacket_j = get_two_point_normalized_wavepacket_hydrogen(j, offset_j)
    return calculate_wavepacket_overlap(wavepacket_i, wavepacket_j)  # type: ignore[return-value]


def get_overlap_hydrogen(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Overlap3d[FundamentalPositionBasis3d[int, int, Literal[250]]]:
    offset_i, offset_j = (offset_i, offset_j) if i < j else (offset_j, offset_i)
    i, j = (i, j) if i < j else (j, i)
    return _get_overlap_inner_hydrogen(i, j, offset_i, offset_j)


def _overlap_inner_cache_deuterium(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Path | None:
    dx0i, dx1i = offset_i
    dx0j, dx1j = offset_j
    too_large_i = not (-2 < dx0i < 2 and -2 < dx1i < 2)  # noqa: PLR2004
    too_large_j = not (-2 < dx0j < 2 and -2 < dx1j < 2)  # noqa: PLR2004
    if too_large_i or too_large_j or i > 1 or j > 1:
        print("Unable to cache", offset_j, offset_i)  # noqa: T201
        return None
    dx0i, dx1i = dx0i % 3, dx1i % 3
    dx0j, dx1j = dx0j % 3, dx1j % 3
    return get_data_path(
        f"overlap/overlap_deuterium_{i}_{dx0i}_{dx1i}_{j}_{dx0j}_{dx1j}.npy"
    )


@npy_cached(_overlap_inner_cache_deuterium, load_pickle=True)
def _get_overlap_inner_deuterium(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Overlap3d[FundamentalPositionBasis3d[int, int, Literal[200]]]:
    wavepacket_i = get_two_point_normalized_wavepacket_deuterium(i, offset_i)
    wavepacket_j = get_two_point_normalized_wavepacket_deuterium(j, offset_j)
    return calculate_wavepacket_overlap(wavepacket_i, wavepacket_j)  # type: ignore[return-value]


def get_overlap_deuterium(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> Overlap3d[FundamentalPositionBasis3d[int, int, Literal[200]]]:
    offset_i, offset_j = (offset_i, offset_j) if i < j else (offset_j, offset_i)
    i, j = (i, j) if i < j else (j, i)
    return _get_overlap_inner_deuterium(i, j, offset_i, offset_j)
