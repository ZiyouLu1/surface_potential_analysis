from __future__ import annotations

from typing import TYPE_CHECKING, Any

from surface_potential_analysis.overlap.calculation import (
    calculate_wavepacket_list_overlap,
)
from surface_potential_analysis.overlap.overlap import (
    Overlap,
    get_single_overlap,
)
from surface_potential_analysis.util.decorators import npy_cached

from .s4_wavepacket import (
    get_wannier90_localized_split_bands_wavepacket_hydrogen,
)
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.overlap.overlap import SingleOverlap


def _fundamental_overlap_cache_hydrogen(
    offset_j: tuple[int, int] = (0, 0),
) -> Path | None:
    return get_data_path(f"overlap/overlap_{offset_j[0]}_{offset_j[1]}.npy")


@npy_cached(_fundamental_overlap_cache_hydrogen, load_pickle=True)
def _get_fundamental_overlaps_hydrogen(
    offset_j: tuple[int, int] = (0, 0),
) -> Overlap[
    StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
    FundamentalBasis[int],
    FundamentalBasis[int],
]:
    wavepackets = get_wannier90_localized_split_bands_wavepacket_hydrogen()
    (n0, n1, _) = wavepackets["basis"][1].shape
    shift = (offset_j[0] * n0, offset_j[1] * n1, 0)
    return calculate_wavepacket_list_overlap(wavepackets, shift=shift)


def _get_overlap_inner_hydrogen(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> SingleOverlap[StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]]:
    return get_single_overlap(
        _get_fundamental_overlaps_hydrogen(
            offset_j=(offset_j[0] - offset_i[0], offset_j[1] - offset_i[1])
        ),
        (i, j),
    )


def get_overlap_hydrogen(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> SingleOverlap[StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]]:
    offset_i, offset_j = (offset_i, offset_j) if i < j else (offset_j, offset_i)
    i, j = (i, j) if i < j else (j, i)
    return _get_overlap_inner_hydrogen(i, j, offset_i, offset_j)
