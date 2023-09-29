from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    TruncatedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.overlap.calculation import (
    calculate_wavepacket_list_overlap,
)
from surface_potential_analysis.overlap.overlap import (
    Overlap,
    SingleOverlap,
    get_single_overlap,
)
from surface_potential_analysis.util.decorators import npy_cached_dict
from surface_potential_analysis.wavepacket.wavepacket import (
    get_unfurled_basis,
    get_wavepacket_basis,
)

from .s4_wavepacket import get_wannier90_localized_wavepacket_hydrogen
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path


def _fundamental_overlap_cache_hydrogen(
    offset_j: tuple[int, int] = (0, 0),
) -> Path | None:
    return get_data_path(f"overlap/overlap_{offset_j[0]}_{offset_j[1]}.npy")


@npy_cached_dict(_fundamental_overlap_cache_hydrogen, load_pickle=True)
def _get_fundamental_overlaps_hydrogen(
    offset_j: tuple[int, int] = (0, 0),
) -> Overlap[
    StackedBasisLike[*tuple[TruncatedPositionBasis[Any, Any, Any], ...]],
    FundamentalBasis[int],
    FundamentalBasis[int],
]:
    wavepackets = get_wannier90_localized_wavepacket_hydrogen(8)
    (n0, n1, n2) = wavepackets["basis"][1].fundamental_shape
    shift = (offset_j[0] * n0, offset_j[1] * n1, 0)
    basis = StackedBasis(
        *tuple(
            TruncatedPositionBasis[Any, Any, Literal[3]](b.delta_x, n, b.fundamental_n)
            for (b, n) in zip(
                get_unfurled_basis(get_wavepacket_basis(wavepackets)),
                (5 * n0, 5 * n1, n2),
                strict=True,
            )
        )
    )
    return calculate_wavepacket_list_overlap(wavepackets, shift=shift, basis=basis)


def _get_overlap_inner_hydrogen(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> SingleOverlap[
    StackedBasisLike[*tuple[TruncatedPositionBasis[Any, Any, Any], ...]]
]:
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
) -> SingleOverlap[
    StackedBasisLike[*tuple[TruncatedPositionBasis[Any, Any, Any], ...]]
]:
    offset_i, offset_j = (offset_i, offset_j) if i < j else (offset_j, offset_i)
    i, j = (i, j) if i < j else (j, i)
    return _get_overlap_inner_hydrogen(i, j, offset_i, offset_j)
