from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

from surface_potential_analysis.basis.basis import (
    Basis3d,
    FundamentalMomentumBasis3d,
    FundamentalPositionBasis3d,
)

if TYPE_CHECKING:
    import numpy as np

    pass

_B3d0_co = TypeVar("_B3d0_co", bound=Basis3d[Any, Any, Any], covariant=True)


class Overlap3d(TypedDict, Generic[_B3d0_co]):
    """Represents the result of an overlap calculation of two wavepackets."""

    basis: _B3d0_co
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

FundamentalMomentumOverlap = Overlap3d[
    FundamentalMomentumBasis3d[_L0Inv, _L1Inv, _L2Inv]
]
FundamentalPositionOverlap = Overlap3d[
    FundamentalPositionBasis3d[_L0Inv, _L1Inv, _L2Inv]
]


def get_overlap_cache_filename(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> str | None:
    """
    Get filename for overlap cache.

    Parameters
    ----------
    i : int
    j : int
    offset_i : tuple[int, int], optional
        offset_i, by default (0, 0)
    offset_j : tuple[int, int], optional
        offset_j, by default (0, 0)

    Returns
    -------
    tuple[int, int, tuple[int, int], tuple[int, int]]
        _description_
    """
    dx0i, dx1i = offset_i
    dx0j, dx1j = offset_j
    too_large_i = not (-2 < dx0i < 2 and -2 < dx1i < 2)  # noqa: PLR2004
    too_large_j = not (-2 < dx0j < 2 and -2 < dx1j < 2)  # noqa: PLR2004
    if too_large_i or too_large_j or i > 1 or j > 1:
        return None
    dx0i, dx1i = dx0i % 3, dx1i % 3
    dx0j, dx1j = dx0j % 3, dx1j % 3
    return f"overlap/overlap_{i}_{dx0i}_{dx1i}_{j}_{dx0j}_{dx1j}.npy"
