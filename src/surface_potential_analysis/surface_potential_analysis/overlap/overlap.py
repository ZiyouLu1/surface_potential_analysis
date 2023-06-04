from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    Basis3d,
    FundamentalMomentumBasis3d,
    FundamentalPositionBasis3d,
)

if TYPE_CHECKING:
    from pathlib import Path

_B3d0Cov = TypeVar("_B3d0Cov", bound=Basis3d[Any, Any, Any], covariant=True)


class Overlap3d(TypedDict, Generic[_B3d0Cov]):
    """Represents the result of an overlap calculation of two wavepackets."""

    basis: _B3d0Cov
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


def save_overlap(path: Path, overlap: Overlap3d[Any]) -> None:
    """
    Save an overlap calculation to a file.

    Parameters
    ----------
    path : Path
    overlap : Overlap[Any]
    """
    np.save(path, overlap)


def load_overlap(path: Path) -> Overlap3d[Any]:
    """
    Load an overlap from a file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Overlap[Any]
    """
    return np.load(path, allow_pickle=True)[()]  # type: ignore[no-any-return]


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

FundamentalMomentumOverlap = Overlap3d[
    FundamentalMomentumBasis3d[_L0Inv, _L1Inv, _L2Inv]
]
FundamentalPositionOverlap = Overlap3d[
    FundamentalPositionBasis3d[_L0Inv, _L1Inv, _L2Inv]
]
