from pathlib import Path
from typing import Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    MomentumBasisConfig,
)

_BC0Cov = TypeVar("_BC0Cov", bound=BasisConfig[Any, Any, Any], covariant=True)


class Overlap(TypedDict, Generic[_BC0Cov]):
    """Represents the result of an overlap calculation of two wavepackets."""

    basis: _BC0Cov
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


def save_overlap(path: Path, overlap: Overlap[Any]) -> None:
    """
    Save an overlap calculation to a file.

    Parameters
    ----------
    path : Path
    overlap : Overlap[Any]
    """
    state = np.array(overlap, dtype=dict)
    np.save(path, state)


def load_overlap(path: Path) -> Overlap[Any]:
    """
    Load an overlap from a file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Overlap[Any]
    """
    return np.load(path)[()]  # type:ignore[no-any-return]


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

OverlapTransform = Overlap[MomentumBasisConfig[_L0Inv, _L1Inv, _L2Inv]]