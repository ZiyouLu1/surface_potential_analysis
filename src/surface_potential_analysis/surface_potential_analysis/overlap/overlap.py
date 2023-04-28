from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    MomentumBasisConfig,
    PositionBasisConfig,
)
from surface_potential_analysis.interpolation import pad_ft_points

if TYPE_CHECKING:
    from pathlib import Path

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
    np.save(path, overlap)


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
    return np.load(path, allow_pickle=True)[()]  # type:ignore[no-any-return]


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

OverlapTransform = Overlap[MomentumBasisConfig[_L0Inv, _L1Inv, _L2Inv]]


def convert_overlap_momentum_basis(
    overlap: Overlap[PositionBasisConfig[_L0Inv, _L1Inv, _L2Inv]]
) -> OverlapTransform[_L0Inv, _L1Inv, _L2Inv]:
    """
    Convert an overlap from position basis to momentum.

    Parameters
    ----------
    overlap : Overlap[PositionBasisConfig[_L0Inv, _L1Inv, _L2Inv]]

    Returns
    -------
    OverlapTransform[_L0Inv, _L1Inv, _L2Inv]
    """
    util = BasisConfigUtil(overlap["basis"])
    transformed = np.fft.ifftn(
        overlap["vector"].reshape(util.shape),
        axes=(0, 1, 2),
        s=util.fundamental_shape,
        norm="forward",
    )
    flattened = transformed.reshape(-1)

    return {
        "basis": (
            {
                "_type": "momentum",
                "delta_x": util.delta_x0,
                "n": util.fundamental_n0,  # type: ignore[typeddict-item]
            },
            {
                "_type": "momentum",
                "delta_x": util.delta_x1,
                "n": util.fundamental_n1,  # type: ignore[typeddict-item]
            },
            {
                "_type": "momentum",
                "delta_x": util.delta_x2,
                "n": util.fundamental_n2,  # type: ignore[typeddict-item]
            },
        ),
        "vector": flattened,
    }


def convert_overlap_position_basis(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv]
) -> Overlap[PositionBasisConfig[_L0Inv, _L1Inv, _L2Inv]]:
    """
    Convert an overlap from momentum basis to position.

    Parameters
    ----------
    overlap : OverlapTransform[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Overlap[PositionBasisConfig[_L0Inv, _L1Inv, _L2Inv]]
    """
    util = BasisConfigUtil(overlap["basis"])
    padded = pad_ft_points(
        overlap["vector"].reshape(util.shape),
        s=util.fundamental_shape,
        axes=(0, 1, 2),
    )
    transformed = np.fft.fftn(padded, axes=(0, 1, 2), s=util.fundamental_shape)
    flattened = transformed.reshape(-1)

    return {
        "basis": (
            {
                "_type": "position",
                "delta_x": util.delta_x0,
                "n": util.fundamental_n0,  # type: ignore[typeddict-item]
            },
            {
                "_type": "position",
                "delta_x": util.delta_x1,
                "n": util.fundamental_n1,  # type: ignore[typeddict-item]
            },
            {
                "_type": "position",
                "delta_x": util.delta_x2,
                "n": util.fundamental_n2,  # type: ignore[typeddict-item]
            },
        ),
        "vector": flattened,
    }
