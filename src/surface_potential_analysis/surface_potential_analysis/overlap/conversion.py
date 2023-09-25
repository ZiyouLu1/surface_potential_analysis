from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.overlap.overlap import Overlap3d

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)


def convert_overlap_to_momentum_basis(
    overlap: Overlap3d[
        StackedBasisLike[
            FundamentalPositionBasis[_L0Inv, Literal[3]],
            FundamentalPositionBasis[_L1Inv, Literal[3]],
            FundamentalPositionBasis[_L2Inv, Literal[3]],
        ]
    ]
) -> Overlap3d[
    StackedBasisLike[
        FundamentalTransformedPositionBasis[_L0Inv, Literal[3]],
        FundamentalTransformedPositionBasis[_L1Inv, Literal[3]],
        FundamentalTransformedPositionBasis[_L2Inv, Literal[3]],
    ]
]:
    """
    Convert an overlap from position basis to momentum.

    Parameters
    ----------
    overlap : Overlap[PositionStackedBasisLike[tuple[_L0Inv, _L1Inv, _L2Inv]]

    Returns
    -------
    OverlapMomentum[_L0Inv, _L1Inv, _L2Inv]
    """
    transformed = np.fft.ifftn(
        overlap["data"].reshape(overlap["basis"].shape),
        axes=(0, 1, 2),
        s=overlap["basis"].fundamental_shape,
        norm="forward",
    )
    flattened = transformed.reshape(-1)

    return {  # type: ignore fails to infer return type
        "basis": stacked_basis_as_fundamental_momentum_basis(overlap["basis"]),
        "data": flattened,
    }


def convert_overlap_to_position_basis(
    overlap: Overlap3d[
        StackedBasisLike[
            FundamentalTransformedPositionBasis[_L0Inv, Literal[3]],
            FundamentalTransformedPositionBasis[_L1Inv, Literal[3]],
            FundamentalTransformedPositionBasis[_L2Inv, Literal[3]],
        ]
    ]
) -> Overlap3d[
    StackedBasisLike[
        FundamentalPositionBasis[_L0Inv, Literal[3]],
        FundamentalPositionBasis[_L1Inv, Literal[3]],
        FundamentalPositionBasis[_L2Inv, Literal[3]],
    ]
]:
    """
    Convert an overlap from momentum basis to position.

    Parameters
    ----------
    overlap : OverlapMomentum[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Overlap[PositionStackedBasisLike[tuple[_L0Inv, _L1Inv, _L2Inv]]
    """
    padded = pad_ft_points(
        overlap["data"].reshape(overlap["basis"].shape),
        s=overlap["basis"].fundamental_shape,
        axes=(0, 1, 2),
    )
    transformed = np.fft.fftn(
        padded, axes=(0, 1, 2), s=overlap["basis"].fundamental_shape
    )
    flattened = transformed.reshape(-1)
    return {  # type: ignore  fails to infer type
        "basis": stacked_basis_as_fundamental_position_basis(overlap["basis"]),
        "data": flattened,
    }
