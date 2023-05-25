from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from surface_potential_analysis.basis_config.conversion import (
    basis_config_as_fundamental_momentum_basis_config,
    basis_config_as_fundamental_position_basis_config,
)
from surface_potential_analysis.basis_config.util import (
    BasisConfigUtil,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

if TYPE_CHECKING:
    from .overlap import FundamentalMomentumOverlap, FundamentalPositionOverlap


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def convert_overlap_to_momentum_basis(
    overlap: FundamentalPositionOverlap[_L0Inv, _L1Inv, _L2Inv]
) -> FundamentalMomentumOverlap[_L0Inv, _L1Inv, _L2Inv]:
    """
    Convert an overlap from position basis to momentum.

    Parameters
    ----------
    overlap : Overlap[PositionBasisConfig[_L0Inv, _L1Inv, _L2Inv]]

    Returns
    -------
    OverlapMomentum[_L0Inv, _L1Inv, _L2Inv]
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
        "basis": basis_config_as_fundamental_momentum_basis_config(overlap["basis"]),
        "vector": flattened,
    }


def convert_overlap_to_position_basis(
    overlap: FundamentalMomentumOverlap[_L0Inv, _L1Inv, _L2Inv]
) -> FundamentalPositionOverlap[_L0Inv, _L1Inv, _L2Inv]:
    """
    Convert an overlap from momentum basis to position.

    Parameters
    ----------
    overlap : OverlapMomentum[_L0Inv, _L1Inv, _L2Inv]

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
        "basis": basis_config_as_fundamental_position_basis_config(overlap["basis"]),
        "vector": flattened,
    }
