from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalMomentumBasis,
    FundamentalPositionBasis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis_config.basis_config import (
        FundamentalMomentumBasisConfig,
        FundamentalPositionBasisConfig,
    )


_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)


def build_position_basis_config_from_resolution(
    resolution: tuple[_NF0Inv, _NF1Inv, _NF2Inv],
    delta_x: tuple[
        np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
        np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
        np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    ]
    | None = None,
) -> FundamentalPositionBasisConfig[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    Given a resolution and a set of directions construct a FundamentalPositionBasisConfig.

    Parameters
    ----------
    resolution : tuple[_NF0Inv, _NF1Inv, _NF2Inv]
        resolution of the basis
    delta_x : tuple[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], ] | None, optional
        vectors for the basis, by default (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

    Returns
    -------
    FundamentalPositionBasisConfig[_NF0Inv, _NF1Inv, _NF2Inv]
    """
    delta_x = (
        (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
        if delta_x is None
        else delta_x
    )
    return (
        FundamentalPositionBasis(delta_x[0], resolution[0]),
        FundamentalPositionBasis(delta_x[1], resolution[1]),
        FundamentalPositionBasis(delta_x[2], resolution[2]),
    )


def build_momentum_basis_config_from_resolution(
    resolution: tuple[_NF0Inv, _NF1Inv, _NF2Inv],
    delta_x: tuple[
        np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
        np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
        np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    ]
    | None = None,
) -> FundamentalMomentumBasisConfig[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    Given a resolution and a set of directions construct a FundamentalMomentumBasisConfig.

    Parameters
    ----------
    resolution : tuple[_NF0Inv, _NF1Inv, _NF2Inv]
        resolution of the basis
    delta_x : tuple[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], ] | None, optional
        vectors for the basis, by default (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

    Returns
    -------
    FundamentalMomentumBasisConfig[_NF0Inv, _NF1Inv, _NF2Inv]
    """
    delta_x = (
        (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
        if delta_x is None
        else delta_x
    )
    return (
        FundamentalMomentumBasis(delta_x[0], resolution[0]),
        FundamentalMomentumBasis(delta_x[1], resolution[1]),
        FundamentalMomentumBasis(delta_x[2], resolution[2]),
    )
