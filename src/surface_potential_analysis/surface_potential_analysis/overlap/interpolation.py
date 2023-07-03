from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar, Unpack

import numpy as np

from surface_potential_analysis.axis.conversion import axis_as_single_point_axis
from surface_potential_analysis.basis.util import (
    BasisUtil,
    wrap_index_around_origin,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.overlap.overlap import FundamentalPositionOverlap

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

ArrayStackedIndexFractionLike = tuple[
    np.ndarray[_S0Inv, np.dtype[np.float_ | np.int_]],
    np.ndarray[_S0Inv, np.dtype[np.float_ | np.int_]],
    np.ndarray[_S0Inv, np.dtype[np.float_ | np.int_]],
]


def get_overlap_momentum_interpolator_k_fractions(
    overlap: FundamentalPositionOverlap[_L0Inv, _L1Inv, _L2Inv]
) -> Callable[
    [ArrayStackedIndexFractionLike[_S0Inv]],
    np.ndarray[_S0Inv, np.dtype[np.complex_]],
]:
    """
    Get an interpolator to calculate the overlap in momentum basis.

    Parameters
    ----------
    overlap : OverlapPosition[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Callable[[np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]], np.ndarray[_S0Inv, np.dtype[np.complex_]], ]
        Interpolator which takes a list of coordinates as fractions of k0, k1, k2 index and returns the
        overlap at this point
    """
    util = BasisUtil(overlap["basis"])
    nx_points = util.nx_points
    nx_points_wrapped = wrap_index_around_origin(
        overlap["basis"], nx_points, axes=(0, 1)
    )
    x_fractions = np.asarray(nx_points_wrapped, dtype=float)[:, :, np.newaxis]
    x_fractions[0] /= util.n0  # type: ignore[misc]
    x_fractions[1] /= util.n1  # type: ignore[misc]
    x_fractions[2] /= util.n2  # type: ignore[misc]

    def _interpolator(
        k_fractions: ArrayStackedIndexFractionLike[_S0Inv],
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        phi = np.sum(x_fractions * np.asarray(k_fractions)[:, np.newaxis, :], axis=0)
        return np.sum(overlap["vector"][:, np.newaxis] * np.exp(2j * np.pi * phi), axis=0)  # type: ignore[no-any-return]

    return _interpolator


def get_overlap_momentum_interpolator(
    overlap: FundamentalPositionOverlap[_L0Inv, _L1Inv, _L2Inv]
) -> Callable[
    [np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]],
    np.ndarray[_S0Inv, np.dtype[np.complex_]],
]:
    """
    Given the overlap create an interpolator to calculate th interpolation in momentum basis.

    Parameters
    ----------
    overlap : OverlapPosition[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Callable[[np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]], np.ndarray[_S0Inv, np.dtype[np.complex_]], ]
        Interpolator, which takes a coordinate list on momentum basis
    """
    util = BasisUtil(overlap["basis"])
    nx_points = util.nx_points
    nx_points_wrapped = wrap_index_around_origin(
        overlap["basis"], nx_points, axes=(0, 1)
    )
    x_points = util.get_x_points_at_index(nx_points_wrapped)[:, :, np.newaxis]

    def _interpolator(
        k_coordinates: np.ndarray[
            tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]
        ]
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        phi = np.sum(x_points * k_coordinates[:, np.newaxis, :], axis=0)
        return np.sum(overlap["vector"][:, np.newaxis] * np.exp(1j * phi), axis=0)  # type: ignore[no-any-return]

    return _interpolator


def get_overlap_momentum_interpolator_flat(
    overlap: FundamentalPositionOverlap[_L0Inv, _L1Inv, _L2Inv]
) -> Callable[
    [np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]],
    np.ndarray[_S0Inv, np.dtype[np.complex_]],
]:
    """
    Given the overlap create an interpolator to calculate the interpolation in momentum basis.

    This only works for kx2=0, and makes the assumption that x2 lies completely perpendicular to
    the x01 direction.

    Parameters
    ----------
    overlap : OverlapPosition[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Callable[[np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]], np.ndarray[_S0Inv, np.dtype[np.complex_]], ]
        Interpolator, which takes a coordinate list in momentum basis ignoring k2 axis
    """
    basis = (*overlap["basis"][0:2], axis_as_single_point_axis(overlap["basis"][2]))
    util = BasisUtil(basis)
    nx_points = util.nx_points
    nx_points_wrapped = wrap_index_around_origin(
        overlap["basis"], nx_points, axes=(0, 1)
    )
    x_points = util.get_x_points_at_index(nx_points_wrapped)[:2, :, np.newaxis]

    vector = overlap["vector"].reshape(BasisUtil(overlap["basis"]).shape)
    vector_transformed = np.fft.ifft(vector, axis=2, norm="forward")[:, :, 0].ravel()

    def _interpolator(
        k_coordinates: np.ndarray[
            tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]
        ]
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        phi = np.sum(x_points * k_coordinates[:, np.newaxis, :], axis=0)
        return np.sum(vector_transformed[:, np.newaxis] * np.exp(1j * phi), axis=0)  # type: ignore[no-any-return]

    return _interpolator