from __future__ import annotations

from functools import cached_property
from typing import Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import (
    ExplicitAxis3d,
    FundamentalMomentumAxis,
    FundamentalPositionAxis,
)

from .axis_like import AxisLike, AxisLike3d, AxisVector3d

_A3d0Inv = TypeVar("_A3d0Inv", bound=AxisLike3d[Any, Any])

_NDInv = TypeVar("_NDInv", bound=int)

_N0Inv = TypeVar("_N0Inv", bound=int)
_N1Inv = TypeVar("_N1Inv", bound=int)

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)


class _RotatedAxis(AxisLike3d[_NF0Inv, _N0Inv]):
    def __init__(
        self,
        axis: AxisLike3d[_NF0Inv, _N0Inv],
        matrix: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]],
    ) -> None:
        self._axis = axis
        self._matrix = matrix
        ##TODO: dunder methods

    def __getattr__(self, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa: ANN204, ANN002, ANN003
        return getattr(self._axis, *args, **kwargs)

    @cached_property
    def delta_x(self) -> AxisVector3d:
        return np.dot(self._matrix, self._axis.delta_x)  # type: ignore[no-any-return]

    @property
    def n(self) -> _N0Inv:
        return self._axis.n

    @property
    def fundamental_n(self) -> _NF0Inv:
        return self._axis.fundamental_n

    @property
    def vectors(self) -> np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]]:
        return self._axis.vectors


def get_rotated_axis(
    axis: _A3d0Inv,
    matrix: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]],
) -> _A3d0Inv:
    """
    Get the axis rotated by the given matrix.

    Parameters
    ----------
    axis : _A3d0Inv
    matrix : np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]]

    Returns
    -------
    _A3d0Inv
        The rotated axis
    """
    return _RotatedAxis(axis, matrix)  # type: ignore[return-value]


def get_axis_conversion_matrix(
    axis_0: AxisLike[_N0Inv, _NF0Inv, _NDInv], axis_1: AxisLike[_N1Inv, _NF1Inv, _NDInv]
) -> np.ndarray[tuple[_NF0Inv, _NF1Inv], np.dtype[np.complex_]]:
    """
    Get the matrix to convert one set of axis axes into another.

    Parameters
    ----------
    axis_0 : AxisLike[_N0Inv, _NF0Inv]
    axis_1 : AxisLike[_N1Inv, _NF1Inv]

    Returns
    -------
    np.ndarray[tuple[_NF0Inv, _NF1Inv], np.dtype[np.complex_]]
    """
    vectors_0 = axis_0.vectors
    vectors_1 = axis_1.vectors
    return np.dot(vectors_0, np.conj(vectors_1).T)  # type: ignore[no-any-return]


def axis_as_fundamental_position_axis(
    axis: AxisLike[_NF0Inv, _N0Inv, _NDInv]
) -> FundamentalPositionAxis[_NF0Inv, _NDInv]:
    """
    Get the fundamental position axis for a given axis.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    FundamentalPositionAxis[_NF0Inv]
    """
    return FundamentalPositionAxis(axis.delta_x, axis.fundamental_n)


def axis_as_fundamental_momentum_axis(
    axis: AxisLike[_NF0Inv, _N0Inv, _NDInv]
) -> FundamentalMomentumAxis[_NF0Inv, _NDInv]:
    """
    Get the fundamental momentum axis for a given axis.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv, _NDInv]

    Returns
    -------
    FundamentalMomentumAxis[_NF0Inv, _NDInv]
    """
    return FundamentalMomentumAxis(axis.delta_x, axis.fundamental_n)


def axis_as_explicit_position_axis(
    axis: AxisLike3d[_NF0Inv, _N0Inv]
) -> ExplicitAxis3d[_NF0Inv, _N0Inv]:
    """
    Convert the axis into an explicit position axis.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    ExplicitAxis[_NF0Inv, _N0Inv]
    """
    return ExplicitAxis3d(axis.delta_x, axis.vectors)


def axis_as_n_point_axis(
    axis: AxisLike[_NF0Inv, _N0Inv, _NDInv], *, n: _N1Inv
) -> FundamentalPositionAxis[_N1Inv, _NDInv]:
    """
    Get the corresponding n point axis for a given axis.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv, _NDInv]
    n : _N1Inv

    Returns
    -------
    FundamentalPositionAxis[_N1Inv, _NDInv]
    """
    return FundamentalPositionAxis(axis.delta_x, n)


def axis_as_single_point_axis(
    axis: AxisLike[_NF0Inv, _N0Inv, _NDInv]
) -> FundamentalPositionAxis[Literal[1], _NDInv]:
    """
    Get the corresponding single point axis for a given axis.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv, _NDInv]

    Returns
    -------
    FundamentalPositionAxis[Literal[1], _NDInv]
    """
    return axis_as_n_point_axis(axis, n=1)
