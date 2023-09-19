from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import (
    ExplicitBasis,
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedBasis,
    FundamentalTransformedPositionBasis,
)
from surface_potential_analysis.axis.util import BasisUtil

if TYPE_CHECKING:
    from .axis_like import (
        BasisLike,
        BasisWithLengthLike,
    )

    _NDInv = TypeVar("_NDInv", bound=int)

    _N0Inv = TypeVar("_N0Inv", bound=int)
    _N1Inv = TypeVar("_N1Inv", bound=int)

    _NF0Inv = TypeVar("_NF0Inv", bound=int)


def axis_as_fundamental_position_axis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _NDInv]
) -> FundamentalPositionBasis[_NF0Inv, _NDInv]:
    """
    Get the fundamental position axis for a given axis.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    FundamentalPositionAxis[_NF0Inv]
    """
    return FundamentalPositionBasis(axis.delta_x, axis.fundamental_n)


def axis_as_fundamental_momentum_axis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _NDInv]
) -> FundamentalTransformedPositionBasis[_NF0Inv, _NDInv]:
    """
    Get the fundamental momentum axis for a given axis.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv, _NDInv]

    Returns
    -------
    FundamentalMomentumAxis[_NF0Inv, _NDInv]
    """
    return FundamentalTransformedPositionBasis(axis.delta_x, axis.fundamental_n)


def axis_as_fundamental_transformed_axis(
    axis: BasisLike[_NF0Inv, _N0Inv]
) -> FundamentalTransformedBasis[_NF0Inv]:
    """
    Get the fundamental momentum axis for a given axis.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv, _NDInv]

    Returns
    -------
    FundamentalMomentumAxis[_NF0Inv, _NDInv]
    """
    return FundamentalTransformedBasis(axis.fundamental_n)


def axis_as_fundamental_axis(
    axis: BasisLike[_NF0Inv, _N0Inv]
) -> FundamentalBasis[_NF0Inv]:
    """
    Get the fundamental momentum axis for a given axis.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv, _NDInv]

    Returns
    -------
    FundamentalMomentumAxis[_NF0Inv, _NDInv]
    """
    return FundamentalBasis(axis.fundamental_n)


def axis_as_explicit_position_axis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _NDInv]
) -> ExplicitBasis[_NF0Inv, _N0Inv, _NDInv]:
    """
    Convert the axis into an explicit position axis.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    ExplicitAxis[_NF0Inv, _N0Inv]
    """
    util = BasisUtil(axis)
    return ExplicitBasis(axis.delta_x, util.vectors)


def axis_as_orthonormal_axis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _NDInv]
) -> ExplicitBasis[_NF0Inv, _N0Inv, _NDInv]:
    """
    make the given axis orthonormal.

    Parameters
    ----------
    axis : AxisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    ExplicitAxis[_NF0Inv, _N0Inv]
    """
    vectors = BasisUtil(axis).vectors
    orthonormal_vectors = np.zeros_like(vectors, dtype=vectors.dtype)
    for i, v in enumerate(vectors):
        vector = v
        for other in orthonormal_vectors[:i]:
            vector -= np.dot(vector, other) * other
        orthonormal_vectors[i] = vector / np.linalg.norm(vector)

    return ExplicitBasis(axis.delta_x, orthonormal_vectors)


def axis_as_n_point_axis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _NDInv], *, n: _N1Inv
) -> FundamentalPositionBasis[_N1Inv, _NDInv]:
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
    return FundamentalPositionBasis(axis.delta_x, n)


def axis_as_single_point_axis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _NDInv]
) -> FundamentalPositionBasis[Literal[1], _NDInv]:
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
