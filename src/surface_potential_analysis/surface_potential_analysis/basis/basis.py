from __future__ import annotations

from typing import Any, Literal, TypeVar

from surface_potential_analysis.axis.axis import (
    FundamentalPositionAxis,
    FundamentalTransformedPositionAxis,
)
from surface_potential_analysis.axis.axis_like import (
    AxisLike,
    AxisWithLengthLike,
    AxisWithLengthLike1d,
    AxisWithLengthLike2d,
    AxisWithLengthLike3d,
)

_ND0Inv = TypeVar("_ND0Inv", bound=int)

Basis = tuple[AxisLike[Any, Any], ...]

AxisWithLengthBasis = tuple[AxisWithLengthLike[Any, Any, _ND0Inv], ...]

_A1d0_co = TypeVar("_A1d0_co", bound=AxisWithLengthLike1d[Any, Any], covariant=True)
Basis1d = tuple[_A1d0_co]

_A2d0_co = TypeVar("_A2d0_co", bound=AxisWithLengthLike2d[Any, Any], covariant=True)
_A2d1_co = TypeVar("_A2d1_co", bound=AxisWithLengthLike2d[Any, Any], covariant=True)
Basis2d = tuple[_A2d0_co, _A2d1_co]

_A3d0_co = TypeVar("_A3d0_co", bound=AxisWithLengthLike3d[Any, Any], covariant=True)
_A3d1_co = TypeVar("_A3d1_co", bound=AxisWithLengthLike3d[Any, Any], covariant=True)
_A3d2_co = TypeVar("_A3d2_co", bound=AxisWithLengthLike3d[Any, Any], covariant=True)
Basis3d = tuple[_A3d0_co, _A3d1_co, _A3d2_co]


_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)

FundamentalPositionBasis3d = tuple[
    FundamentalPositionAxis[_NF0Inv, Literal[3]],
    FundamentalPositionAxis[_NF1Inv, Literal[3]],
    FundamentalPositionAxis[_NF2Inv, Literal[3]],
]

FundamentalMomentumBasis3d = tuple[
    FundamentalTransformedPositionAxis[_NF0Inv, Literal[3]],
    FundamentalTransformedPositionAxis[_NF1Inv, Literal[3]],
    FundamentalTransformedPositionAxis[_NF2Inv, Literal[3]],
]
