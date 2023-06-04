from __future__ import annotations

from typing import Any, TypeVar

from surface_potential_analysis.axis.axis import (
    FundamentalMomentumAxis3d,
    FundamentalPositionAxis3d,
)
from surface_potential_analysis.axis.axis_like import (
    AxisLike,
    AxisLike1d,
    AxisLike2d,
    AxisLike3d,
)

_ND0Inv = TypeVar("_ND0Inv", bound=int)
Basis = tuple[AxisLike[Any, Any, _ND0Inv], ...]

_A1d0Cov = TypeVar("_A1d0Cov", bound=AxisLike1d[Any, Any], covariant=True)
Basis1d = tuple[_A1d0Cov]

_A2d0Cov = TypeVar("_A2d0Cov", bound=AxisLike2d[Any, Any], covariant=True)
_A2d1Cov = TypeVar("_A2d1Cov", bound=AxisLike2d[Any, Any], covariant=True)
Basis2d = tuple[_A2d0Cov, _A2d1Cov]

_A3d0Cov = TypeVar("_A3d0Cov", bound=AxisLike3d[Any, Any], covariant=True)
_A3d1Cov = TypeVar("_A3d1Cov", bound=AxisLike3d[Any, Any], covariant=True)
_A3d2Cov = TypeVar("_A3d2Cov", bound=AxisLike3d[Any, Any], covariant=True)
Basis3d = tuple[_A3d0Cov, _A3d1Cov, _A3d2Cov]


_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)

FundamentalPositionBasis3d = tuple[
    FundamentalPositionAxis3d[_NF0Inv],
    FundamentalPositionAxis3d[_NF1Inv],
    FundamentalPositionAxis3d[_NF2Inv],
]

FundamentalMomentumBasis3d = tuple[
    FundamentalMomentumAxis3d[_NF0Inv],
    FundamentalMomentumAxis3d[_NF1Inv],
    FundamentalMomentumAxis3d[_NF2Inv],
]
