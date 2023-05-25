from __future__ import annotations

from typing import Any, TypeVar

from surface_potential_analysis.basis.basis import (
    FundamentalMomentumBasis,
    FundamentalPositionBasis,
)
from surface_potential_analysis.basis.basis_like import (
    BasisLike,
)

_BX0Cov = TypeVar("_BX0Cov", bound=BasisLike[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=BasisLike[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=BasisLike[Any, Any], covariant=True)


BasisConfig = tuple[_BX0Cov, _BX1Cov, _BX2Cov]

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)

FundamentalPositionBasisConfig = tuple[
    FundamentalPositionBasis[_NF0Inv],
    FundamentalPositionBasis[_NF1Inv],
    FundamentalPositionBasis[_NF2Inv],
]

FundamentalMomentumBasisConfig = tuple[
    FundamentalMomentumBasis[_NF0Inv],
    FundamentalMomentumBasis[_NF1Inv],
    FundamentalMomentumBasis[_NF2Inv],
]
