from pathlib import Path
from typing import Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis import Basis, MomentumBasis
from surface_potential_analysis.basis_config import BasisConfig

_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)


class Overlap(TypedDict, Generic[_BX0Cov, _BX1Cov, _BX2Cov]):
    basis: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]


def save_overlap(path: Path, overlap: Overlap):
    state = np.array(overlap, dtype=dict)
    np.save(path, state)


def load_overlap(path: Path) -> Overlap[Any, Any, Any, Any, Any]:
    return np.load(path)[()]  # type:ignore


_L0Inv = TypeVar("_L0Inv", bound=Basis[Any, Any])
_L1Inv = TypeVar("_L1Inv", bound=Basis[Any, Any])
_L2Inv = TypeVar("_L2Inv", bound=Basis[Any, Any])

OverlapTransform = Overlap[
    MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], MomentumBasis[_L2Inv]
]
