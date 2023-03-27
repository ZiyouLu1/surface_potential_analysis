import math
from typing import TypedDict, TypeVar

import numpy as np
import scipy
import scipy.special
from scipy.constants import hbar

from .basis import ExplicitBasis, FundamentalBasis, PositionBasis, PositionBasisUtil

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)

_FBInv = TypeVar("_FBInv", bound=FundamentalBasis[int])


def calculate_sho_wavefunction(
    x_points: np.ndarray[tuple[int], np.dtype[np.float_]],
    sho_omega: float,
    mass: float,
    n: int,
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    norm = (sho_omega * mass / hbar) ** 0.5
    normalized_x = x_points * norm

    prefactor = math.sqrt((norm / (2**n)) / (math.factorial(n) * math.sqrt(math.pi)))
    hermite = scipy.special.eval_hermite(n, normalized_x)
    exponential = np.exp(-np.square(normalized_x) / 2)
    return prefactor * hermite * exponential  # type: ignore


class SHOBasisConfig(TypedDict):
    sho_omega: float
    mass: float


def sho_basis_from_config(
    parent: _FBInv, config: SHOBasisConfig, n: _L0Inv
) -> ExplicitBasis[_L0Inv, _FBInv]:
    return


def infinate_sho_basis_from_config(
    parent: PositionBasis[_L1Inv], config: SHOBasisConfig, n: _L0Inv
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
    util = PositionBasisUtil(parent)
    vectors = np.array(
        [calculate_sho_wavefunction(util.x_points, n=i, **config) for i in range(n)]
    )
    return {"_type": "explicit", "parent": parent, "vectors": vectors}
