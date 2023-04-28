from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from scipy.stats import special_ortho_group

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        ExplicitBasis,
        MomentumBasis,
        PositionBasis,
    )

rng = np.random.default_rng()


@overload
def get_random_explicit_basis(
    _type: Literal["momentum"],
    *,
    fundamental_n: int | None = None,
    n: int | None = None,
) -> ExplicitBasis[int, MomentumBasis[int]]:
    ...


@overload
def get_random_explicit_basis(
    _type: Literal["position"],
    *,
    fundamental_n: int | None = None,
    n: int | None = None,
) -> ExplicitBasis[int, PositionBasis[int]]:
    ...


def get_random_explicit_basis(
    _type: Literal["position"] | Literal["momentum"],
    *,
    fundamental_n: int | None = None,
    n: int | None = None,
) -> (
    ExplicitBasis[int, PositionBasis[int]]
    | ExplicitBasis[int, MomentumBasis[int]]
    | ExplicitBasis[int, PositionBasis[int] | MomentumBasis[int]]
):
    fundamental_n = (
        rng.integers(2 if n is None else n, 5)
        if fundamental_n is None
        else fundamental_n
    )
    n = rng.integers(1, fundamental_n) if n is None else n
    vectors = special_ortho_group.rvs(fundamental_n)[:n]
    parent: PositionBasis[int] | MomentumBasis[int] = {  # type:ignore[misc,assignment]
        "_type": _type,
        "delta_x": np.array([1, 0, 0]),
        "n": fundamental_n,
    }

    return {
        "_type": "explicit",
        "parent": parent,
        "vectors": vectors,
    }
