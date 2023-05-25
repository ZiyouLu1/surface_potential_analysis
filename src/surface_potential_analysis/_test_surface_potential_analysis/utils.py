from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy.stats import special_ortho_group

from surface_potential_analysis.basis.basis import (
    ExplicitBasis,
)
from surface_potential_analysis.util.util import slice_along_axis

if TYPE_CHECKING:
    from surface_potential_analysis.hamiltonian.hamiltonian import (
        _StackedHamiltonianPoints,
    )

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)

    _LInv = TypeVar("_LInv", bound=int)

rng = np.random.default_rng()


def convert_explicit_basis_x2(
    hamiltonian: _StackedHamiltonianPoints[_L0Inv, _L1Inv, _L2Inv],
    basis: np.ndarray[tuple[_LInv, _L2Inv], np.dtype[np.complex_]],
) -> _StackedHamiltonianPoints[_L0Inv, _L1Inv, _LInv]:
    end_dot = np.sum(
        hamiltonian[slice_along_axis(np.newaxis, -2)]
        * basis.reshape(1, 1, 1, 1, 1, *basis.shape),
        axis=-1,
    )
    return np.sum(  # type: ignore[no-any-return]
        end_dot[slice_along_axis(np.newaxis, 2)]
        * basis.conj().reshape(1, 1, *basis.shape, 1, 1, 1),
        axis=3,
    )


def get_random_explicit_basis(
    fundamental_n: int | None = None,
    n: int | None = None,
) -> ExplicitBasis[int, int]:
    fundamental_n = (
        rng.integers(2 if n is None else n, 5)
        if fundamental_n is None
        else fundamental_n
    )
    n = rng.integers(1, fundamental_n) if n is None else n
    vectors = special_ortho_group.rvs(fundamental_n)[:n]
    return ExplicitBasis(np.array([1, 0, 0]), vectors)
