from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from surface_potential_analysis.state_vector.eigenstate_collection import (
    EigenstateColllection,
    calculate_eigenstate_collection,
)
from surface_potential_analysis.util.decorators import npy_cached

from .s2_hamiltonian import get_hamiltonian
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalTransformedPositionBasis1d,
    )
    from surface_potential_analysis.basis.block_fraction_basis import (
        ExplicitBlockFractionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.operator.operator import SingleBasisOperator

_L0Inv = TypeVar("_L0Inv", bound=int)


def _get_eigenstate_collection_cache(shape: tuple[_L0Inv]) -> Path:
    return get_data_path(f"eigenstates/eigenstates_{shape[0]}.npy")


@npy_cached(_get_eigenstate_collection_cache, load_pickle=True)
def get_eigenstate_collection(
    shape: tuple[_L0Inv],
) -> EigenstateColllection[
    StackedBasisLike[ExplicitBlockFractionBasis[Literal[11]], FundamentalBasis[int]],
    StackedBasisLike[FundamentalTransformedPositionBasis1d[_L0Inv]],
]:
    bloch_fractions = np.linspace(-0.5, 0.5, 11).reshape(1, 11)

    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[
        StackedBasisLike[FundamentalTransformedPositionBasis1d[_L0Inv]]
    ]:
        return get_hamiltonian(shape=shape, bloch_fraction=bloch_fraction)

    return calculate_eigenstate_collection(
        hamiltonian_generator,
        bloch_fractions,
        subset_by_index=(0, min(99, shape[0] - 1)),
    )
