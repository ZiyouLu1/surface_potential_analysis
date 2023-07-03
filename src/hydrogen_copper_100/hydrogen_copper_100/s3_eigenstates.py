from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from surface_potential_analysis.state_vector.eigenstate_collection import (
    calculate_eigenstate_collection,
)
from surface_potential_analysis.util.decorators import npy_cached

from .s2_hamiltonian import get_hamiltonian, get_hamiltonian_relaxed
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    _L0 = TypeVar("_L0", bound=int)
    _L1 = TypeVar("_L1", bound=int)
    _L2 = TypeVar("_L2", bound=int)
    from surface_potential_analysis.axis.axis import (
        ExplicitAxis3d,
        FundamentalMomentumAxis3d,
    )
    from surface_potential_analysis.operator.operator import HamiltonianWith3dBasis
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        EigenstateColllection,
    )


def _get_eigenstate_collection_cache(shape: tuple[_L0, _L1, _L2]) -> Path:
    return get_data_path(
        f"eigenstates/eigenstates_{shape[0]}_{shape[1]}_{shape[2]}.npy"
    )


@npy_cached(_get_eigenstate_collection_cache, load_pickle=True)
def get_eigenstate_collection(
    shape: tuple[_L0, _L1, _L2],
) -> EigenstateColllection[
    tuple[
        FundamentalMomentumAxis3d[_L0],
        FundamentalMomentumAxis3d[_L1],
        ExplicitAxis3d[Literal[250], _L2],
    ],
    Literal[11],
]:
    bloch_fractions_x = np.linspace(-0.5, 0.5, 11)
    bloch_fractions = np.array([[f, 0, 0] for f in bloch_fractions_x])

    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> HamiltonianWith3dBasis[
        FundamentalMomentumAxis3d[_L0],
        FundamentalMomentumAxis3d[_L1],
        ExplicitAxis3d[Literal[250], _L2],
    ]:
        return get_hamiltonian(
            shape=(2 * shape[0], 2 * shape[1], 250),
            bloch_fraction=bloch_fraction,
            resolution=shape,
        )

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_fractions, subset_by_index=(0, 10)  # type: ignore[arg-type]
    )


def _get_eigenstate_collection_relaxed_cache(shape: tuple[_L0, _L1, _L2]) -> Path:
    return get_data_path(
        f"eigenstates/eigenstates_relaxed_{shape[0]}_{shape[1]}_{shape[2]}.npy"
    )


@npy_cached(_get_eigenstate_collection_relaxed_cache, load_pickle=True)
def get_eigenstate_collection_relaxed(
    shape: tuple[_L0, _L1, _L2],
) -> EigenstateColllection[
    tuple[
        FundamentalMomentumAxis3d[_L0],
        FundamentalMomentumAxis3d[_L1],
        ExplicitAxis3d[Literal[250], _L2],
    ],
    Literal[11],
]:
    bloch_fractions_x = np.linspace(-0.5, 0.5, 11)
    bloch_fractions = np.array([[f, 0, 0] for f in bloch_fractions_x])

    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> HamiltonianWith3dBasis[
        FundamentalMomentumAxis3d[_L0],
        FundamentalMomentumAxis3d[_L1],
        ExplicitAxis3d[Literal[250], _L2],
    ]:
        return get_hamiltonian_relaxed(
            shape=(2 * shape[0], 2 * shape[1], 250),
            bloch_fraction=bloch_fraction,
            resolution=shape,
        )

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_fractions, subset_by_index=(0, 10)  # type: ignore[arg-type]
    )
