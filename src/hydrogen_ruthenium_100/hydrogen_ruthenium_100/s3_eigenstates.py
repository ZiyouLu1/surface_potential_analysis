from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from surface_potential_analysis.state_vector.eigenstate_collection import (
    calculate_eigenstate_collection,
)
from surface_potential_analysis.util.decorators import npy_cached

from .s2_hamiltonian import get_hamiltonian_deuterium, get_hamiltonian_hydrogen
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.axis.axis import (
        ExplicitBasis,
        FundamentalBasis,
        FundamentalTransformedPositionBasis,
        TransformedPositionBasis,
    )
    from surface_potential_analysis.axis.block_fraction_axis import (
        ExplicitBlockFractionAxis,
    )
    from surface_potential_analysis.axis.stacked_axis import StackedBasisLike
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        EigenstateColllection,
    )

    _L0 = TypeVar("_L0", bound=int)
    _L1 = TypeVar("_L1", bound=int)
    _L2 = TypeVar("_L2", bound=int)


def _get_eigenstate_collection_cache_hydrogen(shape: tuple[_L0, _L1, _L2]) -> Path:
    return get_data_path(
        f"eigenstates/eigenstates_hydrogen_{shape[0]}_{shape[1]}_{shape[2]}.npy"
    )


@npy_cached(_get_eigenstate_collection_cache_hydrogen, load_pickle=True)  # type: ignore[misc]
def get_eigenstate_collection_hydrogen(
    shape: tuple[_L0, _L1, _L2],
) -> EigenstateColllection[
    StackedBasisLike[
        ExplicitBlockFractionAxis[Literal[11]], FundamentalBasis[Literal[10]]
    ],
    StackedBasisLike[
        TransformedPositionBasis[_L0, _L0, Literal[3]],
        TransformedPositionBasis[_L1, _L1, Literal[3]],
        ExplicitBasis[Literal[250], _L2, Literal[3]],
    ],
]:
    bloch_fractions_x = np.linspace(-0.5, 0.5, 11)
    bloch_fractions = np.array([[f, 0, 0] for f in bloch_fractions_x])

    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[
        StackedBasisLike[
            FundamentalTransformedPositionBasis[_L0, Literal[3]],
            FundamentalTransformedPositionBasis[_L1, Literal[3]],
            ExplicitBasis[Literal[250], _L2, Literal[3]],
        ]
    ]:
        return get_hamiltonian_hydrogen(
            shape=(2 * shape[0], 2 * shape[1], 250),
            bloch_fraction=bloch_fraction,
            resolution=shape,
        )

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_fractions, subset_by_index=(0, 10)  # type: ignore[arg-type,return-value]
    )


def _get_eigenstate_collection_cache_deuterium(shape: tuple[_L0, _L1, _L2]) -> Path:
    return get_data_path(
        f"eigenstates/eigenstates_deuterium_{shape[0]}_{shape[1]}_{shape[2]}.npy"
    )


@npy_cached(_get_eigenstate_collection_cache_deuterium, load_pickle=True)  # type: ignore[misc]
def get_eigenstate_collection_deuterium(
    shape: tuple[_L0, _L1, _L2],
) -> EigenstateColllection[
    StackedBasisLike[
        ExplicitBlockFractionAxis[Literal[11]], FundamentalBasis[Literal[10]]
    ],
    StackedBasisLike[
        TransformedPositionBasis[_L0, _L0, Literal[3]],
        TransformedPositionBasis[_L1, _L1, Literal[3]],
        ExplicitBasis[Literal[250], _L2, Literal[3]],
    ],
]:
    bloch_fractions_x = np.linspace(-0.5, 0.5, 11)
    bloch_fractions = np.array([[f, 0, 0] for f in bloch_fractions_x])

    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[
        StackedBasisLike[
            FundamentalTransformedPositionBasis[_L0, Literal[3]],
            FundamentalTransformedPositionBasis[_L1, Literal[3]],
            ExplicitBasis[Literal[250], _L2, Literal[3]],
        ]
    ]:
        return get_hamiltonian_deuterium(
            shape=(2 * shape[0], 2 * shape[1], 250),
            bloch_fraction=bloch_fraction,
            resolution=shape,
        )

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_fractions, subset_by_index=(0, 10)  # type: ignore[arg-type,return-value]
    )
