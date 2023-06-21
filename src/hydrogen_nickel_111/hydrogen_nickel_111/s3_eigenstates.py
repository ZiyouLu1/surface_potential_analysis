from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from surface_potential_analysis.state_vector.eigenstate_collection import (
    EigenstateColllection3d,
    calculate_eigenstate_collection,
    save_eigenstate_collection,
)
from surface_potential_analysis.util.decorators import npy_cached

from .s2_hamiltonian import generate_hamiltonian_sho, get_hamiltonian_deuterium
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.axis.axis import (
        ExplicitAxis3d,
        FundamentalMomentumAxis3d,
    )
    from surface_potential_analysis.operator import SingleBasisOperator3d
    from surface_potential_analysis.operator.operator import HamiltonianWith3dBasis
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        EigenstateColllection,
    )


def _calculate_eigenstate_collection_sho(
    bloch_fractions: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    resolution: tuple[int, int, int],
) -> EigenstateColllection3d[Any, Any]:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator3d[Any]:
        return generate_hamiltonian_sho(
            shape=(200, 200, 501),
            bloch_fraction=x,
            resolution=resolution,
        )

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_fractions, subset_by_index=(0, 10)  # type: ignore[arg-type]
    )


def _generate_eigenstate_collection_sho(
    bloch_fractions: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    resolution: tuple[int, int, int],
) -> None:
    collection = _calculate_eigenstate_collection_sho(bloch_fractions, resolution)
    filename = f"eigenstates_{resolution[0]}_{resolution[1]}_{resolution[2]}.npy"
    path = get_data_path(filename)
    save_eigenstate_collection(path, collection)


def generate_eigenstates_data() -> None:
    """Generate data on the eigenstates and eigenvalues for a range of resolutions."""
    kx_points = np.linspace(0, 0.5, 5)
    ky_points = np.linspace(0, 0.5, 5)
    kz_points = np.zeros_like(kx_points)
    bloch_fractions = np.array([kx_points, ky_points, kz_points]).T

    # _generate_eigenstate_collection_sho(bloch_fractions, (10, 10, 5))  # noqa: ERA001

    _generate_eigenstate_collection_sho(bloch_fractions, (23, 23, 10))

    # _generate_eigenstate_collection_sho(bloch_fractions, (23, 23, 12)) # noqa: ERA001

    # _generate_eigenstate_collection_sho(bloch_fractions, (25, 25, 16)) # noqa: ERA001


def _get_eigenstate_collection_cache(shape: tuple[_L0, _L1, _L2]) -> Path:
    return get_data_path(
        f"eigenstates/eigenstates_deuterium_{shape[0]}_{shape[1]}_{shape[2]}.npy"
    )


_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)


@npy_cached(_get_eigenstate_collection_cache, load_pickle=True)
def get_eigenstate_collection_deuterium(
    shape: tuple[_L0, _L1, _L2],
) -> EigenstateColllection[
    tuple[
        FundamentalMomentumAxis3d[_L0],
        FundamentalMomentumAxis3d[_L1],
        ExplicitAxis3d[Literal[100], _L2],
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
        ExplicitAxis3d[Literal[100], _L2],
    ]:
        return get_hamiltonian_deuterium(
            shape=(2 * shape[0], 2 * shape[1], 100),
            bloch_fraction=bloch_fraction,
            resolution=shape,
        )

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_fractions, subset_by_index=(0, 99)  # type: ignore[arg-type]
    )
