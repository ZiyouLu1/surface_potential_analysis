from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.basis_config.util import BasisConfigUtil
from surface_potential_analysis.eigenstate.eigenstate_collection import (
    EigenstateColllection,
    calculate_eigenstate_collection,
    save_eigenstate_collection,
)

from copper_100.s1_potential import get_interpolated_potential

from .s2_hamiltonian import generate_hamiltonian_sho, generate_hamiltonian_sho_relaxed
from .surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis.hamiltonian.hamiltonian import Hamiltonian


def _calculate_eigenstate_collection_sho(
    bloch_phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    resolution: tuple[int, int, int],
) -> EigenstateColllection[Any]:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> Hamiltonian[Any]:
        return generate_hamiltonian_sho(
            shape=(200, 200, 501),
            bloch_phase=x,
            resolution=resolution,
        )

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_phases, subset_by_index=(0, 10)
    )


def _generate_eigenstate_collection_sho(
    bloch_phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    resolution: tuple[int, int, int],
) -> None:
    collection = _calculate_eigenstate_collection_sho(bloch_phases, resolution)
    filename = f"eigenstates_{resolution[0]}_{resolution[1]}_{resolution[2]}.npy"
    path = get_data_path(filename)
    save_eigenstate_collection(path, collection)


def generate_eigenstates_data() -> None:
    """Generate data on the eigenstates and eigenvalues for a range of resolutions."""
    potential = get_interpolated_potential(shape=(1, 1, 1))
    util = BasisConfigUtil(potential["basis"])

    kx_points = np.linspace(0, (np.abs(util.dk0[0]) + np.abs(util.dk1[0])) / 2, 5)
    ky_points = np.linspace(0, (np.abs(util.dk0[1]) + np.abs(util.dk1[1])) / 2, 5)
    kz_points = np.zeros_like(kx_points)
    bloch_phases = np.array([kx_points, ky_points, kz_points]).T

    # _generate_eigenstate_collection_sho(bloch_phases, (25, 25, 14))  # noqa: ERA001

    _generate_eigenstate_collection_sho(bloch_phases, (23, 23, 14))

    # _generate_eigenstate_collection_sho(bloch_phases, (23, 23, 15)) # noqa: ERA001

    # _generate_eigenstate_collection_sho(bloch_phases, (25, 25, 16)) # noqa: ERA001


def _calculate_eigenstate_collection_sho_relaxed(
    bloch_phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    resolution: tuple[int, int, int],
) -> EigenstateColllection[Any]:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> Hamiltonian[Any]:
        return generate_hamiltonian_sho_relaxed(
            shape=(200, 200, 501),
            bloch_phase=x,
            resolution=resolution,
        )

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_phases, subset_by_index=(0, 10)
    )


def _generate_eigenstate_collection_sho_relaxed(
    bloch_phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    resolution: tuple[int, int, int],
) -> None:
    collection = _calculate_eigenstate_collection_sho(bloch_phases, resolution)
    filename = (
        f"eigenstates_relaxed_{resolution[0]}_{resolution[1]}_{resolution[2]}.npy"
    )
    path = get_data_path(filename)
    save_eigenstate_collection(path, collection)


def generate_eigenstates_data_relaxed() -> None:
    """Generate data on the eigenstates and eigenvalues for a range of resolutions."""
    potential = get_interpolated_potential(shape=(1, 1, 1))
    util = BasisConfigUtil(potential["basis"])

    kx_points = np.linspace(0, (np.abs(util.dk0[0]) + np.abs(util.dk1[0])) / 2, 5)
    ky_points = np.linspace(0, (np.abs(util.dk0[1]) + np.abs(util.dk1[1])) / 2, 5)
    kz_points = np.zeros_like(kx_points)
    bloch_phases = np.array([kx_points, ky_points, kz_points]).T

    # _generate_eigenstate_collection_sho(bloch_phases, (17, 17, 13))  # noqa: ERA001

    _generate_eigenstate_collection_sho(bloch_phases, (17, 17, 15))

    # _generate_eigenstate_collection_sho(bloch_phases, (21, 21, 14)) # noqa: ERA001

    # _generate_eigenstate_collection_sho(bloch_phases, (21, 21, 15)) # noqa: ERA001
