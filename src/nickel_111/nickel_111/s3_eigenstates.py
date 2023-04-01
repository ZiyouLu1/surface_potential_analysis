from typing import Any, Literal

import numpy as np

from nickel_111.s1_potential import get_interpolated_nickel_potential
from surface_potential_analysis.basis_config import BasisConfigUtil
from surface_potential_analysis.eigenstate.eigenstate_collection import (
    EigenstateColllection,
    calculate_eigenstate_collection,
    save_eigenstate_collection,
)
from surface_potential_analysis.hamiltonian import Hamiltonian

from .s2_hamiltonian import generate_hamiltonian_sho
from .surface_data import get_data_path


def _calculate_eigenstate_collection_sho(
    bloch_phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    resolution: tuple[int, int, int],
) -> EigenstateColllection[Any]:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> Hamiltonian[Any]:
        return generate_hamiltonian_sho(
            shape=(2 * resolution[0], 2 * resolution[1], 100),
            bloch_phase=x,
            resolution=resolution,
        )

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_phases, include_bands=list(range(10))
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
    potential = get_interpolated_nickel_potential(shape=(1, 1, 1))
    util = BasisConfigUtil(potential["basis"])

    kx_points = np.linspace(0, (np.abs(util.dk0[0]) + np.abs(util.dk1[0])) / 2, 5)
    ky_points = np.linspace(0, (np.abs(util.dk0[1]) + np.abs(util.dk1[1])) / 2, 5)
    kz_points = np.zeros_like(kx_points)
    bloch_phases = np.array([kx_points, ky_points, kz_points]).T

    _generate_eigenstate_collection_sho(bloch_phases, (10, 10, 5))

    _generate_eigenstate_collection_sho(bloch_phases, (23, 23, 10))

    _generate_eigenstate_collection_sho(bloch_phases, (23, 23, 12))

    # h = generate_hamiltonian(resolution=(25, 25, 16))
    # eigenstates = calculate_energy_eigenstates(
    #     h, kx_points, ky_points, include_bands=list(range(10))
    # )
    # path = get_data_path("eigenstates_25_25_16.json")
    # save_energy_eigenstates(eigenstates, path)


# def test_eigenstates_partial():
#     h = generate_hamiltonian(resolution=(23, 23, 12))
#     n = 10
#     out_20 = h.calculate_eigenvalues(0, 0, n=n)
#     out_all = h.calculate_eigenvalues(0, 0)

#     np.testing.assert_array_almost_equal(np.sort(out_20[0]), np.sort(out_all[0])[:n])

#     np.testing.assert_array_almost_equal(
#         np.array([x["eigenvector"] for x in out_20[1]])[np.argsort(out_20[0])],
#         np.array([x["eigenvector"] for x in out_all[1]])[np.argsort(out_all[0])][:n],
#     )
