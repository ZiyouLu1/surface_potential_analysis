from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfigUtil,
    MomentumBasisConfig,
)
from surface_potential_analysis.eigenstate.eigenstate_calculation import (
    calculate_energy,
)

if TYPE_CHECKING:
    from surface_potential_analysis.hamiltonian import MomentumBasisHamiltonian


rng = np.random.default_rng()


class HamiltonianEigenstates(unittest.TestCase):
    def test_calculate_energy_diagonal(self) -> None:
        basis: MomentumBasisConfig[int, int, int] = (
            {
                "n": rng.integers(1, 10),
                "_type": "momentum",
                "delta_x": np.array([1, 0, 0]),
            },
            {
                "n": rng.integers(1, 10),
                "_type": "momentum",
                "delta_x": np.array([0, 1, 0]),
            },
            {
                "n": rng.integers(1, 10),
                "_type": "momentum",
                "delta_x": np.array([0, 0, 1]),
            },
        )
        energies = rng.random(len(BasisConfigUtil(basis)))
        hamiltonian: MomentumBasisHamiltonian[int, int, int] = {
            "basis": basis,
            "array": np.diag(energies),
        }
        actual: list[complex] = []
        for i in range(len(BasisConfigUtil(basis))):
            vector = np.zeros(shape=len(BasisConfigUtil(basis)), dtype=complex)
            vector[i] = np.exp(1j * 2 * np.pi * rng.random())

            actual.append(
                calculate_energy(hamiltonian, {"basis": basis, "vector": vector})
            )

        np.testing.assert_allclose(energies, actual)
