from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalMomentumBasis
from surface_potential_analysis.basis_config.util import (
    BasisConfigUtil,
)
from surface_potential_analysis.eigenstate.eigenstate_calculation import (
    calculate_energy,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis_config.basis_config import (
        FundamentalMomentumBasisConfig,
    )
    from surface_potential_analysis.hamiltonian import (
        FundamentalMomentumBasisHamiltonian,
    )


rng = np.random.default_rng()


class HamiltonianEigenstates(unittest.TestCase):
    def test_calculate_energy_diagonal(self) -> None:
        basis: FundamentalMomentumBasisConfig[int, int, int] = (
            FundamentalMomentumBasis(np.array([1, 0, 0]), rng.integers(1, 10)),
            FundamentalMomentumBasis(np.array([0, 1, 0]), rng.integers(1, 10)),
            FundamentalMomentumBasis(np.array([0, 0, 1]), rng.integers(1, 10)),
        )
        energies = rng.random(len(BasisConfigUtil(basis)))
        hamiltonian: FundamentalMomentumBasisHamiltonian[int, int, int] = {
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
