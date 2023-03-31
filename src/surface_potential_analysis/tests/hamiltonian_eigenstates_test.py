import unittest

import numpy as np

from surface_potential_analysis.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    MomentumBasisConfig,
)
from surface_potential_analysis.eigenstate.eigenstate import Eigenstate
from surface_potential_analysis.hamiltonian import Hamiltonian, MomentumBasisHamiltonian
from surface_potential_analysis.hamiltonian_eigenstates import calculate_energy


class HamiltonianEigenstates(unittest.TestCase):
    def test_calculate_energy_diagonal(self) -> None:
        basis: MomentumBasisConfig[int, int, int] = (
            {
                "n": np.random.randint(1, 10),
                "_type": "momentum",
                "delta_x": np.array([1, 0, 0]),
            },
            {
                "n": np.random.randint(1, 10),
                "_type": "momentum",
                "delta_x": np.array([0, 1, 0]),
            },
            {
                "n": np.random.randint(1, 10),
                "_type": "momentum",
                "delta_x": np.array([0, 0, 1]),
            },
        )
        energies = np.random.rand(len(BasisConfigUtil(basis)))
        hamiltonian: MomentumBasisHamiltonian[int, int, int] = {
            "basis": basis,
            "array": np.diag(energies),
        }
        actual: list[complex] = []
        for i in range(len(BasisConfigUtil(basis))):
            vector = np.zeros(shape=len(BasisConfigUtil(basis)), dtype=complex)
            vector[i] = np.exp(1j * 2 * np.pi * np.random.rand())

            actual.append(
                calculate_energy(hamiltonian, {"basis": basis, "vector": vector})
            )

        np.testing.assert_allclose(energies, actual)
