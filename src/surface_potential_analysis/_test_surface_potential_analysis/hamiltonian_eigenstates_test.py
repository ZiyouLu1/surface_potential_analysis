from __future__ import annotations

import unittest

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalTransformedPositionAxis3d
from surface_potential_analysis.basis.basis import (
    FundamentalMomentumBasis3d,
)
from surface_potential_analysis.basis.util import (
    AxisWithLengthBasisUtil,
)
from surface_potential_analysis.operator.operator import SingleBasisOperator
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_expectation,
)

rng = np.random.default_rng()

FundamentalMomentumBasisHamiltonian3d = SingleBasisOperator[
    FundamentalMomentumBasis3d[int, int, int]
]


class HamiltonianEigenstates(unittest.TestCase):
    def test_calculate_energy_diagonal(self) -> None:
        basis: FundamentalMomentumBasis3d[int, int, int] = (
            FundamentalTransformedPositionAxis3d(
                np.array([1, 0, 0]), rng.integers(1, 10)
            ),
            FundamentalTransformedPositionAxis3d(
                np.array([0, 1, 0]), rng.integers(1, 10)
            ),
            FundamentalTransformedPositionAxis3d(
                np.array([0, 0, 1]), rng.integers(1, 10)
            ),
        )
        energies = rng.random(AxisWithLengthBasisUtil(basis).size)
        hamiltonian: FundamentalMomentumBasisHamiltonian3d = {
            "basis": basis,
            "dual_basis": basis,
            "array": np.diag(energies),
        }
        actual: list[complex] = []
        for i in range(AxisWithLengthBasisUtil(basis).size):
            vector = np.zeros(
                shape=(AxisWithLengthBasisUtil(basis).size), dtype=complex
            )
            vector[i] = np.exp(1j * 2 * np.pi * rng.random())

            actual.append(
                calculate_expectation(
                    hamiltonian,
                    {"basis": basis, "vector": vector},
                )
            )

        np.testing.assert_allclose(energies, actual)
