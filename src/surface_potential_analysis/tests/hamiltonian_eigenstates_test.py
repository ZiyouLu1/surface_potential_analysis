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
        basis: MomentumBasisConfig[int, int, int] = {
            "x0_basis": {"_len": np.random.randint(1, 10), "_type": "momentum"},
            "x1_basis": {"_len": np.random.randint(1, 10), "_type": "momentum"},
            "x2_basis": {"_len": np.random.randint(1, 10), "_type": "momentum"},
        }
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

    def test_eigenstate_normalization(self) -> None:
        width = np.random.randint(4, 20)
        nz = 100
        config: EigenstateConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (width // 2, width // 2, 14),
        }

        points = generate_symmetrical_points(nz, width)
        data: EnergyInterpolation = {
            "points": points.tolist(),
            "dz": 1,
        }

        hamiltonian = SurfaceHamiltonianUtil(config, data, 0)

        kx = 0
        ky = 0
        eig_val, eig_states = hamiltonian.calculate_eigenvalues(kx, ky)

        np.testing.assert_allclose(
            np.array([np.linalg.norm(x["eigenvector"]) for x in eig_states]),
            np.ones_like(eig_val),
        )
