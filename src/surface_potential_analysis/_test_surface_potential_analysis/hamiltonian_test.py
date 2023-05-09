from __future__ import annotations

import unittest

import numpy as np

from surface_potential_analysis.hamiltonian.hamiltonian import (
    MomentumBasisStackedHamiltonian,
    flatten_hamiltonian,
    stack_hamiltonian,
)

from .utils import convert_explicit_basis_x2

rng = np.random.default_rng()


class HamiltonianTest(unittest.TestCase):
    def test_flatten_hamiltonian(self) -> None:
        shape = rng.integers(1, 10, size=3)
        hamiltonian: MomentumBasisStackedHamiltonian[int, int, int] = {
            "array": rng.random((*shape, *shape)),
            "basis": (
                {
                    "n": shape.item(0),
                    "_type": "momentum",
                    "delta_x": np.array([1.0, 0, 0]),
                },
                {
                    "n": shape.item(1),
                    "_type": "momentum",
                    "delta_x": np.array([0, 1.0, 0]),
                },
                {
                    "n": shape.item(2),
                    "_type": "momentum",
                    "delta_x": np.array([0, 0, 1.0]),
                },
            ),
        }
        actual = flatten_hamiltonian(hamiltonian)
        expected = np.zeros((np.prod(shape), np.prod(shape)))
        x0t, x1t, zt = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )
        coords = np.array([x0t.ravel(), x1t.ravel(), zt.ravel()]).T
        for i, (ix0, ix1, ix2) in enumerate(coords):
            for j, (jx0, jx1, jx2) in enumerate(coords):
                expected[i, j] = hamiltonian["array"][ix0, ix1, ix2, jx0, jx1, jx2]

        np.testing.assert_array_equal(actual["array"], expected)

    def test_stack_hamiltonian(self) -> None:
        shape = rng.integers(1, 10, size=3)
        hamiltonian: MomentumBasisStackedHamiltonian[int, int, int] = {
            "array": rng.random((*shape, *shape)),
            "basis": (
                {
                    "n": shape.item(0),
                    "_type": "momentum",
                    "delta_x": np.array([1.0, 0, 0]),
                },
                {
                    "n": shape.item(1),
                    "_type": "momentum",
                    "delta_x": np.array([0, 0, 0]),
                },
                {
                    "n": shape.item(2),
                    "_type": "momentum",
                    "delta_x": np.array([0, 1.0, 0]),
                },
            ),
        }

        actual = stack_hamiltonian(flatten_hamiltonian(hamiltonian))
        np.testing.assert_array_equal(hamiltonian["array"], actual["array"])

    def test_convert_explicit_basis_x2_diagonal(self) -> None:
        shape = rng.integers(2, 10, size=3)
        nz: int = rng.integers(1, shape[2])  # type: ignore[assignment]

        points = rng.random((*shape, *shape))
        diagonal = np.eye(nz, shape.item(2))

        actual = convert_explicit_basis_x2(points, diagonal)
        expected = points[:, :, :nz, :, :, :nz]

        np.testing.assert_equal(expected, actual)
