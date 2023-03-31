import unittest
from typing import Any, Literal

import numpy as np

from surface_potential_analysis.basis import Basis, MomentumBasis
from surface_potential_analysis.basis_config import PositionBasisConfig
from surface_potential_analysis.hamiltonian import (
    MomentumBasisStackedHamiltonian,
    flatten_hamiltonian,
    stack_hamiltonian,
    truncate_hamiltonian_basis,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    hamiltonian_from_potential,
)
from surface_potential_analysis.potential.potential import Potential


class HamiltonianTest(unittest.TestCase):
    def test_flatten_hamiltonian(self) -> None:
        shape = np.random.randint(1, 10, size=3)
        hamiltonian: MomentumBasisStackedHamiltonian[int, int, int] = {
            "array": np.random.rand(*shape, *shape),
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
        shape = np.random.randint(1, 10, size=3)
        hamiltonian: MomentumBasisStackedHamiltonian[int, int, int] = {
            "array": np.random.rand(*shape, *shape),
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

    def test_truncate_hamiltonian(self) -> None:
        shape = np.random.randint(2, 10, size=3)
        hamiltonian: MomentumBasisStackedHamiltonian[int, int, int] = {
            "array": np.random.rand(*shape, *shape),
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
        axis: Literal[0, 1, 2, -1, -2, -3] = np.random.randint(-3, 2)  # type: ignore
        expected_parent: MomentumBasis[Any] = hamiltonian["basis"][axis % 3]
        size: int = np.random.randint(1, expected_parent["n"])

        truncated = truncate_hamiltonian_basis(hamiltonian, size, axis)
        expected_basis: list[Basis[Any, Any]] = list(hamiltonian["basis"])
        expected_basis[axis % 3] = {
            "_type": "truncated",
            "n": size,
            "parent": expected_parent,
        }
        for expected, actual in zip(expected_basis, truncated["basis"]):
            self.assertDictEqual(expected, actual)

        expected_shape = np.array(hamiltonian["array"].shape)
        expected_shape[axis % 3] = size
        expected_shape[3 + (axis % 3)] = size
        np.testing.assert_array_equal(expected_shape, truncated["array"].shape)

    def test_hamiltonian_from_potential(self) -> None:
        shape = np.random.randint(1, 10, size=3, dtype=int)
        points = np.random.rand(*shape)

        expected_basis: PositionBasisConfig[Any, Any, Any] = (
            {"n": shape.item(0), "_type": "position", "delta_x": np.array([1.0, 0, 0])},
            {"n": shape.item(1), "_type": "position", "delta_x": np.array([0, 1.0, 0])},
            {"n": shape.item(2), "_type": "position", "delta_x": np.array([0, 0, 1.0])},
        )
        potential: Potential[Any, Any, Any] = {
            "basis": expected_basis,
            "points": points,
        }

        hamiltonian = hamiltonian_from_potential(potential)

        for ix0 in range(shape[0]):
            for ix1 in range(shape[1]):
                for ix2 in range(shape[2]):
                    np.testing.assert_equal(
                        hamiltonian["array"][ix0, ix1, ix2, ix0, ix1, ix2],
                        potential["points"][ix0, ix1, ix2],
                    )
        np.testing.assert_equal(
            np.count_nonzero(hamiltonian["array"]),
            np.count_nonzero(potential["points"]),
        )

        for expected, actual in zip(expected_basis, hamiltonian["basis"]):
            self.assertEqual(expected["n"], actual["n"])
            self.assertEqual(expected["_type"], actual["_type"])
            np.testing.assert_array_equal(expected["delta_x"], actual["delta_x"])
