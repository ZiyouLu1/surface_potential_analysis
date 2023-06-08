from __future__ import annotations

import unittest

import numpy as np

from _test_surface_potential_analysis.hamiltonian_builder_test import (
    convert_explicit_basis_x2,
)

rng = np.random.default_rng()


class HamiltonianTest(unittest.TestCase):
    def test_convert_explicit_basis_x2_diagonal(self) -> None:
        shape = rng.integers(2, 10, size=3)
        nz: int = rng.integers(1, shape[2])  # type: ignore[assignment]

        points = rng.random((*shape, *shape))
        diagonal = np.eye(nz, shape.item(2))

        actual = convert_explicit_basis_x2(points, diagonal)
        expected = points[:, :, :nz, :, :, :nz]

        np.testing.assert_equal(expected, actual)
