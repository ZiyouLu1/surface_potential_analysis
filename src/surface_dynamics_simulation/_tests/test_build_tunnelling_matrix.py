from __future__ import annotations

import unittest

import numpy as np

from surface_dynamics_simulation.simulation_config import build_tunnelling_matrix

rng = np.random.default_rng()


class TunnellingMatrixTest(unittest.TestCase):
    def test_build_tunnelling_matrix_norm(self) -> None:
        coefficients = rng.random((10, 10, 9))
        matrix = build_tunnelling_matrix(coefficients, (1, 1))
        non_diagonal_matrix = matrix.copy()
        np.fill_diagonal(non_diagonal_matrix, 0)
        np.testing.assert_array_almost_equal(
            np.diag(matrix), -np.sum(non_diagonal_matrix, axis=0)
        )
