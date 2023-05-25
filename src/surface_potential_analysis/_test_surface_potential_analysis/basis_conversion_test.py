from __future__ import annotations

import unittest

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalMomentumBasis,
)
from surface_potential_analysis.basis.conversion import (
    get_basis_conversion_matrix,
)

from .utils import get_random_explicit_basis

rng = np.random.default_rng()


class BasisConversionTest(unittest.TestCase):
    def test_as_explicit_position_basis_momentum_normalization(self) -> None:
        fundamental_n = rng.integers(2, 5)
        n = rng.integers(1, fundamental_n)

        basis = get_random_explicit_basis(fundamental_n=fundamental_n, n=n)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(basis.vectors, axis=1), np.ones(n)
        )

    def test_as_explicit_position_basis_momentum(self) -> None:
        n = rng.integers(2, 5)

        basis = FundamentalMomentumBasis(np.array([1, 0, 0]), n)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(basis.vectors, axis=1), np.ones(n)
        )
        np.testing.assert_array_almost_equal(
            basis.vectors,
            np.exp(
                (-1j * 2 * np.pi)
                * np.arange(n)[:, np.newaxis]
                * np.linspace(0, 1, n, endpoint=False)[np.newaxis, :]
            )
            / np.sqrt(n),
        )

    def test_get_basis_conversion_matrix_diagonal(self) -> None:
        fundamental_n = rng.integers(2, 5)
        n = rng.integers(1, fundamental_n)

        basis_0 = get_random_explicit_basis(fundamental_n=fundamental_n, n=n)
        np.testing.assert_array_equal(basis_0.vectors.shape, (n, fundamental_n))

        matrix = get_basis_conversion_matrix(basis_0, basis_0)
        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

        basis_1 = get_random_explicit_basis(fundamental_n=fundamental_n, n=n)

        matrix = get_basis_conversion_matrix(basis_1, basis_1)
        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

    def test_basis_conversion_matrix_position(self) -> None:
        fundamental_n = rng.integers(2, 5)
        n = rng.integers(1, fundamental_n)
        basis_0 = get_random_explicit_basis(fundamental_n=fundamental_n, n=n)

        matrix = get_basis_conversion_matrix(basis_0, basis_0)

        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

        basis_1 = get_random_explicit_basis(fundamental_n=fundamental_n, n=n)

        matrix = get_basis_conversion_matrix(basis_1, basis_1)

        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))
