from __future__ import annotations

import unittest

import numpy as np

from _tests.utils import get_random_explicit_basis
from surface_potential_analysis.basis.basis import (
    MomentumBasis,
    explicit_momentum_basis_in_position,
    explicit_position_basis_in_momentum,
)
from surface_potential_analysis.basis.conversion import (
    as_explicit_position_basis,
    get_basis_conversion_matrix,
)

rng = np.random.default_rng()


class BasisConversionTest(unittest.TestCase):
    def test_as_explicit_position_basis_momentum_normalization(self) -> None:
        fundamental_n = rng.integers(2, 5)
        n = rng.integers(1, fundamental_n)

        basis = get_random_explicit_basis("momentum", fundamental_n=fundamental_n, n=n)

        actual = as_explicit_position_basis(basis)
        np.testing.assert_array_almost_equal(
            np.linalg.norm(actual["vectors"], axis=1), np.ones(n)
        )

    def test_as_explicit_position_basis_momentum(self) -> None:
        n = rng.integers(2, 5)

        basis: MomentumBasis[int] = {
            "_type": "momentum",
            "delta_x": np.array([1, 0, 0]),
            "n": n,
        }

        actual = as_explicit_position_basis(basis)
        np.testing.assert_array_almost_equal(
            np.linalg.norm(actual["vectors"], axis=1), np.ones(n)
        )
        np.testing.assert_array_almost_equal(
            actual["vectors"],
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

        basis_0 = get_random_explicit_basis(
            "position", fundamental_n=fundamental_n, n=n
        )
        np.testing.assert_array_equal(basis_0["vectors"].shape, (n, fundamental_n))

        matrix = get_basis_conversion_matrix(basis_0, basis_0)
        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

        basis_1 = get_random_explicit_basis(
            "momentum", fundamental_n=fundamental_n, n=n
        )

        matrix = get_basis_conversion_matrix(basis_1, basis_1)
        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

    def test_basis_conversion_matrix_position(self) -> None:
        fundamental_n = rng.integers(2, 5)
        n = rng.integers(1, fundamental_n)
        basis_0 = get_random_explicit_basis(
            "position", fundamental_n=fundamental_n, n=n
        )

        transformed_0 = explicit_position_basis_in_momentum(basis_0)

        matrix = get_basis_conversion_matrix(basis_0, transformed_0)

        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

        basis_1 = get_random_explicit_basis(
            "momentum", fundamental_n=fundamental_n, n=n
        )

        transformed_1 = explicit_momentum_basis_in_position(basis_1)

        matrix = get_basis_conversion_matrix(basis_1, transformed_1)

        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

    def test_convert_explicit_basis_flat(self) -> None:
        n = 10

        basis = get_random_explicit_basis("position", fundamental_n=n, n=1)
        basis["vectors"] = np.ones_like(basis["vectors"])
        basis["vectors"] /= np.linalg.norm(basis["vectors"])

        transformed = explicit_position_basis_in_momentum(basis)
        expected = np.zeros_like(basis["vectors"])
        expected[0, 0] = 1

        np.testing.assert_array_almost_equal(transformed["vectors"], expected)

        double_transformed = explicit_momentum_basis_in_position(transformed)
        np.testing.assert_array_almost_equal(
            double_transformed["vectors"], basis["vectors"]
        )

    def test_convert_explicit_basis_random(self) -> None:
        n = 10
        basis = get_random_explicit_basis("position", fundamental_n=n, n=1)

        transformed = explicit_position_basis_in_momentum(basis)
        double_transformed = explicit_momentum_basis_in_position(transformed)
        np.testing.assert_array_almost_equal(
            double_transformed["vectors"], basis["vectors"]
        )
