import unittest
from typing import TYPE_CHECKING

import numpy as np

from surface_potential_analysis.basis.basis import (
    explicit_momentum_basis_in_position,
    explicit_position_basis_in_momentum,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis import ExplicitBasis, PositionBasis

rng = np.random.default_rng()


class BasisTest(unittest.TestCase):
    def test_convert_explicit_basis_flat(self) -> None:
        n = 10
        vectors = np.ones((1, n))
        vectors /= np.linalg.norm(vectors, axis=1)

        basis: ExplicitBasis[int, PositionBasis[int]] = {
            "_type": "explicit",
            "vectors": vectors,
            "parent": {"_type": "position", "delta_x": np.array([1, 0, 0]), "n": 10},
        }

        transformed = explicit_position_basis_in_momentum(basis)
        expected = np.zeros_like(vectors)
        expected[0, 0] = 1

        np.testing.assert_array_almost_equal(transformed["vectors"], expected)

        double_transformed = explicit_momentum_basis_in_position(transformed)
        np.testing.assert_array_almost_equal(
            double_transformed["vectors"], basis["vectors"]
        )

    def test_convert_explicit_basis_random(self) -> None:
        n = 10
        vectors = np.array(rng.random((1, n)), dtype=complex)
        vectors /= np.linalg.norm(vectors, axis=1)

        basis: ExplicitBasis[int, PositionBasis[int]] = {
            "_type": "explicit",
            "vectors": vectors,
            "parent": {"_type": "position", "delta_x": np.array([1, 0, 0]), "n": 10},
        }

        transformed = explicit_position_basis_in_momentum(basis)
        double_transformed = explicit_momentum_basis_in_position(transformed)
        np.testing.assert_array_almost_equal(
            double_transformed["vectors"], basis["vectors"]
        )
