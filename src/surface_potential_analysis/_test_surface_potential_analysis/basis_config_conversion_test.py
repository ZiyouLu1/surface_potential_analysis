from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import special_ortho_group

from _test_surface_potential_analysis.utils import get_random_explicit_basis
from surface_potential_analysis.basis.conversion import convert_vector
from surface_potential_analysis.basis.util import Basis3dUtil

rng = np.random.default_rng()


class BasisConfigConversionTest(unittest.TestCase):
    def test_convert_vector_normalization(self) -> None:
        fundamental_shape = (rng.integers(2, 5), rng.integers(2, 5), rng.integers(2, 5))

        _basis_0 = (
            get_random_explicit_basis(fundamental_n=fundamental_shape[0]),
            get_random_explicit_basis(fundamental_n=fundamental_shape[1]),
            get_random_explicit_basis(fundamental_n=fundamental_shape[2]),
        )
        # Note this only holds if the space spanned by _basis_1 contains the space of _basis_0
        _basis_1 = (
            get_random_explicit_basis(
                fundamental_n=fundamental_shape[0], n=fundamental_shape[0]
            ),
            get_random_explicit_basis(
                fundamental_n=fundamental_shape[1], n=fundamental_shape[1]
            ),
            get_random_explicit_basis(
                fundamental_n=fundamental_shape[2], n=fundamental_shape[2]
            ),
        )

        util0 = Basis3dUtil(_basis_0)
        vector = special_ortho_group.rvs(util0.size)[0]
        converted = convert_vector(vector, _basis_0, _basis_1)

        util1 = Basis3dUtil(_basis_1)
        np.testing.assert_equal(converted.size, util1.size)

        np.testing.assert_array_almost_equal(np.linalg.norm(converted), 1)

        actual_reversed = convert_vector(converted, _basis_1, _basis_0)
        np.testing.assert_array_almost_equal(actual_reversed, vector)

    def test_convert_vector_equivalent(self) -> None:
        fundamental_shape = (rng.integers(2, 5), rng.integers(2, 5), rng.integers(2, 5))

        _basis_0 = (
            get_random_explicit_basis(fundamental_n=fundamental_shape[0]),
            get_random_explicit_basis(fundamental_n=fundamental_shape[1]),
            get_random_explicit_basis(fundamental_n=fundamental_shape[2]),
        )
        _basis_1 = _basis_0

        util = Basis3dUtil(_basis_0)
        vector = special_ortho_group.rvs(util.size)[0]
        converted = convert_vector(vector, _basis_0, _basis_1)

        np.testing.assert_array_almost_equal(np.linalg.norm(converted), 1)
        np.testing.assert_array_almost_equal(converted, vector)
