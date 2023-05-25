from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import special_ortho_group

from surface_potential_analysis.basis_config.conversion import convert_vector
from surface_potential_analysis.basis_config.util import BasisConfigUtil

from .utils import get_random_explicit_basis

rng = np.random.default_rng()


class BasisConfigConversionTest(unittest.TestCase):
    def test_convert_vector_normalization(self) -> None:
        fundamental_shape = (rng.integers(2, 5), rng.integers(2, 5), rng.integers(2, 5))

        _config0 = (
            get_random_explicit_basis(fundamental_n=fundamental_shape[0]),
            get_random_explicit_basis(fundamental_n=fundamental_shape[1]),
            get_random_explicit_basis(fundamental_n=fundamental_shape[2]),
        )
        # Note this only holds if the space spanned by _config1 contains the space of _config0
        _config1 = (
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

        util0 = BasisConfigUtil(_config0)
        vector = special_ortho_group.rvs(util0.size)[0]
        converted = convert_vector(vector, _config0, _config1)

        util1 = BasisConfigUtil(_config1)
        np.testing.assert_equal(converted.size, util1.size)

        np.testing.assert_array_almost_equal(np.linalg.norm(converted), 1)

        actual_reversed = convert_vector(converted, _config1, _config0)
        np.testing.assert_array_almost_equal(actual_reversed, vector)

    def test_convert_vector_equivalent(self) -> None:
        fundamental_shape = (rng.integers(2, 5), rng.integers(2, 5), rng.integers(2, 5))

        _config0 = (
            get_random_explicit_basis(fundamental_n=fundamental_shape[0]),
            get_random_explicit_basis(fundamental_n=fundamental_shape[1]),
            get_random_explicit_basis(fundamental_n=fundamental_shape[2]),
        )
        _config1 = _config0

        util = BasisConfigUtil(_config0)
        vector = special_ortho_group.rvs(util.size)[0]
        converted = convert_vector(vector, _config0, _config1)

        np.testing.assert_array_almost_equal(np.linalg.norm(converted), 1)
        np.testing.assert_array_almost_equal(converted, vector)
