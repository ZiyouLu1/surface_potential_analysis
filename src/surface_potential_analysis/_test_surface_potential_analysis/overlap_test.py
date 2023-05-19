from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import numpy as np

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfigUtil,
    PositionBasisConfigUtil,
)
from surface_potential_analysis.overlap.conversion import (
    convert_overlap_to_momentum_basis,
)
from surface_potential_analysis.overlap.interpolation import (
    get_overlap_momentum_interpolator,
    get_overlap_momentum_interpolator_k_fractions,
)

if TYPE_CHECKING:
    from surface_potential_analysis.overlap.overlap import OverlapPosition

rng = np.random.default_rng()


class OverlapTest(unittest.TestCase):
    def test_overlap_interpolation(self) -> None:
        shape = (10, 10, 10)
        overlap: OverlapPosition[Any, Any, Any] = {
            "basis": PositionBasisConfigUtil.from_resolution(shape),
            "vector": np.array(rng.random(np.prod(shape)), dtype=complex),
        }

        util = BasisConfigUtil(overlap["basis"])
        overlap_momentum = convert_overlap_to_momentum_basis(overlap)
        expected = overlap_momentum["vector"]
        actual = get_overlap_momentum_interpolator(overlap)(util.k_points)  # type: ignore[var-annotated]

        np.testing.assert_array_almost_equal(expected, actual)

    def test_overlap_interpolation_fractions(self) -> None:
        shape = (10, 10, 10)
        overlap: OverlapPosition[Any, Any, Any] = {
            "basis": PositionBasisConfigUtil.from_resolution(shape),
            "vector": np.array(rng.random(np.prod(shape)), dtype=complex),
        }

        util = BasisConfigUtil(overlap["basis"])
        overlap_momentum = convert_overlap_to_momentum_basis(overlap)
        expected = overlap_momentum["vector"]
        actual = get_overlap_momentum_interpolator_k_fractions(overlap)(util.nk_points)

        np.testing.assert_array_almost_equal(expected, actual)
