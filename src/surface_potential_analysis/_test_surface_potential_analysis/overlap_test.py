from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import numpy as np

from surface_potential_analysis.basis.build import (
    position_basis_3d_from_shape,
)
from surface_potential_analysis.basis.util import (
    AxisWithLengthBasisUtil,
)
from surface_potential_analysis.overlap.conversion import (
    convert_overlap_to_momentum_basis,
)
from surface_potential_analysis.overlap.interpolation import (
    get_overlap_momentum_interpolator,
    get_overlap_momentum_interpolator_k_fractions,
)

if TYPE_CHECKING:
    from surface_potential_analysis.overlap.overlap import FundamentalPositionOverlap

rng = np.random.default_rng()


class OverlapTest(unittest.TestCase):
    def test_overlap_interpolation(self) -> None:
        shape = (10, 10, 10)
        overlap: FundamentalPositionOverlap[Any, Any, Any] = {
            "basis": position_basis_3d_from_shape(shape),
            "vector": np.array(rng.random(np.prod(shape)), dtype=complex),
        }

        util = AxisWithLengthBasisUtil(overlap["basis"])
        overlap_momentum = convert_overlap_to_momentum_basis(overlap)
        expected = overlap_momentum["vector"]
        actual = get_overlap_momentum_interpolator(overlap)(util.k_points)  # type: ignore[var-annotated,arg-type]

        np.testing.assert_array_almost_equal(expected, actual)

    def test_overlap_interpolation_fractions(self) -> None:
        shape = (10, 10, 10)
        overlap: FundamentalPositionOverlap[Any, Any, Any] = {
            "basis": position_basis_3d_from_shape(shape),
            "vector": np.array(rng.random(np.prod(shape)), dtype=complex),
        }

        util = AxisWithLengthBasisUtil(overlap["basis"])
        overlap_momentum = convert_overlap_to_momentum_basis(overlap)
        expected = overlap_momentum["vector"]
        actual = get_overlap_momentum_interpolator_k_fractions(overlap)(util.nk_points)  # type: ignore[var-annotated,arg-type]

        np.testing.assert_array_almost_equal(expected, actual)
