from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Literal

import numpy as np
from surface_potential_analysis.basis.basis import FundamentalPositionBasis

from surface_dynamics_simulation.tunnelling_simulation.isf import (
    _calculate_mean_locations,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis_config.basis_config import (
        FundamentalPositionBasisConfig,
    )


class TunnellingMatrixTest(unittest.TestCase):
    def test_calculate_mean_distances(self) -> None:
        basis: FundamentalPositionBasisConfig[Literal[1], Literal[1], Literal[1]] = (
            FundamentalPositionBasis(np.array([3, 0, 0]), 1),
            FundamentalPositionBasis(np.array([0, 3, 0]), 1),
            FundamentalPositionBasis(np.array([0, 0, 1]), 1),
        )
        actual = _calculate_mean_locations((1, 1, 6), basis)
        expected = np.array(
            [
                [[[0.0, 1.0, 0.0, 0.0, 1.0, 1.0]]],
                [[[0.0, 1.0, 0.0, 0.0, 1.0, 1.0]]],
                [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            ]
        )
        np.testing.assert_array_equal(actual, expected)
