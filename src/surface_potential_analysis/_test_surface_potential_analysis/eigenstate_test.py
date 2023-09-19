from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalBasis
from surface_potential_analysis.axis.block_fraction_axis import (
    ExplicitBlockFractionAxis,
)
from surface_potential_analysis.axis.stacked_axis import StackedBasis
from surface_potential_analysis.stacked_basis.build import (
    position_basis_3d_from_shape,
)
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    _get_projected_bloch_phases,
)

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        EigenstateColllection,
    )


class EigenstateTest(unittest.TestCase):
    def test_get_projected_phases(self) -> None:
        phases = np.array([[1.0, 0, 0], [2.0, -3.0, 9.0], [0, 0, 0], [-1.0, 3.0, 4.0]])
        expected = np.array([2 * np.pi, 4 * np.pi, 0.0, -2 * np.pi])
        collection: EigenstateColllection[Any, Any] = {
            "basis": position_basis_3d_from_shape((1, 1, 1)),
            "list_basis": StackedBasis(
                ExplicitBlockFractionAxis(phases), FundamentalBasis(0)
            ),
            "eigenvalues": np.array([]),
            "data": np.array([]),
        }

        direction = np.array([1, 0, 0])
        actual = _get_projected_bloch_phases(collection, direction)
        np.testing.assert_array_equal(expected, actual)

        direction = np.array([2, 0, 0])
        actual = _get_projected_bloch_phases(collection, direction)
        np.testing.assert_array_equal(expected, actual)

        direction = np.array([-1, 0, 0])
        actual = _get_projected_bloch_phases(collection, direction)
        np.testing.assert_array_equal(-expected, actual)
