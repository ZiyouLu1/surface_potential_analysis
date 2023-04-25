import unittest

import numpy as np

from surface_potential_analysis.eigenstate.eigenstate_collection_plot import (
    _get_projected_phases,
)


class EigenstateTest(unittest.TestCase):
    def test_get_projected_phases(self) -> None:
        phases = np.array([[1.0, 0, 0], [2.0, -3.0, 9.0], [0, 0, 0], [-1.0, 3.0, 4.0]])
        expected = np.array([1, 2, 0, -1])

        direction = np.array([1, 0, 0])
        actual = _get_projected_phases(phases, direction)
        np.testing.assert_array_equal(expected, actual)

        direction = np.array([2, 0, 0])
        actual = _get_projected_phases(phases, direction)
        np.testing.assert_array_equal(expected, actual)

        direction = np.array([-1, 0, 0])
        actual = _get_projected_phases(phases, direction)
        np.testing.assert_array_equal(-expected, actual)
