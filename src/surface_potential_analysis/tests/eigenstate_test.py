import unittest
from typing import Any

import numpy as np

from surface_potential_analysis.basis import (
    ExplicitBasis,
    MomentumBasis,
    TruncatedBasis,
)
from surface_potential_analysis.basis_config import PositionBasisConfigUtil
from surface_potential_analysis.eigenstate.eigenstate import (
    Eigenstate,
    EigenstateWithBasis,
    _convert_explicit_basis_x2_to_position,
    _convert_momentum_basis_x01_to_position,
    flatten_eigenstate,
    stack_eigenstate,
)
from surface_potential_analysis.eigenstate.eigenstate_collection_plot import (
    get_projected_phases,
)


class EigenstateTest(unittest.TestCase):
    def test_get_projected_phases(self) -> None:
        phases = np.array([[1.0, 0, 0], [2.0, -3.0, 9.0], [0, 0, 0], [-1.0, 3.0, 4.0]])
        expected = np.array([1, 2, 0, -1])

        direction = np.array([1, 0, 0])
        actual = get_projected_phases(phases, direction)
        np.testing.assert_array_equal(expected, actual)

        direction = np.array([2, 0, 0])
        actual = get_projected_phases(phases, direction)
        np.testing.assert_array_equal(expected, actual)

        direction = np.array([-1, 0, 0])
        actual = get_projected_phases(phases, direction)
        np.testing.assert_array_equal(-expected, actual)

    def test_random_stack_eigenstate(self) -> None:
        basis = PositionBasisConfigUtil.from_resolution((10, 12, 13))
        util = PositionBasisConfigUtil(basis)
        eigenstate: Eigenstate[Any] = {
            "basis": basis,
            "vector": np.array(np.random.rand(len(util)), dtype=complex),
        }
        stacked_eigenstate = stack_eigenstate(eigenstate)

        np.testing.assert_array_equal(
            eigenstate["vector"],
            stacked_eigenstate["vector"][*util.fundamental_nx_points],
        )

        np.testing.assert_array_equal(
            eigenstate["vector"],
            flatten_eigenstate(stacked_eigenstate)["vector"],
        )

    def test_convert_explicit_basis_x2_to_position_shape(self) -> None:
        basis = PositionBasisConfigUtil.from_resolution((10, 12, 13))
        eigenstate: EigenstateWithBasis[Any, Any, ExplicitBasis[int, Any]] = {
            "basis": (
                basis[0],
                basis[1],
                {
                    "_type": "explicit",
                    "parent": basis[2],
                    "vectors": np.random.rand(9, 13),
                },
            ),
            "vector": np.zeros((10 * 12 * 9)),
        }
        eigenstate_stacked = stack_eigenstate(eigenstate)
        stacked_position = _convert_explicit_basis_x2_to_position(eigenstate_stacked)

        np.testing.assert_array_equal((10, 12, 13), stacked_position["vector"].shape)

    def test_convert_sho_basis_order(self) -> None:
        eigenstate: EigenstateWithBasis[
            TruncatedBasis[Any, MomentumBasis[Any]],
            TruncatedBasis[Any, MomentumBasis[Any]],
            ExplicitBasis[int, Any],
        ] = {
            "basis": (
                {
                    "_type": "truncated",
                    "n": 5,
                    "parent": {
                        "_type": "momentum",
                        "delta_x": np.array([1, 0, 0]),
                        "n": 10,
                    },
                },
                {
                    "_type": "truncated",
                    "n": 6,
                    "parent": {
                        "_type": "momentum",
                        "delta_x": np.array([0, 1, 0]),
                        "n": 10,
                    },
                },
                {
                    "_type": "explicit",
                    "parent": {
                        "_type": "position",
                        "delta_x": np.array([1, 0, 0]),
                        "n": 10,
                    },
                    "vectors": np.random.rand(9, 13),
                },
            ),
            "vector": np.random.rand((5 * 6 * 9)),
        }

        eigenstate_stacked = stack_eigenstate(eigenstate)

        expected = _convert_momentum_basis_x01_to_position(
            _convert_explicit_basis_x2_to_position(eigenstate_stacked)
        )
        actual = _convert_explicit_basis_x2_to_position(
            _convert_momentum_basis_x01_to_position(eigenstate_stacked)
        )
        np.testing.assert_array_almost_equal(expected["vector"], actual["vector"])
