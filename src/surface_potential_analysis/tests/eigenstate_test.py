import unittest
from typing import Any

import hamiltonian_generator
import numpy as np

from surface_potential_analysis.basis import (
    ExplicitBasis,
    MomentumBasis,
    TruncatedBasis,
)
from surface_potential_analysis.basis_config import (
    BasisConfigUtil,
    PositionBasisConfigUtil,
)
from surface_potential_analysis.eigenstate.eigenstate import (
    Eigenstate,
    EigenstateWithBasis,
    StackedEigenstateWithBasis,
    _convert_explicit_basis_x2_to_position,
    _convert_momentum_basis_x01_to_position,
    convert_sho_eigenstate_to_position_basis,
    flatten_eigenstate,
    stack_eigenstate,
)
from surface_potential_analysis.eigenstate.eigenstate_collection_plot import (
    get_projected_phases,
)
from surface_potential_analysis.sho_basis import (
    SHOBasisConfig,
    infinate_sho_basis_from_config,
)


def get_random_sho_eigenstate(
    resolution: tuple[int, int, int], fundamental_resolution: tuple[int, int, int]
) -> EigenstateWithBasis[
    TruncatedBasis[Any, MomentumBasis[Any]],
    TruncatedBasis[Any, MomentumBasis[Any]],
    ExplicitBasis[int, Any],
]:
    return {
        "basis": (
            {
                "_type": "truncated",
                "n": resolution[0],
                "parent": {
                    "_type": "momentum",
                    "delta_x": np.array([1, 0, 0]),
                    "n": fundamental_resolution[0],
                },
            },
            {
                "_type": "truncated",
                "n": resolution[1],
                "parent": {
                    "_type": "momentum",
                    "delta_x": np.array([0, 1, 0]),
                    "n": fundamental_resolution[1],
                },
            },
            {
                "_type": "explicit",
                "parent": {
                    "_type": "position",
                    "delta_x": np.array([0, 0, 1]),
                    "n": fundamental_resolution[2],
                },
                "vectors": np.zeros((resolution[2], fundamental_resolution[2])),
            },
        ),
        "vector": np.array(np.random.rand(np.prod(resolution)), dtype=complex),
    }


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
        eigenstate = get_random_sho_eigenstate((10, 12, 9), (10, 12, 13))

        eigenstate_stacked = stack_eigenstate(eigenstate)
        stacked_position = _convert_explicit_basis_x2_to_position(eigenstate_stacked)

        np.testing.assert_array_equal((10, 12, 13), stacked_position["vector"].shape)

    def test_convert_sho_basis_order(self) -> None:
        eigenstate = get_random_sho_eigenstate((5, 6, 9), (10, 10, 10))
        stacked = stack_eigenstate(eigenstate)

        expected = _convert_momentum_basis_x01_to_position(
            _convert_explicit_basis_x2_to_position(stacked)
        )
        actual = _convert_explicit_basis_x2_to_position(
            _convert_momentum_basis_x01_to_position(stacked)
        )
        np.testing.assert_array_almost_equal(expected["vector"], actual["vector"])

    def test_convert_sho_basis_explicit(self) -> None:
        nx = np.random.randint(2, 10)
        fundamental_nx = np.random.randint(nx, nx + 10)
        ny = np.random.randint(2, 10)
        fundamental_ny = np.random.randint(ny, ny + 10)

        nz = np.random.randint(2, 10)
        fundamental_nz = np.random.randint(nz, nz + 10)

        eigenstate = get_random_sho_eigenstate(
            (nx, ny, nz), (fundamental_nx, fundamental_ny, fundamental_nz)
        )
        stacked = stack_eigenstate(eigenstate)

        for i in range(nz):
            vector = np.zeros((fundamental_nz))
            vector[i] = 1
            stacked["basis"][2]["vectors"][i] = vector

        position_eigenstate = _convert_explicit_basis_x2_to_position(stacked)

        expected = np.zeros((nx, ny, fundamental_nz), dtype=complex)
        expected[:, :, :nz] = stacked["vector"]

        np.testing.assert_array_equal(expected, position_eigenstate["vector"])

    def test_convert_sho_eigenstate_rust_simple(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = get_random_sho_eigenstate(resolution, (10, 10, 100))
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, 0]),
        }

        eigenstate["basis"] = (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            infinate_sho_basis_from_config(
                eigenstate["basis"][2]["parent"], config, resolution[2]
            ),
        )

        util = BasisConfigUtil(eigenstate["basis"])

        for i in range(resolution[2]):
            vector = np.zeros_like(eigenstate["vector"])
            vector[np.ravel_multi_index((0, 0, i), resolution)] = 1
            eigenstate["vector"] = vector

            actual = hamiltonian_generator.get_eigenstate_wavefunction(
                resolution,
                (util.delta_x0.item(0), 0),
                (util.delta_x1.item(0), util.delta_x1.item(1)),
                config["mass"],
                config["sho_omega"],
                0,
                0,
                eigenstate["vector"].tolist(),
                util.fundamental_x_points.T.tolist(),
            )

            expected = convert_sho_eigenstate_to_position_basis(eigenstate)

            np.testing.assert_allclose(expected["vector"], actual)

    def test_convert_sho_eigenstate_rust(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = get_random_sho_eigenstate(resolution, (10, 10, 100))
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, 0]),
        }

        eigenstate["basis"] = (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            infinate_sho_basis_from_config(
                eigenstate["basis"][2]["parent"], config, resolution[2]
            ),
        )

        util = BasisConfigUtil(eigenstate["basis"])

        actual = hamiltonian_generator.get_eigenstate_wavefunction(
            resolution,
            (util.delta_x0.item(0), 0),
            (util.delta_x1.item(0), util.delta_x1.item(1)),
            config["mass"],
            config["sho_omega"],
            0,
            0,
            eigenstate["vector"].tolist(),
            util.fundamental_x_points.T.tolist(),
        )

        expected = convert_sho_eigenstate_to_position_basis(eigenstate)

        np.testing.assert_allclose(expected["vector"], actual)
