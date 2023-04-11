import unittest

import numpy as np

from surface_potential_analysis.basis_config import (
    MomentumBasisConfigUtil,
    PositionBasisConfigUtil,
)
from surface_potential_analysis.util import slice_along_axis


class TestBasisConfig(unittest.TestCase):
    def test_surface_volume_100(self) -> None:
        points = np.random.rand(3)
        basis = PositionBasisConfigUtil.from_resolution(
            (1, 1, 1),
            (
                np.array([points[0], 0, 0]),
                np.array([0, points[1], 0]),
                np.array([0, 0, points[2]]),
            ),
        )
        util = PositionBasisConfigUtil(basis)

        np.testing.assert_almost_equal(util.volume, np.prod(points))
        np.testing.assert_almost_equal(
            util.reciprocal_volume, (2 * np.pi) ** 3 / np.prod(points)
        )

    def test_inverse_lattuice_points_100(self) -> None:
        delta_x = (
            np.array([1, 0, 0]),
            np.array([0, 2, 0]),
            np.array([0, 0, 1]),
        )
        basis = PositionBasisConfigUtil.from_resolution((1, 1, 1), delta_x)
        util = PositionBasisConfigUtil(basis)

        np.testing.assert_array_equal(delta_x[0], util.delta_x0)
        np.testing.assert_array_equal(delta_x[1], util.delta_x1)
        np.testing.assert_array_equal(delta_x[2], util.delta_x2)

        self.assertEqual(util.dk0[0], 2 * np.pi)
        self.assertEqual(util.dk0[1], 0)
        self.assertEqual(util.dk1[0], 0)
        self.assertEqual(util.dk1[1], np.pi)

    def test_inverse_lattuice_points_111(self) -> None:
        delta_x = (
            np.array([1, 0, 0]),
            np.array([0.5, np.sqrt(3) / 2, 0]),
            np.array([0, 0, 1]),
        )
        basis = PositionBasisConfigUtil.from_resolution((1, 1, 1), delta_x)
        util = PositionBasisConfigUtil(basis)

        np.testing.assert_array_equal(delta_x[0], util.delta_x0)
        np.testing.assert_array_equal(delta_x[1], util.delta_x1)
        np.testing.assert_array_equal(delta_x[2], util.delta_x2)

        self.assertEqual(util.dk0[0], 2 * np.pi)
        self.assertEqual(util.dk0[1], -2 * np.pi / np.sqrt(3))
        self.assertEqual(util.dk1[0], 0)
        self.assertEqual(util.dk1[1], 4 * np.pi / np.sqrt(3))

    def test_reciprocal_lattuice(self) -> None:
        delta_x = (
            np.random.rand(3),
            np.random.rand(3),
            np.random.rand(3),
        )
        basis = PositionBasisConfigUtil.from_resolution((1, 1, 1), delta_x)
        util = PositionBasisConfigUtil(basis)

        np.testing.assert_array_almost_equal(delta_x[0], util.delta_x0)
        np.testing.assert_array_almost_equal(delta_x[1], util.delta_x1)
        np.testing.assert_array_almost_equal(delta_x[2], util.delta_x2)

        reciprocal = util.get_reciprocal_basis()
        reciprocal_util = MomentumBasisConfigUtil(reciprocal)

        np.testing.assert_array_almost_equal(reciprocal_util.delta_x0, util.delta_x0)
        np.testing.assert_array_almost_equal(reciprocal_util.delta_x1, util.delta_x1)
        np.testing.assert_array_almost_equal(reciprocal_util.delta_x2, util.delta_x2)

        np.testing.assert_array_almost_equal(reciprocal_util.dk0, util.dk0)
        np.testing.assert_array_almost_equal(reciprocal_util.dk1, util.dk1)
        np.testing.assert_array_almost_equal(reciprocal_util.dk2, util.dk2)

        np.testing.assert_array_almost_equal(reciprocal_util.volume, util.volume)
        np.testing.assert_array_almost_equal(
            reciprocal_util.reciprocal_volume, util.reciprocal_volume
        )

        reciprocal_2 = reciprocal_util.get_reciprocal_basis()
        reciprocal_2_util = PositionBasisConfigUtil(reciprocal_2)

        np.testing.assert_array_almost_equal(reciprocal_2_util.delta_x0, util.delta_x0)
        np.testing.assert_array_almost_equal(reciprocal_2_util.delta_x1, util.delta_x1)
        np.testing.assert_array_almost_equal(reciprocal_2_util.delta_x2, util.delta_x2)

        np.testing.assert_array_almost_equal(reciprocal_2_util.dk0, util.dk0)
        np.testing.assert_array_almost_equal(reciprocal_2_util.dk1, util.dk1)
        np.testing.assert_array_almost_equal(reciprocal_2_util.dk2, util.dk2)

    def test_get_stacked_index(self) -> None:
        delta_x = (
            np.array([1, 0, 0]),
            np.array([0, 2, 0]),
            np.array([0, 0, 1]),
        )
        resolution = (
            np.random.randint(1, 10),
            np.random.randint(1, 10),
            np.random.randint(1, 10),
        )
        basis = PositionBasisConfigUtil.from_resolution(resolution, delta_x)
        util = PositionBasisConfigUtil(basis)
        for i in range(np.prod(resolution)):
            self.assertEqual(i, util.get_flat_index(util.get_stacked_index(i)))

    def test_rotated_basis_111(self) -> None:
        delta_x = (
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        )
        basis = PositionBasisConfigUtil.from_resolution((1, 1, 1), delta_x)
        util = PositionBasisConfigUtil(basis)

        rotated_0 = util.get_rotated_basis(0, delta_x[0])
        np.testing.assert_array_almost_equal(rotated_0[0]["delta_x"], [1, 0, 0])
        np.testing.assert_array_almost_equal(rotated_0[1]["delta_x"], [0, 1, 0])
        np.testing.assert_array_almost_equal(rotated_0[2]["delta_x"], [0, 0, 1])

        rotated_1 = util.get_rotated_basis(0, delta_x[1])
        np.testing.assert_array_almost_equal(rotated_1[0]["delta_x"], [0, 1, 0])
        np.testing.assert_array_almost_equal(rotated_1[1]["delta_x"], [-1, 0, 0])
        np.testing.assert_array_almost_equal(rotated_1[2]["delta_x"], [0, 0, 1])

        rotated_2 = util.get_rotated_basis(0, delta_x[2])
        np.testing.assert_array_almost_equal(rotated_2[0]["delta_x"], [0, 0, 1])
        np.testing.assert_array_almost_equal(rotated_2[1]["delta_x"], [0, 1, 0])
        np.testing.assert_array_almost_equal(rotated_2[2]["delta_x"], [-1, 0, 0])

    def test_rotated_basis(self) -> None:
        delta_x = (
            np.random.rand(3),
            np.random.rand(3),
            np.random.rand(3),
        )
        basis = PositionBasisConfigUtil.from_resolution((1, 1, 1), delta_x)
        util = PositionBasisConfigUtil(basis)

        for i in (0, 1, 2):
            rotated = util.get_rotated_basis(i)  # type:ignore
            np.testing.assert_array_almost_equal(
                rotated[i]["delta_x"], [0, 0, np.linalg.norm(delta_x[i])]
            )
            for j in (0, 1, 2):
                np.testing.assert_almost_equal(
                    np.linalg.norm(rotated[j]["delta_x"]),
                    np.linalg.norm(basis[j]["delta_x"]),
                )

            direction = np.random.rand(3)
            rotated = util.get_rotated_basis(i, direction)  # type:ignore
            np.testing.assert_almost_equal(
                np.dot(rotated[i]["delta_x"], direction),
                np.linalg.norm(direction) * np.linalg.norm(rotated[i]["delta_x"]),
            )
            for j in (0, 1, 2):
                np.testing.assert_almost_equal(
                    np.linalg.norm(rotated[j]["delta_x"]),
                    np.linalg.norm(basis[j]["delta_x"]),
                )

    def test_nx_points_simple(self) -> None:
        delta_x = (
            np.random.rand(3),
            np.random.rand(3),
            np.random.rand(3),
        )
        basis = PositionBasisConfigUtil.from_resolution((2, 2, 2), delta_x)
        util = PositionBasisConfigUtil(basis)

        actual = util.fundamental_nx_points
        expected = [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ]
        np.testing.assert_array_equal(expected, actual)

        resolution = (
            np.random.randint(1, 20),
            np.random.randint(1, 20),
            np.random.randint(1, 20),
        )
        basis = PositionBasisConfigUtil.from_resolution(resolution, delta_x)
        util = PositionBasisConfigUtil(basis)
        actual = util.fundamental_nx_points

        for axis in range(3):
            basis_for_axis = actual[axis].reshape(*resolution)
            for j in range(resolution[axis]):
                slice_j = basis_for_axis[slice_along_axis(j, axis)]
                np.testing.assert_equal(slice_j, j)

    def test_nk_points_simple(self) -> None:
        delta_x = (
            np.random.rand(3),
            np.random.rand(3),
            np.random.rand(3),
        )
        basis = PositionBasisConfigUtil.from_resolution((2, 2, 2), delta_x)
        util = PositionBasisConfigUtil(basis)

        actual = util.fundamental_nk_points
        expected = [
            [0, 0, 0, 0, -1, -1, -1, -1],
            [0, 0, -1, -1, 0, 0, -1, -1],
            [0, -1, 0, -1, 0, -1, 0, -1],
        ]
        np.testing.assert_array_equal(expected, actual)

        resolution = (
            np.random.randint(1, 20),
            np.random.randint(1, 20),
            np.random.randint(1, 20),
        )
        basis = PositionBasisConfigUtil.from_resolution(resolution, delta_x)
        util = PositionBasisConfigUtil(basis)
        actual = util.fundamental_nk_points

        for axis in range(3):
            basis_for_axis = actual[axis].reshape(*resolution)
            expected_for_axis = np.fft.fftfreq(resolution[axis], 1 / resolution[axis])

            for j, expected in enumerate(expected_for_axis):
                slice_j = basis_for_axis[slice_along_axis(j, axis)]
                np.testing.assert_equal(slice_j, expected)

    def test_x_points_100(self) -> None:
        delta_x = (
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        )
        basis = PositionBasisConfigUtil.from_resolution((3, 3, 3), delta_x)
        util = PositionBasisConfigUtil(basis)

        actual = util.fundamental_x_points
        # fmt: off
        expected_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) / 3
        expected_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]) / 3
        expected_z = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]) / 3
        # fmt: on

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

        delta_x = (
            np.array([0, 1, 0], dtype=float),
            np.array([3, 0, 0], dtype=float),
            np.array([0, 0, 5], dtype=float),
        )
        basis = PositionBasisConfigUtil.from_resolution((3, 3, 3), delta_x)
        util = PositionBasisConfigUtil(basis)
        actual = util.fundamental_x_points

        # fmt: off
        expected_x = np.array([0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0]) / 3
        expected_y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) / 3
        expected_z = np.array([0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0]) / 3
        # fmt: on

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

    def test_k_points_100(self) -> None:
        dx = (
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        )
        basis = MomentumBasisConfigUtil.from_resolution((3, 3, 3), dx)
        util = MomentumBasisConfigUtil(basis)

        actual = util.fundamental_nk_points
        # fmt: off
        expected_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        expected_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        expected_z = np.array([0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0])
        # fmt: on

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

        delta_x = (
            np.array([0, 1, 0], dtype=float),
            np.array([3, 0, 0], dtype=float),
            np.array([0, 0, 5], dtype=float),
        )
        basis = MomentumBasisConfigUtil.from_resolution((3, 3, 3), delta_x)
        util = MomentumBasisConfigUtil(basis)
        actual = util.fundamental_nk_points

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

        actual_k = util.fundamental_k_points
        expected_kx = 2 * np.pi * expected_y / 3.0
        expected_ky = 2 * np.pi * expected_x / 1.0
        expected_kz = 2 * np.pi * expected_z / 5.0

        print(util.dk0 / (2 * np.pi), util.dk1 / (2 * np.pi), util.dk2 / (2 * np.pi))

        np.testing.assert_array_almost_equal(expected_kx, actual_k[0])
        np.testing.assert_array_almost_equal(expected_ky, actual_k[1])
        np.testing.assert_array_almost_equal(expected_kz, actual_k[2])
