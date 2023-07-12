from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalPositionAxis
from surface_potential_analysis.basis.build import (
    momentum_basis_3d_from_resolution,
    position_basis_3d_from_shape,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    furl_eigenstate,
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.localization import (
    _get_global_phases,
    _get_zero_point_locations,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_wavepacket_sample_fractions,
)

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.wavepacket import (
        MomentumBasisWavepacket3d,
        PositionBasisWavepacket3d,
    )

rng = np.random.default_rng()


class WavepacketTest(unittest.TestCase):
    def test_get_zero_point_locations(self) -> None:
        ns0 = rng.integers(3, 10)
        ns1 = rng.integers(1, 10)
        resolution = (
            rng.integers(2, 10),
            rng.integers(1, 10),
        )
        basis = (
            FundamentalPositionAxis(np.array([1, 0]), resolution[0]),
            FundamentalPositionAxis(np.array([0, 1]), resolution[1]),
        )
        shape = (ns0, ns1)
        locations = _get_zero_point_locations(basis, shape, 0)
        assert locations

    def test_get_global_phases(self) -> None:
        ns0 = rng.integers(1, 10)
        ns1 = rng.integers(1, 10)
        resolution = (
            rng.integers(2, 10),
            rng.integers(1, 10),
            rng.integers(1, 10),
        )
        wavepacket: PositionBasisWavepacket3d[Any, Any, Any, Any, Any] = {
            "basis": position_basis_3d_from_shape(resolution),
            "vectors": np.zeros((ns0 * ns1, np.prod(resolution))),
            "eigenvalues": np.zeros(ns0 * ns1),
            "shape": (ns0, ns1, 1),
        }

        idx = rng.integers(0, np.product(resolution).item())
        actual = _get_global_phases(wavepacket, idx)
        np.testing.assert_array_equal(actual.shape, (ns0 * ns1,))
        np.testing.assert_equal(actual[0], 0)

        idx = 0
        actual = _get_global_phases(wavepacket, idx)
        np.testing.assert_array_equal(actual, np.zeros_like(actual))

        idx_array = rng.integers(0, np.product(resolution).item(), size=(10, 10, 11))
        actual_large = _get_global_phases(wavepacket, idx_array)
        np.testing.assert_array_equal(actual_large.shape, (ns0 * ns1, *idx_array.shape))
        np.testing.assert_equal(actual_large[0], 0)
        np.testing.assert_equal(actual_large[:, idx_array == 0], 0)

    def test_unfurl_wavepacket(self) -> None:
        wavepacket: MomentumBasisWavepacket3d[int, int, int, int, int] = {
            "basis": momentum_basis_3d_from_resolution((3, 3, 3)),
            "shape": (3, 2, 1),
            "vectors": np.zeros((3, 2, 27)),
            "eigenvalues": np.zeros((3, 2)),
        }
        wavepacket["vectors"][0][0][0] = 1
        wavepacket["vectors"][1][0][0] = 2
        wavepacket["vectors"][2][0][0] = 3
        wavepacket["vectors"][0][1][0] = 4
        wavepacket["vectors"][1][1][0] = 5
        wavepacket["vectors"][2][1][0] = 6

        expected = np.zeros(162)
        expected[np.ravel_multi_index((0, 0, 0), (9, 6, 3))] = 1
        expected[np.ravel_multi_index((1, 0, 0), (9, 6, 3))] = 2
        expected[np.ravel_multi_index((8, 0, 0), (9, 6, 3))] = 3
        expected[np.ravel_multi_index((0, 5, 0), (9, 6, 3))] = 4
        expected[np.ravel_multi_index((1, 5, 0), (9, 6, 3))] = 5
        expected[np.ravel_multi_index((8, 5, 0), (9, 6, 3))] = 6

        eigenstate = unfurl_wavepacket(wavepacket)
        np.testing.assert_array_equal(eigenstate["vector"], expected / np.sqrt(2 * 3))

    def test_furl_eigenstate(self) -> None:
        wavepacket: MomentumBasisWavepacket3d[int, int, int, int, int] = {
            "basis": momentum_basis_3d_from_resolution((3, 3, 3)),
            "vectors": np.array(rng.random((3, 2, 27)), dtype=complex),
            "shape": (3, 2, 1),
            "eigenvalues": np.zeros((3, 2)),
        }
        eigenstate = unfurl_wavepacket(wavepacket)
        actual = furl_eigenstate(eigenstate, (3, 2, 1))

        np.testing.assert_array_almost_equal(wavepacket["vectors"], actual["vectors"])

        np.testing.assert_array_almost_equal(
            wavepacket["basis"][0].delta_x, actual["basis"][0].delta_x
        )
        np.testing.assert_array_almost_equal(
            wavepacket["basis"][1].delta_x, actual["basis"][1].delta_x
        )
        np.testing.assert_array_almost_equal(
            wavepacket["basis"][2].delta_x, actual["basis"][2].delta_x
        )

    def test_get_wavepacket_sample_fractions(self) -> None:
        shape = tuple(rng.integers(1, 10, size=rng.integers(1, 5)))

        actual = get_wavepacket_sample_fractions(shape)
        meshgrid = np.meshgrid(
            *[np.fft.fftfreq(s, 1) for s in shape],
            indexing="ij",
        )
        expected = np.array([x.ravel() for x in meshgrid])
        np.testing.assert_array_almost_equal(expected, actual)
