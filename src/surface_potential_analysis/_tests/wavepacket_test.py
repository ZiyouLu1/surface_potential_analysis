from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import numpy as np

from surface_potential_analysis.basis_config.basis_config import (
    MomentumBasisConfigUtil,
    PositionBasisConfigUtil,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    furl_eigenstate,
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.normalization import _get_global_phases

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.wavepacket import (
        MomentumBasisWavepacket,
        PositionBasisWavepacket,
    )

rng = np.random.default_rng()


class WavepacketTest(unittest.TestCase):
    def test_get_global_phases(self) -> None:
        ns0 = rng.integers(1, 10)
        ns1 = rng.integers(1, 10)
        resolution = (
            rng.integers(2, 10),
            rng.integers(1, 10),
            rng.integers(1, 10),
        )
        wavepacket: PositionBasisWavepacket[Any, Any, Any, Any, Any] = {
            "basis": PositionBasisConfigUtil.from_resolution(resolution),
            "vectors": np.zeros((ns0, ns1, np.prod(resolution))),
            "energies": np.zeros((ns0, ns1)),
        }

        idx = rng.integers(0, np.product(resolution).item())
        actual = _get_global_phases(wavepacket, idx)
        np.testing.assert_array_equal(actual.shape, (ns0, ns1))
        np.testing.assert_equal(actual[0, 0], 0)

        idx = 0
        actual = _get_global_phases(wavepacket, idx)
        np.testing.assert_array_equal(actual, np.zeros_like(actual))

    def test_unfurl_wavepacket(self) -> None:
        wavepacket: MomentumBasisWavepacket[int, int, int, int, int] = {
            "basis": MomentumBasisConfigUtil.from_resolution((3, 3, 3)),
            "vectors": np.zeros((3, 2, 27)),
            "energies": np.zeros((3, 2)),
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
        wavepacket: MomentumBasisWavepacket[int, int, int, int, int] = {
            "basis": MomentumBasisConfigUtil.from_resolution((3, 3, 3)),
            "vectors": np.array(rng.random((3, 2, 27)), dtype=complex),
            "energies": np.zeros((3, 2)),
        }
        eigenstate = unfurl_wavepacket(wavepacket)
        actual = furl_eigenstate(eigenstate, (3, 2))

        np.testing.assert_array_almost_equal(wavepacket["vectors"], actual["vectors"])

        np.testing.assert_array_almost_equal(
            wavepacket["basis"][0]["delta_x"], actual["basis"][0]["delta_x"]
        )
        np.testing.assert_array_almost_equal(
            wavepacket["basis"][1]["delta_x"], actual["basis"][1]["delta_x"]
        )
        np.testing.assert_array_almost_equal(
            wavepacket["basis"][2]["delta_x"], actual["basis"][2]["delta_x"]
        )
