import unittest
from typing import Any

import numpy as np

from surface_potential_analysis.basis_config import PositionBasisConfigUtil
from surface_potential_analysis.wavepacket import (
    PositionBasisWavepacket,
    get_global_phases,
)


class WavepacketTest(unittest.TestCase):
    def test_get_global_phases(self) -> None:
        ns0 = np.random.randint(1, 10)
        ns1 = np.random.randint(1, 10)
        resolution = (
            np.random.randint(2, 10),
            np.random.randint(1, 10),
            np.random.randint(1, 10),
        )
        wavepacket: PositionBasisWavepacket[Any, Any, Any, Any, Any] = {
            "basis": PositionBasisConfigUtil.from_resolution(resolution),
            "vectors": np.zeros((ns0, ns1, np.prod(resolution))),
        }

        idx = np.random.randint(0, np.product(resolution).item())
        actual = get_global_phases(wavepacket, idx)
        np.testing.assert_array_equal(actual.shape, (ns0, ns1))
        np.testing.assert_equal(actual[0, 0], 0)

        idx = 0
        actual = get_global_phases(wavepacket, idx)
        np.testing.assert_array_equal(actual, np.zeros_like(actual))
