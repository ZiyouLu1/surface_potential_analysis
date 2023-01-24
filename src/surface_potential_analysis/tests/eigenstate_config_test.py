import unittest

import numpy as np

from surface_potential_analysis.energy_eigenstate import (
    EigenstateConfig,
    EigenstateConfigUtil,
)


class TestEigenstateConfig(unittest.TestCase):
    def test_inverse_lattuice_points_100(self) -> None:
        config: EigenstateConfig = {
            "mass": 1,
            "resolution": (1, 1, 1),
            "sho_omega": 1,
            "delta_x1": (1, 0),
            "delta_x2": (0, 2),
        }
        util = EigenstateConfigUtil(config)

        self.assertEqual(util.dkx1[0], 2 * np.pi)
        self.assertEqual(util.dkx1[1], 0)
        self.assertEqual(util.dkx2[0], 0)
        self.assertEqual(util.dkx2[1], np.pi)

    def test_inverse_lattuice_points_111(self) -> None:
        config: EigenstateConfig = {
            "mass": 1,
            "resolution": (1, 1, 1),
            "sho_omega": 1,
            "delta_x1": (1, 0),
            "delta_x2": (0.5, np.sqrt(3) / 2),
        }
        util = EigenstateConfigUtil(config)

        self.assertEqual(util.dkx1[0], 2 * np.pi)
        self.assertEqual(util.dkx1[1], -2 * np.pi / np.sqrt(3))
        self.assertEqual(util.dkx2[0], 0)
        self.assertEqual(util.dkx2[1], 4 * np.pi / np.sqrt(3))
