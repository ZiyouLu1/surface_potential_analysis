import random
import unittest

import numpy as np

from surface_potential_analysis.energy_data import EnergyInterpolation
from surface_potential_analysis.energy_eigenstate import generate_sho_config_minimum


class TestSHOConfig(unittest.TestCase):
    def test_generate_sho_config(self) -> None:
        sho_omega = random.randrange(1, 100)
        z_points = np.linspace(-50 * 1, 50 * 1, 101)
        points = np.ones(shape=(100, 100, 101))
        points[0, 0] = 0.5 * 1 * (sho_omega * z_points) ** 2
        interpolation: EnergyInterpolation = {
            "points": points.tolist(),
            "dz": 1,
        }
        out, _ = generate_sho_config_minimum(interpolation, mass=1)
        self.assertAlmostEqual(sho_omega, out)
