import random
import unittest

import numpy as np

from energy_data import EnergyInterpolation
from sho_config import generate_sho_config_minimum


class TestSurfaceHamiltonian(unittest.TestCase):
    def test_generate_sho_config(self) -> None:
        sho_omega = random.randrange(1, 100)
        z_points = np.linspace(-50 * 1, 50 * 1, 101)
        points = np.ones(shape=(100, 100, 101))
        points[0, 0] = 0.5 * 1 * (sho_omega * z_points) ** 2
        interpolation: EnergyInterpolation = {
            "points": points.tolist(),
            "delta_x": 1,
            "delta_y": 1,
            "dz": 1,
        }
        out = generate_sho_config_minimum(interpolation, mass=1)
        self.assertAlmostEqual(sho_omega, out["sho_omega"])
